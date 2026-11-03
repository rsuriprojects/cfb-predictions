"""The model: Elo ratings, seeded each season from roster strength.

Every input works on raw CollegeFootballData responses, so the backtest and the
daily job run the exact same code.

How a game is predicted:
  1. Elo. Every team carries a rating that moves after each game, more for
     bigger wins. FCS teams are rated too, starting well below FBS.
  2. Preseason. Ratings carry over between seasons, but only partly: they are
     pulled back toward a prior built from the roster. That prior uses 247's
     team talent composite (how good the players are), returning production
     (how much of last year's output is back), and recruiting (who just
     arrived). This is where rosters matter most, before teams have played.
  3. A logistic regression turns the Elo gap, home field and the roster gaps
     into a win probability. Its weights are fit on past seasons only.
"""
import math

import numpy as np

BASE = 1500          # average FBS team
FCS_BASE = 1150      # FCS teams start here every season
HFA = 55             # home-field advantage in Elo points (~2.5 points)

# Tuned by fit_model.py on the training seasons; see model.json.
DEFAULTS = {"K": 30.0, "CARRY": 0.6, "W_TALENT": 60.0, "W_RET": 40.0, "W_REC": 20.0}


# ---------------------------------------------------------------- inputs
def index(rows, key="team", value=None):
    """{team: value} from a list of CFBD records."""
    return {r[key]: r[value] for r in rows if r.get(value) is not None}


def zscores(d, teams):
    """Standardize a {team: number} map over `teams`; missing teams get 0."""
    xs = [d[t] for t in teams if t in d]
    if len(xs) < 10:
        return {t: 0.0 for t in teams}
    mu, sd = float(np.mean(xs)), float(np.std(xs)) or 1.0
    return {t: (d[t] - mu) / sd if t in d else 0.0 for t in teams}


def roster(talent, returning, recruiting, fbs):
    """Per-team roster z-scores for one season: talent, returning, recruiting."""
    teams = [t["school"] for t in fbs]
    return {
        "talent": zscores(index(talent, value="talent"), teams),
        "ret": zscores(index(returning, value="percentPPA"), teams),
        "rec": zscores(index(recruiting, value="points"), teams),
    }


def is_final(g):
    return g.get("completed") and g.get("homePoints") is not None \
        and g.get("awayPoints") is not None


# ---------------------------------------------------------------- Elo
def preseason(last, ros, fbs, p):
    """Start-of-season ratings: last season's, pulled toward the roster prior."""
    elo = {}
    for t in fbs:
        name = t["school"]
        prior = BASE + p["W_TALENT"] * ros["talent"].get(name, 0) \
                     + p["W_RET"] * ros["ret"].get(name, 0) \
                     + p["W_REC"] * ros["rec"].get(name, 0)
        old = last.get(name)
        elo[name] = prior if old is None else prior + p["CARRY"] * (old - prior)
    return elo


def rating(elo, team, cls):
    if team not in elo:
        elo[team] = BASE if cls == "fbs" else FCS_BASE
    return elo[team]


def elo_gap(elo, g):
    """Home rating minus away rating, including home field."""
    rh = rating(elo, g["homeTeam"], g.get("homeClassification"))
    ra = rating(elo, g["awayTeam"], g.get("awayClassification"))
    return rh - ra + (0 if g.get("neutralSite") else HFA)


def update(elo, g, p):
    """Move both teams' ratings after a final. Bigger wins move them more."""
    gap = elo_gap(elo, g)
    exp = 1 / (1 + 10 ** (-gap / 400))
    mov = g["homePoints"] - g["awayPoints"]
    act = 1.0 if mov > 0 else 0.0 if mov < 0 else 0.5
    # 538-style margin multiplier, damped when the favorite wins big.
    winner_gap = gap if mov > 0 else -gap
    mult = math.log(abs(mov) + 1) * 2.2 / (winner_gap * 0.001 + 2.2)
    shift = p["K"] * mult * (act - exp)
    elo[g["homeTeam"]] += shift
    elo[g["awayTeam"]] -= shift


# ---------------------------------------------------------------- efficiency
# Elo sees only the final score. These ratings see every play: points added per
# play (PPA, garbage time removed), on offense and defense, adjusted for the
# opponent. A team's offense in a game is expected to be
#   EFF_MEAN + its offense rating + the opponent's defense rating
# (defense rating = PPA it allows above average, so lower is better), and both
# ratings move toward what actually happened.
EFF_MEAN = 0.13          # average offensive PPA in FBS games, 2015-2025
FCS_EFF = 0.15           # FCS teams start this much worse on each side


def ppa_index(ppa):
    """{(game id, team): (offense PPA, defense PPA)} from CFBD /ppa/games rows."""
    out = {}
    for r in ppa or []:
        o, d = (r.get("offense") or {}).get("overall"), (r.get("defense") or {}).get("overall")
        if o is not None and d is not None:
            out[(r["gameId"], r["team"])] = (o, d)
    return out


def eff_rating(eff, team, cls):
    if team not in eff:
        eff[team] = [0.0, 0.0] if cls == "fbs" else [-FCS_EFF, FCS_EFF]
    return eff[team]


def eff_update(eff, g, idx, p):
    """Move both teams' offense/defense ratings after a game with PPA data."""
    h, a = g["homeTeam"], g["awayTeam"]
    rh = eff_rating(eff, h, g.get("homeClassification"))
    ra = eff_rating(eff, a, g.get("awayClassification"))
    # Each offense's PPA, from its own row or else from the opponent's defense.
    hrow, arow = idx.get((g["id"], h)), idx.get((g["id"], a))
    h_off = hrow[0] if hrow else (arow[1] if arow else None)
    a_off = arow[0] if arow else (hrow[1] if hrow else None)
    for off, opp, obs in ((rh, ra, h_off), (ra, rh, a_off)):
        if obs is None:
            continue
        err = obs - (EFF_MEAN + off[0] + opp[1])
        off[0] += p["K_EFF"] * err
        opp[1] += p["K_EFF"] * err


def eff_gap(eff, g):
    """Home net efficiency minus away (offense rating minus defense rating)."""
    rh = eff_rating(eff, g["homeTeam"], g.get("homeClassification"))
    ra = eff_rating(eff, g["awayTeam"], g.get("awayClassification"))
    return (rh[0] - rh[1]) - (ra[0] - ra[1])


# ---------------------------------------------------------------- features
FEATURES = ["elo_gap", "talent_gap", "talent_gap_early", "ret_gap_early", "eff_gap"]


def features(elo, ros, g, eff=None):
    """Inputs to the regression for one game, using only pre-game information.

    Roster gaps get an extra "early" copy that fades out by week 6: before
    teams have played, the roster says a lot; by midseason Elo has caught up.
    """
    h, a = g["homeTeam"], g["awayTeam"]
    early = max(0.0, 1 - (g["week"] - 1) / 5) if g.get("seasonType") == "regular" else 0.0
    t = ros["talent"].get(h, -2.5 if g.get("homeClassification") != "fbs" else 0) - \
        ros["talent"].get(a, -2.5 if g.get("awayClassification") != "fbs" else 0)
    r = ros["ret"].get(h, 0) - ros["ret"].get(a, 0)
    e = eff_gap(eff, g) * 10 if eff is not None else 0.0
    return [elo_gap(elo, g) / 100, t, t * early, r * early, e]


def prob(weights, x):
    z = weights["bias"] + sum(w * v for w, v in zip(weights["coef"], x))
    return 1 / (1 + math.exp(-z))


# ---------------------------------------------------------------- betting favorite
def ml_prob(ml):
    ml = float(ml)
    return 100 / (ml + 100) if ml > 0 else -ml / (-ml + 100)


def market(line_rec):
    """(home win probability, source) from a CFBD lines record, or (None, None).

    Moneylines first, with each book's margin removed and books averaged. If no
    book posted a moneyline, the median spread, at about 14 points per standard
    deviation of college margins.
    """
    probs, spreads = [], []
    for l in line_rec.get("lines", []) if line_rec else []:
        h, a = l.get("homeMoneyline"), l.get("awayMoneyline")
        if h not in (None, 0) and a not in (None, 0):
            ph, pa = ml_prob(h), ml_prob(a)
            probs.append(ph / (ph + pa))
        if l.get("spread") is not None:
            spreads.append(float(l["spread"]))
    if probs:
        return float(np.mean(probs)), "moneyline"
    if spreads:
        s = float(np.median(spreads))           # negative = home favored
        return 0.5 * (1 + math.erf(-s / 14 / math.sqrt(2))), "spread"
    return None, None


# ---------------------------------------------------------------- one season
def run_season(games, ros, fbs, last, p, lines=None, weights=None, collect=None,
               ppa=None, last_eff=None, eff_out=None):
    """Walk a season in order, predicting each game before updating on it.

    Returns end-of-season Elo ratings. If `collect` is a list, appends one row
    per game: features, outcome, and the market line if `lines` is given.
    With `ppa` (CFBD /ppa/games rows), efficiency ratings are kept too, carried
    from `last_eff` at CARRY_EFF; `eff_out`, if a dict, gets the final ones.
    """
    elo = preseason(last, ros, fbs, p)
    eff = None
    if ppa is not None:
        eff = {t: [p["CARRY_EFF"] * o, p["CARRY_EFF"] * d]
               for t, (o, d) in (last_eff or {}).items()}
        idx = ppa_index(ppa)
    by_id = {l["id"]: l for l in (lines or [])}
    order = sorted(games, key=lambda g: (g.get("startDate") or "", g["id"]))
    for g in order:
        if collect is not None:
            x = features(elo, ros, g, eff)
            row = {"id": g["id"], "season": g["season"], "week": g["week"],
                   "seasonType": g.get("seasonType"), "x": x,
                   "fbs_both": g.get("homeClassification") == "fbs"
                               and g.get("awayClassification") == "fbs",
                   "final": bool(is_final(g))}
            if is_final(g) and g["homePoints"] != g["awayPoints"]:
                row["home_won"] = g["homePoints"] > g["awayPoints"]
            if lines is not None:
                row["p_market"], _ = market(by_id.get(g["id"]))
            if weights:
                row["p_home"] = prob(weights, x)
            collect.append(row)
        if is_final(g):
            update(elo, g, p)
            if eff is not None:
                eff_update(eff, g, idx, p)
    if eff_out is not None and eff is not None:
        eff_out.clear()
        eff_out.update({t: [round(o, 4), round(d, 4)] for t, (o, d) in eff.items()})
    return elo
