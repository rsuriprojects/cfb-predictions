"""Daily college football picks and grading. Runs standalone in GitHub Actions.

Each run:
  1. Rebuilds this season's ratings from last season's (ratings/<prev>.json)
     and every final so far.
  2. Logs a pick for every game in the next 8 days that hasn't kicked off and
     isn't logged yet, with the betting favorite next to it when a line is up.
     Logged picks are never changed.
  3. Grades this season's logged games that have finished, into
     results/<season>.csv. Each season has its own file, so a new season
     starts at 0-0 and past seasons stay as they were.
  4. Writes upcoming.json (the rest of the schedule, plus current ratings) and
     ratings/<season>.json, which seeds next season.

API calls: 2 a day (games and lines). Roster data is fetched once per season
and saved to data/roster_<season>.json. The key is read from CFBD_API_KEY.
"""
import csv
import json
import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import cfbd
import model

NOW = datetime.now(timezone.utc)
if os.environ.get("TEST_NOW"):
    NOW = datetime.fromisoformat(os.environ["TEST_NOW"])
SEASON = int(os.environ.get("SEASON") or (NOW.year if NOW.month >= 8 else NOW.year - 1))
LOOKAHEAD = timedelta(days=8)
ET = ZoneInfo("America/New_York")
LOCAL = os.environ.get("DATA")      # local testing: read saved responses instead of calling


def fetch(name, path, **params):
    if LOCAL:
        return json.load(open(f"{LOCAL}/{name}_{SEASON}.json"))
    return cfbd.get(path, year=SEASON, **params)


def load_json(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def save_json(path, data, indent=1):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent)


def kickoff(g):
    return datetime.fromisoformat(g["startDate"].replace("Z", "+00:00"))


def week_key(g):
    return f"{g['season']}-{'post' if g.get('seasonType') == 'postseason' else 'w%02d' % g['week']}"


def roster_inputs():
    """Talent, returning production, recruiting and the FBS list, once a season."""
    path = f"data/roster_{SEASON}.json"
    saved = load_json(path, None)
    if saved:
        return saved
    saved = {
        "talent": fetch("talent", "/talent"),
        "returning": fetch("returning", "/player/returning"),
        "recruiting": fetch("recruiting", "/recruiting/teams"),
        "fbs": [{k: t.get(k) for k in ("school", "abbreviation", "conference", "color")}
                for t in fetch("fbs", "/teams/fbs")],
    }
    save_json(path, saved)
    print(f"saved {SEASON} roster data to {path}")
    return saved


def main():
    m = load_json("model.json", None)
    params, weights = m["params"], m["weights"]
    last = load_json(f"ratings/{SEASON - 1}.json", {})
    if not last:
        print(f"WARNING: no ratings/{SEASON - 1}.json; starting from rosters alone")

    r = roster_inputs()
    ros = model.roster(r["talent"], r["returning"], r["recruiting"], r["fbs"])
    games = fetch("games", "/games", seasonType="both", classification="fbs")
    lines = fetch("lines", "/lines", seasonType="both")
    print(f"{SEASON}: {len(games)} games, {sum(map(model.is_final, games))} final")

    # Pre-game state for every game, walking the season in order.
    rows = []
    elo = model.run_season(games, ros, r["fbs"], last, params, lines=lines,
                           weights=weights, collect=rows)
    pre = {row["id"]: row for row in rows}
    by_id = {g["id"]: g for g in games}

    # ---- log picks
    new = 0
    for g in sorted(games, key=lambda g: (g["startDate"], g["id"])):
        if not g.get("startDate"):
            continue
        k = kickoff(g)
        started = k <= NOW or model.is_final(g)
        path = f"predictions/{week_key(g)}.json"
        logged = load_json(path, [])
        mine = next((p for p in logged if p["id"] == g["id"]), None)
        x = pre[g["id"]]

        if mine is None:
            if started:
                # A game from before this site existed. Its pick is computed
                # only from what was known before kickoff, and flagged.
                if not model.is_final(g):
                    continue
                backfilled = True
            elif k - NOW <= LOOKAHEAD:
                backfilled = False
            else:
                continue
            mine = {
                "id": g["id"],
                "logged_at_utc": NOW.isoformat(),
                "game_date": str(k.astimezone(ET).date()),
                "kickoff_utc": g["startDate"],
                "season": g["season"], "week": g["week"],
                "season_type": g.get("seasonType"),
                "away": g["awayTeam"], "home": g["homeTeam"],
                "fbs_both": x["fbs_both"],
                "neutral": bool(g.get("neutralSite")),
                "p_home": round(x["p_home"], 4),
                "pick": g["homeTeam"] if x["p_home"] > 0.5 else g["awayTeam"],
            }
            if backfilled:
                mine["backfilled"] = True
            logged.append(mine)
            new += 1

        # The favorite can be added any time before kickoff; lines often go up
        # after the pick is logged. Never after kickoff.
        if "fav" not in mine and x.get("p_market") not in (None, 0.5) and \
                (not started or mine.get("backfilled")):
            pm = x["p_market"]
            mine["fav"] = g["homeTeam"] if pm > 0.5 else g["awayTeam"]
            mine["fav_prob"] = round(max(pm, 1 - pm), 4)

        logged.sort(key=lambda p: (p["kickoff_utc"], p["id"]))
        save_json(path, logged, indent=2)
    print(f"logged {new} new picks")

    # ---- grade (this season only; finished seasons are left alone)
    graded = []
    for path in sorted(os.listdir("predictions")) if os.path.isdir("predictions") else []:
        if not path.startswith(f"{SEASON}-"):
            continue
        for p in load_json(f"predictions/{path}", []):
            g = by_id.get(p["id"])
            if not g or not model.is_final(g) or g["homePoints"] == g["awayPoints"]:
                continue
            winner = g["homeTeam"] if g["homePoints"] > g["awayPoints"] else g["awayTeam"]
            graded.append({
                "game_date": p["game_date"], "season": p["season"], "week": p["week"],
                "season_type": p["season_type"],
                "away": p["away"], "home": p["home"],
                "pick": p["pick"], "p_home": p["p_home"],
                "conf": round(max(p["p_home"], 1 - p["p_home"]), 4),
                "away_pts": g["awayPoints"], "home_pts": g["homePoints"],
                "winner": winner,
                "correct": p["pick"] == winner,
                "fbs_both": p["fbs_both"],
                "backfilled": bool(p.get("backfilled")),
                "fav": p.get("fav", ""),
                "fav_prob": p.get("fav_prob", ""),
                "fav_correct": (p["fav"] == winner) if p.get("fav") else "",
            })
    graded.sort(key=lambda x: (x["game_date"], x["home"]))
    if graded:
        os.makedirs("results", exist_ok=True)
        with open(f"results/{SEASON}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(graded[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(graded)
        fbs = [x for x in graded if x["fbs_both"]]
        both = [x for x in fbs if x["fav"]]
        print(f"graded {len(graded)}; FBS vs FBS "
              f"{sum(x['correct'] for x in fbs)}/{len(fbs)}"
              f" = {sum(x['correct'] for x in fbs) / max(len(fbs), 1):.1%}")
        if both:
            print(f"  vs favorite on {len(both)}: model {sum(x['correct'] for x in both)},"
                  f" favorite {sum(x['fav_correct'] for x in both)}")
    # The site's season menu, newest first.
    seasons = sorted((int(f[:-4]) for f in os.listdir("results") if f[:-4].isdigit()),
                     reverse=True) if os.path.isdir("results") else []
    save_json("results/index.json", seasons)

    # ---- upcoming + ratings
    upcoming = []
    for g in games:
        if model.is_final(g) or not g.get("startDate") or kickoff(g) <= NOW:
            continue
        x = pre[g["id"]]
        upcoming.append({
            "id": g["id"],
            "game_date": str(kickoff(g).astimezone(ET).date()),
            "kickoff_utc": g["startDate"],
            "week": g["week"], "season_type": g.get("seasonType"),
            "away": g["awayTeam"], "home": g["homeTeam"],
            "fbs_both": x["fbs_both"], "neutral": bool(g.get("neutralSite")),
            "p_home": round(x["p_home"], 4),
            "pick": g["homeTeam"] if x["p_home"] > 0.5 else g["awayTeam"],
        })
    upcoming.sort(key=lambda u: (u["kickoff_utc"], u["id"]))
    fbs_names = {t["school"] for t in r["fbs"]}
    ranked = sorted(((t, v) for t, v in elo.items() if t in fbs_names), key=lambda kv: -kv[1])
    save_json("upcoming.json", {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "season": SEASON,
        "elo": {t: round(v, 1) for t, v in ranked},
        "teams": {t["school"]: [t.get("abbreviation") or t["school"][:4].upper(),
                                t.get("color") or "#555", t.get("conference") or ""]
                  for t in r["fbs"]},
        "games": upcoming,
    })
    save_json(f"ratings/{SEASON}.json", {t: round(v, 1) for t, v in
                                          sorted(elo.items(), key=lambda kv: -kv[1])})
    print(f"wrote {len(upcoming)} upcoming games; API calls this run: {cfbd.CALLS}")


if __name__ == "__main__":
    main()
