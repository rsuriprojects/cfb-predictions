"""Tune and backtest the model on past seasons, then save it to model.json.

Seasons are split so nothing is tested on data it learned from:
  2014            warm-up only (ratings need a season to settle)
  2015-2020       fit the regression weights
  2021-2022       pick the Elo and roster settings
  2023-2025       held-out test: the honest accuracy number
  2026 to date    also out of sample

Reads raw CFBD responses from DATA (default data/raw), as saved by
fetch_history.py.
"""
import itertools
import json
import os

import numpy as np

import model

DATA = os.environ.get("DATA", "data/raw")
SEASONS = list(range(2014, 2027))
TRAIN, VALID, TEST = range(2015, 2021), range(2021, 2023), range(2023, 2026)


def load(name, season):
    path = f"{DATA}/{name}_{season}.json"
    return json.load(open(path)) if os.path.exists(path) else []


RAW = {s: {n: load(n, s) for n in ("games", "lines", "talent", "returning", "recruiting", "fbs")}
       for s in SEASONS}
ROSTER = {s: model.roster(RAW[s]["talent"], RAW[s]["returning"], RAW[s]["recruiting"], RAW[s]["fbs"])
          for s in SEASONS}


def walk(p, weights=None):
    """Run every season; return per-game rows and each season's final ratings."""
    rows, last, ends = [], {}, {}
    for s in SEASONS:
        last = model.run_season(RAW[s]["games"], ROSTER[s], RAW[s]["fbs"], last, p,
                                lines=RAW[s]["lines"], weights=weights, collect=rows)
        ends[s] = dict(last)
    return rows, ends


def fit_logistic(X, y, l2=1.0, iters=50):
    """Plain Newton's-method logistic regression with a light ridge penalty."""
    X1 = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X1.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X1 @ w))
        g = X1.T @ (p - y) + l2 * np.r_[0, w[1:]]
        H = (X1.T * (p * (1 - p))) @ X1 + l2 * np.diag(np.r_[0, np.ones(len(w) - 1)])
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return {"bias": float(w[0]), "coef": [float(c) for c in w[1:]]}


def graded(rows, seasons):
    return [r for r in rows if r["season"] in seasons and "home_won" in r]


def xy(rows):
    return np.array([r["x"] for r in rows]), np.array([float(r["home_won"]) for r in rows])


def logloss(w, rows):
    X, y = xy(rows)
    p = np.clip([model.prob(w, x) for x in X], 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


# ---------------------------------------------------------------- tune
grid = {
    "K": [10, 15, 20],
    "CARRY": [0.15, 0.25, 0.4],
    "W_TALENT": [90, 130, 170],
    "W_RET": [15, 30, 45],
    "W_REC": [0, 20, 40],
}
best = None
for combo in itertools.product(*grid.values()):
    p = dict(zip(grid, combo))
    rows, _ = walk(p)
    w = fit_logistic(*xy(graded(rows, TRAIN)))
    ll = logloss(w, graded(rows, VALID))
    if best is None or ll < best[0]:
        best = (ll, p)
ll, params = best
print(f"best settings {params} (validation log loss {ll:.4f})")

# Refit the weights on train + validation with those settings.
rows, ends = walk(params)
weights = fit_logistic(*xy(graded(rows, list(TRAIN) + list(VALID))))
print("weights", dict(zip(["bias"] + model.FEATURES, [weights["bias"]] + weights["coef"])))
for r in rows:
    r["p_home"] = model.prob(weights, r["x"])


# ---------------------------------------------------------------- report
def report(label, rs):
    if not rs:
        return
    hit = [(r["p_home"] > 0.5) == r["home_won"] for r in rs]
    with_mkt = [r for r in rs if r.get("p_market") not in (None, 0.5)]
    m_hit = [(r["p_home"] > 0.5) == r["home_won"] for r in with_mkt]
    f_hit = [(r["p_market"] > 0.5) == r["home_won"] for r in with_mkt]
    split = [r for r in with_mkt if (r["p_home"] > 0.5) != (r["p_market"] > 0.5)]
    s_m = sum((r["p_home"] > 0.5) == r["home_won"] for r in split)
    print(f"{label:<28} model {np.mean(hit):.1%} of {len(rs):>5}"
          f" | with a line: model {np.mean(m_hit):.1%}, favorite {np.mean(f_hit):.1%}"
          f" ({len(with_mkt)}) | disagreed {len(split)}: model won {s_m}")


print()
for label, seasons in (("TEST 2023-2025", TEST), ("2026 to date", [2026])):
    rs = graded(rows, seasons)
    report(f"{label}, FBS vs FBS", [r for r in rs if r["fbs_both"]])
    report(f"{label}, all games", rs)
for s in TEST:
    report(f"  {s} FBS vs FBS", [r for r in graded(rows, [s]) if r["fbs_both"]])

test_fbs = [r for r in graded(rows, TEST) if r["fbs_both"]]
print("\ncalibration on test, FBS vs FBS (confidence: predicted -> actual)")
for lo, hi in ((.5, .6), (.6, .7), (.7, .8), (.8, .9), (.9, 1.01)):
    b = [r for r in test_fbs if lo <= max(r["p_home"], 1 - r["p_home"]) < hi]
    if b:
        pred = np.mean([max(r["p_home"], 1 - r["p_home"]) for r in b])
        act = np.mean([(r["p_home"] > 0.5) == r["home_won"] for r in b])
        print(f"  {lo:.0%}-{min(hi, 1):.0%}: predicted {pred:.1%}, actual {act:.1%}  ({len(b)} games)")

# Ablation: the same pipeline with rosters switched off, to show what they add.
plain = dict(params, W_TALENT=0, W_RET=0, W_REC=0)
prow, _ = walk(plain)
fit_rows = graded(prow, list(TRAIN) + list(VALID))
pw = fit_logistic(np.array([[r["x"][0]] for r in fit_rows]),
                  np.array([float(r["home_won"]) for r in fit_rows]))
pt = [r for r in graded(prow, TEST) if r["fbs_both"]]
acc = np.mean([(model.prob(pw, [r["x"][0]]) > 0.5) == r["home_won"] for r in pt])
print(f"\nwithout rosters (Elo + home only), test FBS vs FBS: {acc:.1%}")

with open("model.json", "w") as f:
    json.dump({"params": params, "weights": weights, "features": model.FEATURES,
               "trained_on": f"{TRAIN.start}-{VALID.stop - 1}"}, f, indent=2)
os.makedirs("ratings", exist_ok=True)
with open("ratings/2025.json", "w") as f:
    json.dump({t: round(v, 1) for t, v in sorted(ends[2025].items(), key=lambda kv: -kv[1])}, f, indent=1)
print("wrote model.json and ratings/2025.json")
