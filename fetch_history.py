"""Download past seasons for building and backtesting the model.

Saves raw API responses to data/raw/ (uploaded as a workflow artifact, not
committed). About 8 calls per season.
"""
import json
import os

import cfbd

SEASONS = [int(s) for s in os.environ.get("SEASONS", "").split(",") if s] or \
          list(range(2014, 2027))

ENDPOINTS = {
    "games":      ("/games",               {"seasonType": "both", "classification": "fbs"}),
    "lines":      ("/lines",               {"seasonType": "both"}),
    "talent":     ("/talent",              {}),
    "returning":  ("/player/returning",    {}),
    "recruiting": ("/recruiting/teams",    {}),
    "fbs":        ("/teams/fbs",           {}),
    # Points added per play, offense and defense, for each team in each game.
    "ppa":        ("/ppa/games",           {"seasonType": "regular", "excludeGarbageTime": "true"}),
    "ppa_post":   ("/ppa/games",           {"seasonType": "postseason", "excludeGarbageTime": "true"}),
}
# ONLY=ppa,ppa_post fetches just those, to save calls.
ONLY = [s for s in os.environ.get("ONLY", "").split(",") if s]
if ONLY:
    ENDPOINTS = {k: v for k, v in ENDPOINTS.items() if k in ONLY}

os.makedirs("data/raw", exist_ok=True)
for season in SEASONS:
    for name, (path, params) in ENDPOINTS.items():
        try:
            data = cfbd.get(path, year=season, **params)
        except Exception as e:
            print(f"{season} {name}: FAILED ({e})")
            continue
        with open(f"data/raw/{name}_{season}.json", "w") as f:
            json.dump(data, f)
        print(f"{season} {name}: {len(data)} records")

# One sample record of each, so field names show up in the log.
for name in ENDPOINTS:
    path = f"data/raw/{name}_{SEASONS[-1]}.json"
    if os.path.exists(path):
        sample = json.load(open(path))[:1]
        print(f"--- sample {name}: {json.dumps(sample)[:1500]}")

print(f"API calls used: {cfbd.CALLS}")
