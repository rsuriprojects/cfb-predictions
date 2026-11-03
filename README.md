# cfb-predictions

College football game predictions: Elo ratings seeded each season from roster strength (247 team talent, returning production, recruiting), plus opponent-adjusted offense and defense efficiency (points added per play, garbage time removed), graded after every game. Live at https://rsuriprojects.github.io/cfb-predictions/.

**Backtest** (held-out 2023–2025 seasons, FBS vs FBS, 2,398 games): **71.4%** correct (70.4% without the efficiency ratings). The betting favorite got 72.6% on the same games, and the site tracks the model against it.

- `daily.py` runs every morning in GitHub Actions: logs picks up to a week before kickoff (never changed after), adds the betting favorite, grades finals into `results/<season>.csv` (one file per season, so each season starts at 0–0).
- `fit_model.py` tunes and backtests the model on past seasons and writes `model.json`. Data for it comes from the **fetch history** workflow.
- Data: [CollegeFootballData.com](https://collegefootballdata.com). The API key lives only in the `CFBD_API_KEY` Actions secret.
