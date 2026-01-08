
# FanDuel NBA DFS Optimizer

An NBA daily fantasy sports (DFS) lineup optimization and simulation framework built for FanDuel classic contests.

This project estimates FanDuel fantasy points using a combination of WOWY on/off–court splits, Vegas market data, and NBA.com projections. It is designed to approximate the workflow of professional DFS simulation tools, prioritizing flexibility and experimentation over a single deterministic “optimal” lineup.

---

## Basic Workflow

Run the project in the following order.

### 1. Load player and on/off data
```
python load_all.py NYK "Jalen Brunson" "Josh Hart"
````

Scrapes WOWY on/off data. Optional inactive players can be passed as command-line arguments. This step is the most time-intensive.

### 2. Pull Vegas data

```
python get_vegas.py
```

Pulls Vegas totals and spreads via The Odds API. Requires an API key set as an environment variable. Writes `data/vegas.csv`.

### 3. Build the slate

```
python build.py 2025-01-07 2025-01-08
```

Aggregates player data, projections, and Vegas context into a slate-level dataset. Times are UTC; a single slate may span multiple dates.

### 4. Simulate and optimize

Run `sim.ipynb`.

The notebook calls `model.py`, `sim_prep.py`, and `cash_opt.py` in separate cells. Using a notebook allows manual overrides (minutes, exposure caps, player locks) without rerunning upstream steps.

---

## Notes

* API keys are handled via environment variables and are not committed
* The project is rough, experimental and intended for experimentation
* Users can inspect individual files for implementation details

---

## Disclaimer

Not affiliated with FanDuel. Independent, educational project intended to explore DFS simulation and lineup construction techniques. Advice/Questions encouraged.

```
```
