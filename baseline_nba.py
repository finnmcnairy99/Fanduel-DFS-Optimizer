# baseline_nba.py
# Fast daily NBA baseline per-100 possessions (no args, no Playwright).
#
# Run:
#   python baseline_nba.py
#
# Output:
#   data/baseline_rates.csv
#   data/baseline_raw.json

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import requests


OUT_CSV = Path("data") / "baseline_rates.csv"
OUT_RAW = Path("data") / "baseline_raw.json"

URL = (
    "https://stats.nba.com/stats/leaguedashplayerstats"
    "?College=&Conference=&Country=&DateFrom=&DateTo=&Division=&DraftPick=&DraftYear=&"
    "GameScope=&GameSegment=&Height=&ISTRound=&LastNGames=0&LeagueID=00&Location=&"
    "MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PORound=0&PaceAdjust=N&"
    "PerMode=Per100Possessions&Period=0&PlayerExperience=&PlayerPosition=&"
    "PlusMinus=N&Rank=N&Season=2025-26&SeasonSegment=&SeasonType=Regular%20Season&"
    "ShotClockRange=&StarterBench=&TeamID=0&VsConference=&VsDivision=&Weight="
)

HEADERS = {
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/143.0.0.0 Safari/537.36"
    ),
}


def normalize_name(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # One request, hard timeout
    r = requests.get(URL, headers=HEADERS, timeout=10)
    r.raise_for_status()

    payload = r.json()
    OUT_RAW.write_text(json.dumps(payload), encoding="utf-8")

    rs = payload["resultSets"][0]
    df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])

    # Map columns (Per100 already)
    out = pd.DataFrame({
        "player": df["PLAYER_NAME"],
        "team_abbr": df["TEAM_ABBREVIATION"],
        "min_per100": df["MIN"],
        "pts_per100": df["PTS"],
        "reb_per100": df["REB"],
        "ast_per100": df["AST"],
        "stl_per100": df["STL"],
        "blk_per100": df["BLK"],
        "tov_per100": df["TOV"],
    })

    out["team_abbr"] = out["team_abbr"].astype(str).str.upper().str.strip()
    out["name_key"] = out["player"].map(normalize_name)

    num_cols = [
        "min_per100",
        "pts_per100",
        "reb_per100",
        "ast_per100",
        "stl_per100",
        "blk_per100",
        "tov_per100",
    ]
    out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce")

    out = (
        out.dropna(subset=["name_key", "team_abbr"])
           .drop_duplicates(subset=["team_abbr", "name_key"])
           .sort_values("pts_per100", ascending=False)
           .reset_index(drop=True)
    )

    out.to_csv(OUT_CSV, index=False)

    print(f"✅ wrote {OUT_CSV.resolve()} ({len(out)} players)")
    print(f"🧾 raw saved → {OUT_RAW.resolve()}")


if __name__ == "__main__":
    main()
