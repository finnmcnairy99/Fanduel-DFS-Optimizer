import csv
import os
from pathlib import Path

import requests

# -------------------------
# Config
# -------------------------
DATA_DIR = Path("data")
OUT_PATH = DATA_DIR / "vegas.csv"

SPORT = "basketball_nba"
REGIONS = "us"
MARKETS = "spreads,totals"
ODDS_FORMAT = "american"

BASE_URL = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"

PREFERRED_BOOKS = [
    "DraftKings",
    "FanDuel",
    "Caesars",
    "BetMGM",
    "PointsBet",
]

# -------------------------
# API Key (env var)
# -------------------------
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
if not ODDS_API_KEY:
    raise RuntimeError(
        "ODDS_API_KEY not set. Add this to ~/.zshrc:\n"
        '  export ODDS_API_KEY="YOUR_KEY"\n'
        "Then run:\n"
        "  source ~/.zshrc"
    )


def pick_bookmaker(bookmakers):
    """
    Pick first available preferred bookmaker (by title), else fallback to first bookmaker.
    """
    if not bookmakers:
        return None

    for name in PREFERRED_BOOKS:
        for b in bookmakers:
            if b.get("title") == name:
                return b

    return bookmakers[0]


def extract_total_and_spread(event, book):
    """
    Return (total, spread_home) for the selected bookmaker.
    total: game total points line (float or None)
    spread_home: home team spread (float or None)
    """
    home = event.get("home_team")
    total = None
    spread_home = None

    for market in book.get("markets", []):
        key = market.get("key")

        if key == "totals":
            outcomes = market.get("outcomes", [])
            if outcomes:
                # totals market usually has Over/Under outcomes with same point
                total = outcomes[0].get("point")

        elif key == "spreads":
            for o in market.get("outcomes", []):
                if o.get("name") == home:
                    spread_home = o.get("point")

    try:
        total = float(total) if total is not None else None
    except Exception:
        total = None

    try:
        spread_home = float(spread_home) if spread_home is not None else None
    except Exception:
        spread_home = None

    return total, spread_home


def main():
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
    }

    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    events = r.json()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "game_id",
                "home",
                "away",
                "commence_time_utc",
                "book",
                "total",
                "spread_home",
            ]
        )

        for e in events:
            home = e.get("home_team")
            away = e.get("away_team")
            commence = e.get("commence_time")

            book = pick_bookmaker(e.get("bookmakers", []))
            if not book:
                continue

            total, spread_home = extract_total_and_spread(e, book)
            game_id = f"{away}@{home}"

            writer.writerow(
                [
                    game_id,
                    home,
                    away,
                    commence,
                    book.get("title"),
                    total,
                    spread_home,
                ]
            )

    print(f"✅ wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
