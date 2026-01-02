import sys
import pandas as pd
from pathlib import Path
from playwright.sync_api import sync_playwright

from IDs.helpers import build_wowy_url

def main():
    if len(sys.argv) < 2:
        print("Usage: python load_stocks.py TEAM [PLAYER_OFF_NAME ...]")
        sys.exit(1)

    team = sys.argv[1].upper()
    players_off = sys.argv[2:]  # can be empty

    # Stocks live under Totals -> Misc on pbpstats WOWY
    url = build_wowy_url(
        team=team,
        players_off=players_off,
        table="Misc",
        stat_type="Per100Possessions",
    )

    out_dir = Path.home() / "Desktop" / "Optimizer" / "data" / team
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "stocks.csv"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(url, wait_until="domcontentloaded", timeout=120000)
        page.wait_for_selector("table", timeout=120000)

        df = pd.read_html(page.content())[0]
        df.to_csv(out_path, index=False)

        browser.close()

    print(str(out_path))


if __name__ == "__main__":
    main()
