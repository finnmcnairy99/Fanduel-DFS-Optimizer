import re
import sys
import unicodedata
from pathlib import Path
from typing import List, Optional

import pandas as pd

DATA_DIR = Path("data")
DAILY_PATH = DATA_DIR / "DAILY.csv"
VEGAS_PATH = DATA_DIR / "vegas.csv"

STAT_FILES = ["points.csv", "assists.csv", "rebounds.csv", "turnovers.csv", "stocks.csv"]

OUT_CSV = Path("build.csv")
REPORT_TXT = Path("slate_merge_report.txt")

# default slate day(s) (YYYY-MM-DD) if you don't pass args
DEFAULT_TARGET_DATES = ["2025-12-21"]


# -------------------------
# Team mappings
# -------------------------
TEAM_NAME_TO_ABBR = {
    "atlanta hawks": "ATL",
    "boston celtics": "BOS",
    "brooklyn nets": "BKN",
    "charlotte hornets": "CHA",
    "chicago bulls": "CHI",
    "cleveland cavaliers": "CLE",
    "dallas mavericks": "DAL",
    "denver nuggets": "DEN",
    "detroit pistons": "DET",
    "golden state warriors": "GSW",
    "houston rockets": "HOU",
    "indiana pacers": "IND",
    "la clippers": "LAC",
    "los angeles clippers": "LAC",
    "la lakers": "LAL",
    "los angeles lakers": "LAL",
    "memphis grizzlies": "MEM",
    "miami heat": "MIA",
    "milwaukee bucks": "MIL",
    "minnesota timberwolves": "MIN",
    "new orleans pelicans": "NOP",
    "new york knicks": "NYK",
    "oklahoma city thunder": "OKC",
    "orlando magic": "ORL",
    "philadelphia 76ers": "PHI",
    "phoenix suns": "PHX",
    "portland trail blazers": "POR",
    "sacramento kings": "SAC",
    "san antonio spurs": "SAS",
    "toronto raptors": "TOR",
    "utah jazz": "UTA",
    "washington wizards": "WAS",
}

# alias handling for what various sources might use
ABBR_ALIASES = {
    "GS": "GSW",
    "SA": "SAS",
    "NO": "NOP",
    "UT": "UTA",
    "NY": "NYK",
    "WSH": "WAS",
    "PHO": "PHX",
    "BK": "BKN",
    "BRK": "BKN",
    "OKLA": "OKC",
    "NOR": "NOP",
}

# Some books/sources shorten these weirdly
TEAM_NAME_ALIASES = {
    "oklahoma city": "oklahoma city thunder",
    "la clippers": "los angeles clippers",
    "los angeles clippers": "los angeles clippers",
    "la lakers": "los angeles lakers",
    "los angeles lakers": "los angeles lakers",
}


# -------------------------
# CLI date parsing (multi-date)
# -------------------------
def parse_target_dates(argv: List[str]) -> List[str]:
    """
    Accepts:
      python build.py 2025-12-16
      python build.py 2025-12-25 2025-12-26
      python build.py --date 2025-12-16
      python build.py --date 2025-12-25 --date 2025-12-26
      python build.py --date 2025-12-25 2025-12-26
      python build.py --dates 2025-12-25,2025-12-26
      python build.py --dates 2025-12-25 2025-12-26

    Returns: list of YYYY-MM-DD strings (unique, sorted)
    """
    dates: List[str] = []

    def _is_date_token(tok: str) -> bool:
        return bool(re.match(r"^\d{4}-\d{2}-\d{2}$", tok.strip()))

    i = 1
    while i < len(argv):
        tok = argv[i].strip()

        if tok == "--date":
            if i + 1 >= len(argv):
                raise ValueError("--date requires YYYY-MM-DD")
            val = argv[i + 1].strip().strip(",")
            if not _is_date_token(val):
                raise ValueError(f"--date must be YYYY-MM-DD, got: {val}")
            dates.append(val)
            i += 2
            continue

        if tok == "--dates":
            if i + 1 >= len(argv):
                raise ValueError("--dates requires YYYY-MM-DD values")
            j = i + 1
            # consume one or more tokens until next flag
            while j < len(argv) and not argv[j].startswith("-"):
                chunk = argv[j].strip()
                if chunk:
                    for part in chunk.split(","):
                        part = part.strip()
                        if not part:
                            continue
                        if not _is_date_token(part):
                            raise ValueError(f"--dates must be YYYY-MM-DD, got: {part}")
                        dates.append(part)
                j += 1
            i = j
            continue

        # positional dates (allow multiple)
        if not tok.startswith("-") and _is_date_token(tok):
            dates.append(tok)
            i += 1
            continue

        # ignore other unknown flags/tokens to stay backward-compatible
        i += 1

    if not dates:
        dates = list(DEFAULT_TARGET_DATES)

    # unique + sorted
    dates = sorted(set(dates))

    # final validation
    for d in dates:
        if not _is_date_token(d):
            raise ValueError(f"TARGET_DATE must be YYYY-MM-DD, got: {d}")

    return dates


# -------------------------
# Normalizers
# -------------------------
def normalize_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)

    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    s = s.lower().strip()
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # merge spaced initials: "p j washington" -> "pj washington"
    s = re.sub(r"\b([a-z])\s+([a-z])\b", r"\1\2", s)

    # drop suffixes
    toks = s.split()
    suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}
    toks = [t for t in toks if t not in suffixes]
    return " ".join(toks)


def normalize_team_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip().lower()
    s = s.replace("&", "and")
    s = re.sub(r"\s+", " ", s)
    s = TEAM_NAME_ALIASES.get(s, s)
    return s


def normalize_game_abbr(s: str) -> str:
    if pd.isna(s):
        return ""
    x = str(s).upper().strip()

    # kill punctuation like "L.A."
    x = re.sub(r"[^A-Z0-9@\s]", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    # normalize separators to "@"
    x = x.replace("VS.", "@").replace("VS", "@")
    x = re.sub(r"\s*@\s*", " @ ", x)

    m = re.match(r"^([A-Z]{2,5})\s+@\s+([A-Z]{2,5})$", x)
    if not m:
        return ""

    away, home = m.group(1), m.group(2)

    if away == "LA":
        away = "LAL"
    if home == "LA":
        home = "LAL"

    away = ABBR_ALIASES.get(away, away)
    home = ABBR_ALIASES.get(home, home)

    if home == "UTAH":
        home = "UTA"
    if away == "UTAH":
        away = "UTA"

    return f"{away} @ {home}"


def parse_salary(x) -> float:
    if pd.isna(x):
        return float("nan")
    s = str(x).strip().replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return float("nan")


# -------------------------
# Stat table helpers
# -------------------------
def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, engine="python")


def clean_table_headers(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        c2 = re.sub(r"\s+Sort table by.*$", "", str(c)).strip()
        new_cols.append(c2)
    df.columns = new_cols
    df = df.rename(columns={df.columns[0]: "player"})
    return df


def coerce_numeric_columns(df: pd.DataFrame, exclude=("player", "join_key")) -> pd.DataFrame:
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype == object:
            s = df[c].astype(str).str.strip()
            s = s.str.replace(",", "", regex=False)
            s = s.str.replace("%", "", regex=False)
            s = s.replace({"nan": None, "None": None, "N/A": None, "NA": None})
            df[c] = pd.to_numeric(s, errors="coerce")
        else:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# -------------------------
# Main
# -------------------------
def main():
    TARGET_DATES = parse_target_dates(sys.argv)
    report_lines = [f"TARGET_DATES={','.join(TARGET_DATES)}"]

    # --- load vegas and restrict to TARGET_DATES by commence_time_utc ---
    vegas = pd.read_csv(VEGAS_PATH)

    required_vegas_cols = {"game_id", "home", "away", "commence_time_utc", "book", "total", "spread_home"}
    missing = required_vegas_cols - set(vegas.columns)
    if missing:
        raise ValueError(f"vegas.csv missing columns: {sorted(missing)}")

    vegas["commence_time_utc"] = pd.to_datetime(vegas["commence_time_utc"], utc=True, errors="coerce")
    vegas["game_date"] = vegas["commence_time_utc"].dt.date.astype(str)

    vegas_days = vegas[vegas["game_date"].isin(TARGET_DATES)].copy()
    if vegas_days.empty:
        raise ValueError(f"No vegas games found for TARGET_DATES={TARGET_DATES}")

    # Build an abbrev-based game key: "AAA @ BBB"
    vegas_days["home_abbr"] = vegas_days["home"].map(lambda x: TEAM_NAME_TO_ABBR.get(normalize_team_name(x)))
    vegas_days["away_abbr"] = vegas_days["away"].map(lambda x: TEAM_NAME_TO_ABBR.get(normalize_team_name(x)))
    bad_vegas = vegas_days["home_abbr"].isna() | vegas_days["away_abbr"].isna()
    if bad_vegas.any():
        sample = vegas_days.loc[bad_vegas, ["away", "home"]].head(40).to_string(index=False)
        raise ValueError(
            "Could not map some vegas team names to abbreviations.\n"
            "Fix TEAM_NAME_TO_ABBR / TEAM_NAME_ALIASES.\n"
            f"Sample:\n{sample}"
        )

    vegas_days["game_abbr"] = vegas_days["away_abbr"] + " @ " + vegas_days["home_abbr"]
    allowed_games_abbr = set(vegas_days["game_abbr"].astype(str))

    report_lines.append(f"vegas games (all target dates): {len(vegas_days)}")
    report_lines.append("vegas allowed game_abbr:")
    report_lines.append(", ".join(sorted(allowed_games_abbr)))

    vegas_join = vegas_days[["game_abbr", "total", "spread_home", "game_date"]].copy()

    # --- load DAILY ---
    daily_raw = pd.read_csv(DAILY_PATH)

    keep = ["player", "team", "gameInfo", "salary", "minutes"]
    missing_cols = [c for c in keep if c not in daily_raw.columns]
    if missing_cols:
        raise ValueError(f"DAILY.csv missing columns: {missing_cols}")

    daily = daily_raw[keep].copy()

    # normalize DAILY gameInfo to abbrev key
    daily["game_abbr"] = daily["gameInfo"].map(normalize_game_abbr)

    # report mapping issues
    unmapped = daily["game_abbr"].eq("") | daily["game_abbr"].isna()
    report_lines.append(f"\nDAILY rows: {len(daily)}")
    report_lines.append(f"DAILY unmapped gameInfo rows: {int(unmapped.sum())}")
    if unmapped.any():
        report_lines.append("Sample unmapped DAILY gameInfo:")
        report_lines.append(daily.loc[unmapped, ["gameInfo"]].head(30).to_string(index=False))

    # filter slate to games for TARGET_DATES by abbrev key
    before = len(daily)
    daily = daily[daily["game_abbr"].isin(allowed_games_abbr)].copy()
    after = len(daily)

    report_lines.append(f"\nDAILY rows after vegas game filter: {after} (dropped {before - after})")
    report_lines.append(f"matched games: {daily['game_abbr'].nunique()} / {len(allowed_games_abbr)}")
    report_lines.append("matched game_abbr:")
    report_lines.append(", ".join(sorted(set(daily["game_abbr"]))))

    if daily.empty:
        raise ValueError(
            f"After filtering DAILY.csv by vegas games for {TARGET_DATES}, no rows remained.\n"
            "Check DAILY gameInfo formatting and normalize_game_abbr()."
        )

    # merge vegas onto players by abbrev game key
    daily = daily.merge(vegas_join, on="game_abbr", how="left")

    # derive team_abbr from DAILY 'team' full name when possible; else fallback
    daily["team_abbr"] = daily["team"].map(lambda x: TEAM_NAME_TO_ABBR.get(normalize_team_name(x), None))
    failed_team = daily["team_abbr"].isna()
    if failed_team.any():
        daily.loc[failed_team, "team_abbr"] = daily.loc[failed_team, "team"].astype(str).str.upper().str.strip()

    # canonicalize money/minutes
    daily["salary"] = daily["salary"].map(parse_salary)
    daily["minutes"] = pd.to_numeric(daily["minutes"], errors="coerce")

    # Build join_key for stat merges: TEAM_ABBR|normalized_player_name
    daily["player_key"] = daily["player"].map(normalize_name)

    # manual fix example
    daily.loc[daily["player_key"].str.contains("hyland", na=False), ["player_key", "player"]] = [
        "bones hyland",
        "Bones Hyland",
    ]



    # PJ Washington: clean.csv has "PJ Washington" but players.csv has "P.J. Washington"
    mask = (
        daily["player_key"].str.contains(r"\bpj washington\b", case=False, na=False)
        | daily["player_key"].str.contains(r"\bp\.j\.?\s*w[ae]shington\b", case=False, na=False)
    )
    daily.loc[mask, ["player_key", "player"]] = ["pj washington", "P.J. Washington"]

    # CJ McCollum: clean.csv has "C.J. McCollum" but players.csv has "CJ McCollum"
    mask = (
        daily["player_key"].str.contains(r"\bcj mccollum\b", case=False, na=False)
        | daily["player_key"].str.contains(r"\bc\.j\.?\s*mccollum\b", case=False, na=False)
    )
    daily.loc[mask, ["player_key", "player"]] = ["cj mccollum", "CJ McCollum"]
    
    
    mask = (
        daily["player_key"].str.contains(r"\bnic claxton\b", case=False, na=False)
    |   daily["player_key"].str.contains(r"\bnicolas claxton\b", case=False, na=False)
)
    daily.loc[mask, ["player_key", "player"]] = ["nic claxton", "Nic Claxton"]

    
    
    # Alex Sarr: build.csv has "Alexandre Sarr", players.csv has "Alex Sarr"
    mask = (
        daily["player_key"].str.contains(r"\balexandre sarr\b", case=False, na=False)
        | daily["player_key"].str.contains(r"\balex sarr\b", case=False, na=False)
    )
    daily.loc[mask, ["player_key", "player"]] = ["alex sarr", "Alex Sarr"]

    
    
    mask = (
        daily["player_key"].str.contains(r"\bbub carrington\b", case=False, na=False)
    |   daily["player_key"].str.contains(r"\bcarlton carrington\b", case=False, na=False)
)
    daily.loc[mask, ["player_key", "player"]] = ["bub carrington", "Bub Carrington"]


    
    
    
    

    daily["team_key"] = daily["team_abbr"].astype(str).str.upper().str.strip()
    daily["join_key"] = daily["team_key"] + "|" + daily["player_key"]

    slate_teams = sorted(set(daily["team_key"]))

    master = daily[
        [
            "player",
            "team",
            "team_key",
            "gameInfo",
            "game_abbr",
            "salary",
            "minutes",
            "total",
            "spread_home",
            "game_date",
            "join_key",
        ]
    ].copy()

    # --- merge each stat table ---
    for stat_file in STAT_FILES:
        per_team_tables = []

        for team in slate_teams:
            team_path = DATA_DIR / team / stat_file
            df = safe_read_csv(team_path)
            if df is None:
                continue

            df = clean_table_headers(df)

            df["team_key"] = team
            df["player_key"] = df["player"].map(normalize_name)
            df["join_key"] = df["team_key"] + "|" + df["player_key"]

            df = coerce_numeric_columns(df, exclude=("player", "player_key", "team_key", "join_key"))

            feature_cols = [c for c in df.columns if c not in {"player", "team_key", "player_key", "join_key"}]
            df = df[["join_key"] + feature_cols].copy()

            prefix = stat_file.replace(".csv", "")
            df = df.rename(columns={c: f"{prefix}_{c}" for c in feature_cols})

            per_team_tables.append(df)

        report_lines.append(f"\n=== {stat_file} ===")
        report_lines.append(f"Loaded team tables: {len(per_team_tables)} / {len(slate_teams)}")

        if not per_team_tables:
            report_lines.append(f"[WARN] No data loaded for {stat_file}")
            continue

        big_stat = pd.concat(per_team_tables, ignore_index=True).drop_duplicates(subset=["join_key"])

        before_cols = master.shape[1]
        master = master.merge(big_stat, on="join_key", how="left")
        added_cols = master.shape[1] - before_cols

        prefix = stat_file.replace(".csv", "")
        stat_cols = [c for c in master.columns if c.startswith(prefix + "_")]

        missing_mask = master[stat_cols].isna().all(axis=1) if stat_cols else pd.Series([True] * len(master))

        report_lines.append(f"Added columns: {added_cols}")
        report_lines.append(f"Missing matches (rows with ALL {prefix}_* NaN): {int(missing_mask.sum())} / {len(master)}")
        if missing_mask.any():
            sample = master.loc[missing_mask, ["player", "team_key"]].head(40).to_string(index=False)
            report_lines.append("Sample missing (first 40):")
            report_lines.append(sample)

    # final cleanup
    master = master.drop(columns=["team_key", "join_key"])

    master.to_csv(OUT_CSV, index=False)
    REPORT_TXT.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"✅ Wrote {OUT_CSV}")
    print(f"🧾 Wrote {REPORT_TXT}")
    print(f"📅 Included vegas dates: {', '.join(TARGET_DATES)}")


if __name__ == "__main__":
    main()
