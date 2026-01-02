# model.py
# Mean + variance projection builder for FanDuel NBA (classic)
#
# Fixes included (all "makes-sense" fixes, not blunt tuning):
# 1) USAGE USED SIGNIFICANTLY:
#    - usage-dependent shrink strength (low usage => heavier shrink; high usage => lighter shrink)
#    - usage-dependent poss/min shrink strength
# 2) ROLE-PLAYER OVERPROJECTION FIX (not global nerf):
#    - role-dependent diminishing returns at high minutes for low-usage players (non-linear scaling)
# 3) TEAM REAL-POINTS CONSERVATION (Vegas reconciliation):
#    - after defense adj, reconcile team real points to vegas implied points (soft blend)
#    - prevents slate-wide inflation from independent player projections
#
# Inputs:
# - build.csv (from build.py)
# - data/baseline_rates.csv (from baseline_nba.py)
# - ratings.csv (Team, DefRtg)
# - data/players.csv (FanDuel export with IDs/positions)
# - data/overrides.json (optional minutes/own overrides)
#
# Outputs:
# - model.csv (debuggable)
# - clean.csv (sim-facing)

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

# -------------------------
# Paths
# -------------------------
BUILD_PATH = Path("build.csv")
PLAYERS_PATH = Path("data") / "players.csv"
OVERRIDES_PATH = Path("data") / "overrides.json"

RATINGS_PATH = Path("ratings.csv")
BASELINE_PATH = Path("data") / "baseline_rates.csv"

MODEL_OUT = Path("model.csv")
CLEAN_OUT = Path("clean.csv")

# -------------------------
# FanDuel scoring
# -------------------------
FD_PTS = 1.0
FD_REB = 1.2
FD_AST = 1.5
FD_STL = 3.0
FD_BLK = 3.0
FD_TOV = -1.0

# -------------------------
# Pace scaling (mean)
# -------------------------
USE_VEGAS_PACE = True
LEAGUE_TOTAL_BASELINE = 228.0
PACE_SCALE_CAP_MEAN = (0.90, 1.10)

# -------------------------
# Variance knobs (proxy)
# -------------------------
MINUTES_SD_FRACTION = 0.10
PACE_VAR_FRACTION = 0.06

EVENT_VAR_MULT_MAIN = 1.10
EVENT_VAR_MULT_STOCKS = 1.65

FD_SD_FLOOR_FRAC = 0.12

# -------------------------
# Defense adjustment (mean)
# -------------------------
USE_OPP_DEF_ADJ = True
DEF_WEIGHT_MEAN = 0.70
DEF_MULT_CAP_MEAN = (0.94, 1.06)

DEF_WEIGHT_RAW = 1.35

# -------------------------
# WOWY -> baseline shrink (BASE values; actual becomes usage-dependent)
# -------------------------
K_PTS = 200
K_REB = 600
K_AST = 600
K_TOV = 800
K_STL = 2500
K_BLK = 2500
K_PPM = 1200

# -------------------------
# Usage config (significant)
# -------------------------
USG_BASELINE = 0.22
K_USG = 1200
USG_K_POWER = 1.35
USG_K_CAP = (0.55, 2.20)

# Usage-driven variance (optional but useful)
SD_USG_BETA = 0.22

# -------------------------
# Role-player diminishing returns (targets "6th-man-ish+" inflation)
# -------------------------
# Only kicks in when usage is below baseline AND minutes are high.
ROLE_DECAY_MAX = 0.10   # max 10% decay (nonlinear; only for low-usg high-min)
ROLE_USG_WIDTH = 0.12   # how quickly role_factor rises as usg drops below baseline
ROLE_MIN_START = 22.0   # minutes where decay begins
ROLE_MIN_WIDTH = 16.0   # minutes range to full effect

# Apply decay mostly to scoring and assists (reb mostly stable)
ROLE_DECAY_PTS_W = 1.00
ROLE_DECAY_AST_W = 0.55
ROLE_DECAY_TOV_W = 0.25  # light (lower usage -> fewer TOs, but we already model TO; keep small)

# -------------------------
# Team points reconciliation (Vegas conservation)
# -------------------------
TEAM_PTS_BLEND = 0.78     # 0 = none, 1 = full force
TEAM_PTS_CAP = (0.88, 1.12)  # don't over-correct too hard per team

# -------------------------
# Team mapping
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

# -------------------------
# Utilities
# -------------------------
def normalize_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = s.replace("’", "'").replace("–", "-").replace("—", "-")
    s = re.sub(r"[^a-z0-9\s\-']", " ", s)
    s = s.replace("-", " ")
    s = s.replace("'", " ")
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()
    suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}
    tokens = [t for t in tokens if t not in suffixes]
    return " ".join(tokens)


def to_num(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    s = str(x).strip().replace(",", "").replace("$", "")
    try:
        return float(s)
    except Exception:
        return np.nan


def pct_to_frac(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        v = float(x)
        return v / 100.0 if v > 1.0 else v
    s = str(x).strip()
    if s.endswith("%"):
        s = s[:-1].strip()
        try:
            return float(s) / 100.0
        except Exception:
            return np.nan
    try:
        v = float(s)
        return v / 100.0 if v > 1.0 else v
    except Exception:
        return np.nan


def safe_div(a, b, default=np.nan):
    try:
        if b == 0 or pd.isna(b) or pd.isna(a):
            return default
        return a / b
    except Exception:
        return default


def clamp(x, lo, hi):
    if pd.isna(x):
        return np.nan
    return max(lo, min(hi, x))


def team_to_abbr(team_full: str) -> str:
    if pd.isna(team_full):
        return ""
    key = str(team_full).strip().lower()
    key = re.sub(r"\s+", " ", key)
    return TEAM_NAME_TO_ABBR.get(key, str(team_full).strip().upper())


def parse_gameinfo(gameinfo: str) -> tuple[str, str]:
    if pd.isna(gameinfo):
        return ("", "")
    s = str(gameinfo).strip()
    m = re.match(r"^([A-Z]{2,3})\s*@\s*([A-Z]{2,3})$", s)
    if not m:
        return ("", "")
    return (m.group(1), m.group(2))


def opponent_from_gameinfo(team_abbr: str, gameinfo: str) -> str:
    away, home = parse_gameinfo(gameinfo)
    t = str(team_abbr).upper().strip()
    if t == away:
        return home
    if t == home:
        return away
    return home or away or ""


def abbrev_player(name: str) -> str:
    if pd.isna(name):
        return ""
    s = str(name).strip()
    parts = s.split()
    if len(parts) == 1:
        return s
    return f"{parts[0][0]}. {parts[-1]}"


def load_overrides(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def fmt_money(x) -> str:
    if pd.isna(x):
        return ""
    try:
        return f"${int(round(float(x)))}"
    except Exception:
        return ""


# -------------------------
# Baseline loader
# -------------------------
def load_baseline_rates(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    b = pd.read_csv(path)
    if "name_key" not in b.columns:
        b["name_key"] = b.get("player", "").map(normalize_name)

    for c in ["min_per100", "pts_per100", "reb_per100", "ast_per100", "stl_per100", "blk_per100", "tov_per100"]:
        if c in b.columns:
            b[c] = b[c].map(to_num)

    if "min_per100" in b.columns:
        b["base_poss_per_min"] = b["min_per100"].map(lambda m: safe_div(100.0, m, default=np.nan))
    else:
        b["base_poss_per_min"] = np.nan

    b = b.sort_values(["name_key"]).drop_duplicates(subset=["name_key"], keep="first").reset_index(drop=True)
    return b


# -------------------------
# Defense ratings loader
# -------------------------
def load_def_ratings(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["TEAM_ABBR", "DEF_RTG"])

    r = pd.read_csv(path)

    team_col = None
    def_col = None
    for c in r.columns:
        cl = str(c).strip().lower()
        if cl in {"team", "team_name", "teamraw"} and team_col is None:
            team_col = c
        if cl in {"defrtg", "def_rtg", "drtg", "def"} and def_col is None:
            def_col = c

    if team_col is None:
        for c in r.columns:
            if "team" in str(c).lower():
                team_col = c
                break
    if def_col is None:
        for c in r.columns:
            if "def" in str(c).lower() and "rtg" in str(c).lower():
                def_col = c
                break
        if def_col is None:
            for c in r.columns:
                if str(c).lower() in {"defrtg", "drtg"}:
                    def_col = c
                    break

    if team_col is None or def_col is None:
        return pd.DataFrame(columns=["TEAM_ABBR", "DEF_RTG"])

    out = r[[team_col, def_col]].copy()
    out.columns = ["TEAM_RAW", "DEF_RTG"]
    out["DEF_RTG"] = out["DEF_RTG"].map(to_num)
    out["TEAM_ABBR"] = out["TEAM_RAW"].map(team_to_abbr)
    out = out[["TEAM_ABBR", "DEF_RTG"]].dropna(subset=["TEAM_ABBR"]).drop_duplicates(subset=["TEAM_ABBR"])
    return out


# -------------------------
# Ownership prior
# -------------------------
def ownership_prior(df: pd.DataFrame) -> pd.Series:
    proj = df["fd_mean"].fillna(0.0).to_numpy()
    mins = df["minutes"].fillna(0.0).to_numpy()
    sal = df["salary"].fillna(0.0).to_numpy()
    sal_k = np.where(sal > 0, sal / 1000.0, np.nan)

    value = np.zeros_like(proj, dtype=float)
    ok = ~np.isnan(sal_k) & (sal_k > 0)
    value[ok] = proj[ok] / sal_k[ok]

    def z(x):
        x = np.asarray(x, dtype=float)
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        scale = max(1e-9, 1.4826 * mad)
        return (x - med) / scale

    z_val = z(value)
    z_proj = z(proj)

    expensive = (sal >= 9000).astype(float)
    score = 1.10 * z_val + 0.15 * expensive * z_proj + 0.10 * z(mins)

    p = 1.0 / (1.0 + np.exp(-score))
    own_pct = 1.0 + 32.0 * p
    own_pct += np.clip((6.0 - np.nan_to_num(sal_k, nan=0.0)), 0.0, 2.5) * 0.6
    own_pct = np.clip(own_pct, 0.2, 55.0)
    return pd.Series(own_pct, index=df.index)


# -------------------------
# FanDuel mapping loader
# -------------------------
def load_fd_mapping(players_path: Path) -> pd.DataFrame:
    if not players_path.exists():
        return pd.DataFrame(columns=["TEAM_ABBR", "NAME_KEY", "FD_ID", "FD_POS", "FD_GAME"])

    fd = pd.read_csv(players_path)

    if "Nickname" in fd.columns and fd["Nickname"].notna().any():
        fd["PLAYER_FULL"] = fd["Nickname"].astype(str)
    else:
        fd["PLAYER_FULL"] = (fd.get("First Name", "").astype(str) + " " + fd.get("Last Name", "").astype(str)).str.strip()

    fd["NAME_KEY"] = fd["PLAYER_FULL"].map(normalize_name)
    fd["TEAM_ABBR"] = fd.get("Team", "").astype(str).str.upper().str.strip()

    pos_col = "Roster Position" if "Roster Position" in fd.columns else ("Position" if "Position" in fd.columns else None)
    fd["FD_POS"] = fd[pos_col].astype(str).str.strip() if pos_col else ""

    fd["FD_ID"] = fd.get("Id", "").astype(str).str.strip()
    fd["FD_GAME"] = fd.get("Game", "").astype(str).str.strip()

    fd = fd[["TEAM_ABBR", "NAME_KEY", "FD_ID", "FD_POS", "FD_GAME"]].drop_duplicates(subset=["TEAM_ABBR", "NAME_KEY"])
    return fd


# -------------------------
# Shrink helpers
# -------------------------
def shrink_rate(base: float, wowy: float, minutes_sample: float, k: float) -> float:
    if pd.isna(base) and pd.isna(wowy):
        return np.nan
    if pd.isna(base):
        return wowy
    if pd.isna(wowy):
        return base
    mins = minutes_sample if (not pd.isna(minutes_sample) and minutes_sample > 0) else 0.0
    w = mins / (mins + k) if (mins + k) > 0 else 0.0
    return base + w * (wowy - base)


def k_with_usage(k_base: float, usg: float) -> float:
    if pd.isna(usg) or usg <= 0:
        return k_base
    factor = (USG_BASELINE / max(1e-6, usg)) ** USG_K_POWER
    factor = max(USG_K_CAP[0], min(USG_K_CAP[1], factor))
    return k_base * factor


def implied_team_points_from_vegas(gameinfo: str, total: float, spread_home: float) -> tuple[float, float]:
    """
    Returns (away_points, home_points). Uses the standard:
      home = total/2 - spread_home/2
      away = total/2 + spread_home/2
    where spread_home is from home team perspective (negative means home favored).
    """
    away, home = parse_gameinfo(gameinfo)
    if pd.isna(total) or total <= 0 or pd.isna(spread_home):
        return (np.nan, np.nan)
    home_pts = (total / 2.0) - (spread_home / 2.0)
    away_pts = (total / 2.0) + (spread_home / 2.0)
    return (away_pts, home_pts)


# -------------------------
# Main
# -------------------------
def main():
    if not BUILD_PATH.exists():
        raise FileNotFoundError(f"Missing {BUILD_PATH}. Run build.py first.")

    df = pd.read_csv(BUILD_PATH)

    # core numeric coercions
    for c in ["salary", "minutes", "total", "spread_home"]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].map(to_num)

    # WOWY columns used
    need_cols = [
        "points_Possessions",
        "points_Minutes",
        "points_Pts",
        "points_Usage",
        "assists_Assists",
        "turnovers_TOs",
        "stocks_Steals",
        "stocks_Blocks",
        "rebounds_DReb",
        "rebounds_OReb",
    ]
    for c in need_cols:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = df[c].map(to_num)

    df["TEAM_ABBR"] = df["team"].map(team_to_abbr)
    df["OPPONENT"] = df.apply(lambda r: opponent_from_gameinfo(r["TEAM_ABBR"], r.get("gameInfo", "")), axis=1)

    df["PLAYER_FULL"] = df["player"].astype(str)
    df["NAME_KEY"] = df["PLAYER_FULL"].map(normalize_name)

    # Apply overrides
    overrides = load_overrides(OVERRIDES_PATH)
    minutes_over = overrides.get("minutes", {}) if isinstance(overrides, dict) else {}
    own_over = overrides.get("own", {}) if isinstance(overrides, dict) else {}

    minutes_over_norm = {normalize_name(k): float(v) for k, v in minutes_over.items()}
    own_over_norm = {normalize_name(k): float(v) for k, v in own_over.items()}

    if minutes_over_norm:
        df["minutes"] = df.apply(lambda r: minutes_over_norm.get(r["NAME_KEY"], r["minutes"]), axis=1)

    # -------------------------
    # Baseline merge
    # -------------------------
    baseline = load_baseline_rates(BASELINE_PATH)
    if not baseline.empty:
        bcols = [
            "name_key",
            "pts_per100", "reb_per100", "ast_per100", "stl_per100", "blk_per100", "tov_per100",
            "base_poss_per_min",
        ]
        for c in bcols:
            if c not in baseline.columns:
                baseline[c] = np.nan
        baseline_small = baseline[bcols].copy().rename(columns={
            "name_key": "NAME_KEY",
            "pts_per100": "BASE_PTS100",
            "reb_per100": "BASE_REB100",
            "ast_per100": "BASE_AST100",
            "stl_per100": "BASE_STL100",
            "blk_per100": "BASE_BLK100",
            "tov_per100": "BASE_TOV100",
            "base_poss_per_min": "BASE_PPM",
        })
        df = df.merge(baseline_small, on="NAME_KEY", how="left")
    else:
        df["BASE_PTS100"] = np.nan
        df["BASE_REB100"] = np.nan
        df["BASE_AST100"] = np.nan
        df["BASE_STL100"] = np.nan
        df["BASE_BLK100"] = np.nan
        df["BASE_TOV100"] = np.nan
        df["BASE_PPM"] = np.nan

    # -------------------------
    # Step 0: usage (shrink usage toward baseline)
    # -------------------------
    df["WOWY_MINS_SAMPLE"] = df["points_Minutes"]
    df["WOWY_USG"] = df["points_Usage"].map(pct_to_frac)

    df["USG"] = df.apply(
        lambda r: shrink_rate(USG_BASELINE, r["WOWY_USG"], r["WOWY_MINS_SAMPLE"], K_USG),
        axis=1,
    )
    df["USG"] = df["USG"].clip(lower=0.08, upper=0.40)

    # usage-dependent effective Ks
    df["K_PPM_EFF"] = df["USG"].map(lambda u: k_with_usage(K_PPM, u))
    df["K_PTS_EFF"] = df["USG"].map(lambda u: k_with_usage(K_PTS, u))
    df["K_REB_EFF"] = df["USG"].map(lambda u: k_with_usage(K_REB, u))
    df["K_AST_EFF"] = df["USG"].map(lambda u: k_with_usage(K_AST, u))
    df["K_TOV_EFF"] = df["USG"].map(lambda u: k_with_usage(K_TOV, u))
    df["K_STL_EFF"] = df["USG"].map(lambda u: k_with_usage(K_STL, u))
    df["K_BLK_EFF"] = df["USG"].map(lambda u: k_with_usage(K_BLK, u))

    # -------------------------
    # Step 1: poss/min (usage-dependent shrink)
    # -------------------------
    df["WOWY_PPM"] = df.apply(lambda r: safe_div(r["points_Possessions"], r["points_Minutes"], default=np.nan), axis=1)
    df["poss_per_min"] = df.apply(
        lambda r: shrink_rate(r["BASE_PPM"], r["WOWY_PPM"], r["WOWY_MINS_SAMPLE"], r["K_PPM_EFF"]),
        axis=1,
    )
    df["poss_per_min"] = df["poss_per_min"].clip(lower=1.2, upper=2.4)

    # -------------------------
    # Step 2: pace scaling (mean)
    # -------------------------
    if USE_VEGAS_PACE:
        df["pace_scale_raw"] = df["total"].map(lambda t: safe_div(t, LEAGUE_TOTAL_BASELINE, default=1.0))
        df["pace_scale"] = df["pace_scale_raw"].map(lambda x: clamp(x, *PACE_SCALE_CAP_MEAN))
    else:
        df["pace_scale_raw"] = 1.0
        df["pace_scale"] = 1.0

    # -------------------------
    # Step 3: expected possessions
    # -------------------------
    df["poss_exp_raw"] = df["minutes"] * df["poss_per_min"]
    df["poss_exp"] = df["poss_exp_raw"] * df["pace_scale"]

    # -------------------------
    # Step 4: WOWY per-100 rates, then usage-dependent shrink
    # -------------------------
    df["WOWY_PTS100"] = df["points_Pts"]
    df["WOWY_AST100"] = df["assists_Assists"]
    df["WOWY_TOV100"] = df["turnovers_TOs"]
    df["WOWY_STL100"] = df["stocks_Steals"]
    df["WOWY_BLK100"] = df["stocks_Blocks"]
    df["WOWY_REB100"] = df["rebounds_DReb"] + df["rebounds_OReb"]

    df["PTS100"] = df.apply(lambda r: shrink_rate(r["BASE_PTS100"], r["WOWY_PTS100"], r["WOWY_MINS_SAMPLE"], r["K_PTS_EFF"]), axis=1)
    df["REB100"] = df.apply(lambda r: shrink_rate(r["BASE_REB100"], r["WOWY_REB100"], r["WOWY_MINS_SAMPLE"], r["K_REB_EFF"]), axis=1)
    df["AST100"] = df.apply(lambda r: shrink_rate(r["BASE_AST100"], r["WOWY_AST100"], r["WOWY_MINS_SAMPLE"], r["K_AST_EFF"]), axis=1)
    df["TOV100"] = df.apply(lambda r: shrink_rate(r["BASE_TOV100"], r["WOWY_TOV100"], r["WOWY_MINS_SAMPLE"], r["K_TOV_EFF"]), axis=1)
    df["STL100"] = df.apply(lambda r: shrink_rate(r["BASE_STL100"], r["WOWY_STL100"], r["WOWY_MINS_SAMPLE"], r["K_STL_EFF"]), axis=1)
    df["BLK100"] = df.apply(lambda r: shrink_rate(r["BASE_BLK100"], r["WOWY_BLK100"], r["WOWY_MINS_SAMPLE"], r["K_BLK_EFF"]), axis=1)

    # -------------------------
    # Step 5: counting stats from per-100 + possessions
    # -------------------------
    df["PTS"] = (df["PTS100"] / 100.0) * df["poss_exp"]
    df["REB"] = (df["REB100"] / 100.0) * df["poss_exp"]
    df["AST"] = (df["AST100"] / 100.0) * df["poss_exp"]
    df["TOV"] = (df["TOV100"] / 100.0) * df["poss_exp"]
    df["STL"] = (df["STL100"] / 100.0) * df["poss_exp"]
    df["BLK"] = (df["BLK100"] / 100.0) * df["poss_exp"]

    for c in ["PTS", "REB", "AST", "STL", "BLK", "TOV"]:
        df[c] = df[c].clip(lower=0)

    # -------------------------
    # Step 5b: role-player diminishing returns (low usage × high minutes)
    # -------------------------
    role_factor = ((USG_BASELINE - df["USG"]) / ROLE_USG_WIDTH).clip(lower=0.0, upper=1.0).fillna(0.0)
    min_factor = ((df["minutes"] - ROLE_MIN_START) / ROLE_MIN_WIDTH).clip(lower=0.0, upper=1.0).fillna(0.0)

    decay = 1.0 - (ROLE_DECAY_MAX * role_factor * min_factor)
    decay = decay.clip(lower=1.0 - ROLE_DECAY_MAX, upper=1.0)

    # apply to key stats (not rebounds)
    df["PTS"] *= (1.0 + (decay - 1.0) * ROLE_DECAY_PTS_W)
    df["AST"] *= (1.0 + (decay - 1.0) * ROLE_DECAY_AST_W)
    df["TOV"] *= (1.0 + (decay - 1.0) * ROLE_DECAY_TOV_W)

    # -------------------------
    # Step 6: defense adjustment (mean + raw debug)
    # -------------------------
    df["OPP_DEF_RTG"] = np.nan
    df["DEF_MULT_MEAN"] = 1.0
    df["DEF_MULT_RAW"] = 1.0

    if USE_OPP_DEF_ADJ:
        def_map = load_def_ratings(RATINGS_PATH)
        if not def_map.empty and def_map["DEF_RTG"].notna().any():
            def_by_team = dict(zip(def_map["TEAM_ABBR"], def_map["DEF_RTG"]))
            league_def_avg = float(np.nanmean(list(def_by_team.values()))) if len(def_by_team) else np.nan

            def lookup_opp_def(opp):
                o = str(opp).upper().strip()
                return def_by_team.get(o, np.nan)

            df["OPP_DEF_RTG"] = df["OPPONENT"].map(lookup_opp_def)

            if not pd.isna(league_def_avg):
                def_ratio = (df["OPP_DEF_RTG"] - league_def_avg) / league_def_avg

                df["DEF_MULT_RAW"] = 1.0 + DEF_WEIGHT_RAW * def_ratio
                df["DEF_MULT_RAW"] = df["DEF_MULT_RAW"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

                df["DEF_MULT_MEAN"] = 1.0 + DEF_WEIGHT_MEAN * def_ratio
                df["DEF_MULT_MEAN"] = df["DEF_MULT_MEAN"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
                df["DEF_MULT_MEAN"] = df["DEF_MULT_MEAN"].map(lambda x: clamp(x, *DEF_MULT_CAP_MEAN))

    # Apply mean defense multiplier
    df["PTS"] *= df["DEF_MULT_MEAN"]
    df["AST"] *= (1.0 + 0.35 * (df["DEF_MULT_MEAN"] - 1.0))
    df["TOV"] *= (1.0 + 0.40 * (df["DEF_MULT_MEAN"] - 1.0))

    # -------------------------
    # Step 6b: Vegas team points reconciliation (real points conservation)
    # -------------------------
    # Build implied points per team from each unique game_abbr row
    # (game_abbr is already in build.csv; gameInfo format: "AWY @ HOM")
    team_implied = {}

    # Use one row per gameInfo to compute implied (away, home)
    gcols = ["gameInfo", "total", "spread_home"]
    games = df[gcols].dropna(subset=["gameInfo"]).drop_duplicates(subset=["gameInfo"])
    for _, r in games.iterrows():
        gi = r["gameInfo"]
        total = r["total"]
        spread_home = r["spread_home"]
        away_pts, home_pts = implied_team_points_from_vegas(gi, total, spread_home)
        away, home = parse_gameinfo(gi)
        if away and not pd.isna(away_pts):
            team_implied[away] = float(away_pts)
        if home and not pd.isna(home_pts):
            team_implied[home] = float(home_pts)

    df["TEAM_IMPLIED_PTS"] = df["TEAM_ABBR"].map(team_implied)

    # Current projected team points (sum of player PTS)
    df["TEAM_MODEL_PTS_SUM"] = df.groupby("TEAM_ABBR")["PTS"].transform("sum")

    raw_scale = df["TEAM_IMPLIED_PTS"] / df["TEAM_MODEL_PTS_SUM"].replace(0, np.nan)
    raw_scale = raw_scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # Blend + cap
    scale = 1.0 + TEAM_PTS_BLEND * (raw_scale - 1.0)
    scale = scale.map(lambda x: clamp(x, *TEAM_PTS_CAP))

    # Apply to scoring-derived pieces
    df["TEAM_PTS_SCALE"] = scale
    df["PTS"] *= df["TEAM_PTS_SCALE"]
    # assists often scale with made shots / scoring environment; keep mild
    df["AST"] *= (1.0 + 0.50 * (df["TEAM_PTS_SCALE"] - 1.0))
    # turnovers scale weakly with game environment; keep very mild
    df["TOV"] *= (1.0 + 0.20 * (df["TEAM_PTS_SCALE"] - 1.0))
    # Rebounds/stocks left unchanged (not directly tied to team points)

    # -------------------------
    # Step 7: FanDuel mean
    # -------------------------
    df["fd_mean"] = (
        FD_PTS * df["PTS"]
        + FD_REB * df["REB"]
        + FD_AST * df["AST"]
        + FD_STL * df["STL"]
        + FD_BLK * df["BLK"]
        + FD_TOV * df["TOV"]
    )

    # -------------------------
    # Step 8: variance proxy -> fd_sd
    # -------------------------
    var_fd_events = (
        (FD_PTS ** 2) * (EVENT_VAR_MULT_MAIN * df["PTS"])
        + (FD_REB ** 2) * (EVENT_VAR_MULT_MAIN * df["REB"])
        + (FD_AST ** 2) * (EVENT_VAR_MULT_MAIN * df["AST"])
        + (FD_TOV ** 2) * (EVENT_VAR_MULT_MAIN * df["TOV"])
        + (FD_STL ** 2) * (EVENT_VAR_MULT_STOCKS * df["STL"])
        + (FD_BLK ** 2) * (EVENT_VAR_MULT_STOCKS * df["BLK"])
    )

    mins = df["minutes"].replace(0, np.nan)
    minutes_sd = MINUTES_SD_FRACTION * mins
    minutes_var_factor = (minutes_sd / mins) ** 2
    minutes_var_factor = minutes_var_factor.fillna(0.0)

    pace_sd = PACE_VAR_FRACTION * df["pace_scale"].fillna(1.0)
    pace_var_factor = (pace_sd / df["pace_scale"].replace(0, np.nan)) ** 2
    pace_var_factor = pace_var_factor.fillna(0.0)

    df["fd_var"] = var_fd_events * (1.0 + minutes_var_factor + pace_var_factor)
    df["fd_sd"] = np.sqrt(df["fd_var"].clip(lower=0.0))

    # Usage-driven volatility
    df["fd_sd"] *= (1.0 + SD_USG_BETA * (df["USG"] - USG_BASELINE).fillna(0.0))
    df["fd_sd"] = df["fd_sd"].clip(lower=0.0)

    # SD floor
    df["fd_sd"] = np.maximum(df["fd_sd"], FD_SD_FLOOR_FRAC * df["fd_mean"].clip(lower=0.0))

    # -------------------------
    # Step 9: value + ownership
    # -------------------------
    df["salary_k"] = df["salary"] / 1000.0
    df["value"] = df["fd_mean"] / df["salary_k"].replace(0, np.nan)
    df["value"] = df["value"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    df["own_pct"] = ownership_prior(df)
    if own_over_norm:
        df["own_pct"] = df.apply(lambda r: own_over_norm.get(r["NAME_KEY"], r["own_pct"]), axis=1)

    # -------------------------
    # Merge FD IDs + roster positions
    # -------------------------
    fd_map = load_fd_mapping(PLAYERS_PATH)
    out = df.merge(fd_map, on=["TEAM_ABBR", "NAME_KEY"], how="left")

    miss = out["FD_ID"].isna() | (out["FD_ID"].astype(str).str.strip() == "")
    if miss.any():
        fd_map2 = fd_map.drop_duplicates(subset=["NAME_KEY"]).copy()
        out2 = out.loc[miss].drop(columns=["FD_ID", "FD_POS", "FD_GAME"], errors="ignore").merge(
            fd_map2[["NAME_KEY", "FD_ID", "FD_POS", "FD_GAME"]],
            on="NAME_KEY",
            how="left",
        )
        for col in ["FD_ID", "FD_POS", "FD_GAME"]:
            out.loc[miss, col] = out2[col].values

    df = out

    # -------------------------
    # Outputs
    # -------------------------
    df["PLAYER"] = df["PLAYER_FULL"].map(abbrev_player)
    df["TEAM"] = df["TEAM_ABBR"].astype(str).str.upper()
    df["SAL"] = df["salary"].map(fmt_money)

    df = df.sort_values(["fd_mean", "salary"], ascending=[False, False]).reset_index(drop=True)

    # model.csv (debug)
    model_cols = [
        "FD_ID", "FD_POS", "FD_GAME",
        "PLAYER_FULL", "PLAYER", "NAME_KEY",
        "team", "TEAM", "OPPONENT",
        "gameInfo", "game_abbr",
        "salary", "SAL", "minutes",

        # usage
        "WOWY_USG", "USG",

        # team points reconciliation debug
        "TEAM_IMPLIED_PTS", "TEAM_MODEL_PTS_SUM", "TEAM_PTS_SCALE",

        # pace / possessions
        "WOWY_PPM", "BASE_PPM", "poss_per_min",
        "pace_scale_raw", "pace_scale",
        "poss_exp",

        # rates
        "WOWY_MINS_SAMPLE",
        "BASE_PTS100", "WOWY_PTS100", "PTS100",
        "BASE_REB100", "WOWY_REB100", "REB100",
        "BASE_AST100", "WOWY_AST100", "AST100",
        "BASE_STL100", "WOWY_STL100", "STL100",
        "BASE_BLK100", "WOWY_BLK100", "BLK100",
        "BASE_TOV100", "WOWY_TOV100", "TOV100",

        # counting stats
        "PTS", "REB", "AST", "STL", "BLK", "TOV",

        # defense
        "OPP_DEF_RTG", "DEF_MULT_MEAN", "DEF_MULT_RAW",

        # projection + var
        "fd_mean", "fd_sd", "fd_var",
        "value", "own_pct",
    ]
    for c in model_cols:
        if c not in df.columns:
            df[c] = np.nan
    df[model_cols].to_csv(MODEL_OUT, index=False)

    # clean.csv (sim-facing)
    clean = pd.DataFrame({
        "FD_ID": df["FD_ID"].fillna("").astype(str),
        "PLAYER_FULL": df["PLAYER_FULL"],
        "NAME_KEY": df["NAME_KEY"],
        "PLAYER": df["PLAYER"],
        "TEAM": df["TEAM"],
        "OPPONENT": df["OPPONENT"],
        "POS": df["FD_POS"].fillna("").astype(str).str.strip(),
        "FD_GAME": df["FD_GAME"].fillna("").astype(str).str.strip(),

        "SALARY": df["salary"].round(0).astype(int, errors="ignore"),
        "SAL": df["SAL"],

        "PROJECTION": df["fd_mean"],
        "FD_SD": df["fd_sd"],

        "VALUE": df["value"],
        "MINUTES": df["minutes"],
        "OWN_PCT": df["own_pct"],

        "Pace Scale": df["pace_scale"],
        "Pace Scale Raw": df["pace_scale_raw"],
        "Proj Poss": df["poss_exp"],

        "Proj PTS": df["PTS"],
        "Proj REB": df["REB"],
        "Proj AST": df["AST"],
        "Proj STL": df["STL"],
        "Proj BLK": df["BLK"],
        "Proj TOV": df["TOV"],

        "Opp Def": df["OPP_DEF_RTG"],
        "Def Mult Mean": df["DEF_MULT_MEAN"],
        "Def Mult Raw": df["DEF_MULT_RAW"],

        "WOWY Min Sample": df["WOWY_MINS_SAMPLE"],
        "USG": df["USG"],
    })

    num_cols = clean.select_dtypes(include=[np.number]).columns
    clean[num_cols] = clean[num_cols].round(2)
    clean.to_csv(CLEAN_OUT, index=False)


if __name__ == "__main__":
    main()
