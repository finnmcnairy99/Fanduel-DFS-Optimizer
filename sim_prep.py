# sim_prep.py
# Fast late-swap aware slate sim + lineup builder for FanDuel NBA (classic)
# - Runtime-focused: still one lineup per sim; no expensive re-optimizing loops
# - Adds ceiling bias + stocks (STL/BLK) boost (high variance)
# - Encourages 2-2 game correlation / bringbacks
# - Softly discourages 4-from-team (but still allows 3)
# - Selection uses BOTH hit_count and max_sim_pts (cheap ceiling proxy)

from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# FD NBA roster (classic)
# -------------------------
ROSTER_SLOTS = ["PG", "PG", "SG", "SG", "SF", "SF", "PF", "PF", "C"]
SLOT_COUNTS = {"PG": 2, "SG": 2, "SF": 2, "PF": 2, "C": 1}
SLOT_KEYS = ["PG", "PG2", "SG", "SG2", "SF", "SF2", "PF", "PF2", "C"]

FD_SALARY_CAP = 60000
FD_MIN_SALARY = 59000
FD_MAX_PER_TEAM = 4

ENFORCE_FD_TEAM_ALSO = True

FD_UPLOAD_HEADER = ["PG", "PG", "SG", "SG", "SF", "SF", "PF", "PF", "C"]
FD_EDIT_HEADER = ["entry_id", "contest_id", "contest_name", "entry_fee"] + FD_UPLOAD_HEADER

FD_TO_CANON = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "PHO": "PHX", "SA": "SAS"}
CANON_TO_FD = {v: k for k, v in FD_TO_CANON.items()}


# -------------------------
# Printing
# -------------------------
def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _print(msg: str):
    print(msg, flush=True)


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
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    tokens = s.split()
    suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}
    tokens = [t for t in tokens if t not in suffixes]
    return " ".join(tokens)


def as_float(x, default=np.nan) -> float:
    try:
        if pd.isna(x):
            return default
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip().replace(",", "").replace("$", "")
        if s == "":
            return default
        return float(s)
    except Exception:
        return default


def split_pos(pos_str: str) -> List[str]:
    if pos_str is None or (isinstance(pos_str, float) and np.isnan(pos_str)):
        return []
    s = str(pos_str).strip().upper()
    if not s:
        return []
    parts = [p.strip() for p in s.split("/") if p.strip()]

    out: List[str] = []
    for p in parts:
        if p == "G":
            out += ["PG", "SG"]
        elif p == "F":
            out += ["SF", "PF"]
        else:
            out.append(p)

    out = [p for p in out if p in {"PG", "SG", "SF", "PF", "C"}]
    return sorted(set(out))


def canon_team(team_abbr: str) -> str:
    t = str(team_abbr).upper().strip()
    return FD_TO_CANON.get(t, t)


def fd_team(team_abbr: str) -> str:
    t = str(team_abbr).upper().strip()
    return CANON_TO_FD.get(t, t)


def parse_fd_game(game: str) -> Tuple[str, str]:
    s = str(game).strip().upper()
    m = re.match(r"^([A-Z]{2,3})@([A-Z]{2,3})$", s)
    if not m:
        return ("", "")
    return (m.group(1), m.group(2))


def team_from_fd_game_for_player(row_team_canon: str, fd_game: str) -> str:
    a, h = parse_fd_game(fd_game)
    if not a or not h:
        return fd_team(row_team_canon)

    a_can = canon_team(a)
    h_can = canon_team(h)
    t_can = canon_team(row_team_canon)

    if t_can == a_can:
        return a
    if t_can == h_can:
        return h
    return fd_team(row_team_canon)


def _strip_quotes(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.strip()
    if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
        s = s[1:-1].strip()
    return s


def _clean_player_cell_to_id(cell: str) -> str:
    s = _strip_quotes(cell)
    if not s:
        return ""
    if ":" in s:
        s = s.split(":", 1)[0].strip()
    return s.strip().replace("\u200b", "")


def _is_cash_protected_contest(contest_name: str) -> bool:
    s = ("" if contest_name is None else str(contest_name)).lower()
    return ("double up" in s) or ("double-up" in s) or ("quintuple up" in s) or ("quintuple-up" in s)


# -------------------------
# Contest loading
# -------------------------
@dataclass
class Contest:
    name: str
    field_size: int
    entry_fee: float
    max_entries: int
    entries_you: int
    paid_entries: int
    payouts: List[Tuple[int, int, float]]  # (rank_min, rank_max, payout)


def pick_contest_json(contest_pick: str) -> Path:
    p = Path(contest_pick)
    if p.exists() and p.suffix.lower() == ".json":
        return p

    candidates: List[Path] = []
    contests_dir = Path("data") / "contests"
    if contests_dir.exists():
        candidates += sorted(contests_dir.glob("*.json"))

    candidates += [
        Path(f"data/contest_{contest_pick}.json"),
        Path(f"contest_{contest_pick}.json"),
        Path("contest.json"),
    ]
    candidates = [c for c in candidates if c.exists() and c.suffix.lower() == ".json"]

    if not candidates:
        raise FileNotFoundError(
            f"contest file not found for pick='{contest_pick}'. Put it in data/contests/*.json or pass a direct path."
        )

    key = str(contest_pick).lower().strip()

    def score(path: Path) -> Tuple[int, int]:
        s = path.name.lower()
        return (1 if key in s else 0, -len(s))

    return sorted(candidates, key=score, reverse=True)[0]


def load_contest(contest_pick: str) -> Contest:
    path = pick_contest_json(contest_pick)
    raw = json.loads(path.read_text(encoding="utf-8"))

    payouts = []
    for x in raw.get("payouts", []):
        payouts.append((int(x["rank_min"]), int(x["rank_max"]), float(x["payout"])))

    return Contest(
        name=str(raw.get("name", path.stem)),
        field_size=int(raw.get("field_size", 0)),
        entry_fee=float(raw.get("entry_fee", 0.0)),
        max_entries=int(raw.get("max_entries", 0)),
        entries_you=int(raw.get("entries_you", 0)),
        paid_entries=int(raw.get("paid_entries", 0)),
        payouts=payouts,
    )


def payout_for_rank(contest: Contest, rank: int) -> float:
    if rank <= 0:
        return 0.0
    for rmin, rmax, pay in contest.payouts:
        if rmin <= rank <= rmax:
            return float(pay)
    return 0.0


# -------------------------
# Load clean.csv
# -------------------------
def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def load_clean(clean_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(clean_csv)

    required = ["FD_ID", "PLAYER_FULL", "TEAM", "POS", "FD_GAME", "SALARY", "PROJECTION", "OWN_PCT"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"clean.csv missing columns: {missing}\nfound: {list(df.columns)}")

    df = df.copy()
    df["FD_ID"] = df["FD_ID"].astype(str).str.strip()
    df["PLAYER_FULL"] = df["PLAYER_FULL"].astype(str).str.strip()
    df["NAME_KEY"] = df.get("NAME_KEY", df["PLAYER_FULL"].map(normalize_name))
    df["TEAM"] = df["TEAM"].astype(str).str.upper().str.strip().map(canon_team)
    df["TEAM_CANON"] = df["TEAM"].astype(str).str.upper().str.strip().map(canon_team)
    df["FD_GAME"] = df["FD_GAME"].astype(str).str.upper().str.strip()

    df["SALARY"] = df["SALARY"].map(as_float).fillna(0.0).astype(int)
    df["PROJECTION"] = df["PROJECTION"].map(as_float).fillna(0.0).astype(float)
    df["OWN_PCT"] = df["OWN_PCT"].map(as_float).fillna(0.0).astype(float)

    # Optional: FD_SD
    if "FD_SD" in df.columns:
        df["FD_SD"] = df["FD_SD"].map(as_float).fillna(np.nan).astype(float)
    else:
        df["FD_SD"] = np.nan

    # Pace/Def multipliers
    pace_raw_col = _first_existing_col(df, ["Pace Scale Raw", "Pace Scale", "PACE_RAW"])
    def_raw_col = _first_existing_col(df, ["Def Mult Raw", "Def Mult", "DEF_RAW"])
    df["PACE_RAW"] = df[pace_raw_col].map(as_float) if pace_raw_col else 1.0
    df["DEF_RAW"] = df[def_raw_col].map(as_float) if def_raw_col else 1.0
    df["PACE_RAW"] = df["PACE_RAW"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    df["DEF_RAW"] = df["DEF_RAW"].replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # Stocks columns (whatever naming you use)
    stl_col = _first_existing_col(df, ["Proj STL", "PROJ_STL", "STL", "Steals", "STEALS"])
    blk_col = _first_existing_col(df, ["Proj BLK", "PROJ_BLK", "BLK", "Blocks", "BLOCKS"])
    if stl_col:
        df["PROJ_STL"] = df[stl_col].map(as_float).fillna(0.0).astype(float)
    else:
        df["PROJ_STL"] = 0.0
    if blk_col:
        df["PROJ_BLK"] = df[blk_col].map(as_float).fillna(0.0).astype(float)
    else:
        df["PROJ_BLK"] = 0.0
    df["PROJ_STOCKS"] = (df["PROJ_STL"] + df["PROJ_BLK"]).astype(float)

    df["ELIG"] = df["POS"].map(split_pos)
    df["TEAM_FD"] = df.apply(lambda r: team_from_fd_game_for_player(r["TEAM_CANON"], r["FD_GAME"]), axis=1)
    df["TEAM_FD"] = df["TEAM_FD"].astype(str).str.upper().str.strip()

    df = df[df["SALARY"] > 0].copy()
    df = df[df["PROJECTION"] > 0].copy()
    df = df[df["ELIG"].map(len) > 0].copy()
    df = df[df["TEAM_CANON"].astype(str).str.len() > 0].copy()

    return df.reset_index(drop=True)


# -------------------------
# Punt fade rule (keep it, but don’t murder GPP)
# -------------------------
def _apply_punt_fade_filter(
    clean_df: pd.DataFrame,
    exposure_min: Optional[Dict[str, float]],
    salary_cut: int = 4200,
    min_value_x: float = 3.6,
) -> pd.DataFrame:
    exposure_min = exposure_min or {}

    forced: set[str] = set()
    for k, v in exposure_min.items():
        try:
            if float(v) > 0:
                forced.add(normalize_name(k))
        except Exception:
            continue

    df = clean_df.copy()

    sal = df["SALARY"].astype(float)
    proj = df["PROJECTION"].astype(float)
    value_x = proj / np.clip(sal / 1000.0, 1e-9, None)

    low_sal_bad = (sal < float(salary_cut)) & (value_x < float(min_value_x))

    nk_norm = df["NAME_KEY"].astype(str).map(normalize_name)
    pf_norm = df["PLAYER_FULL"].astype(str).map(normalize_name)
    is_forced = nk_norm.isin(forced) | pf_norm.isin(forced)

    keep = (~low_sal_bad) | is_forced

    dropped = int((~keep).sum())
    if dropped > 0:
        _print(
            f"[sim_prep] punt_fade: dropped {dropped} (SAL<{salary_cut} & <{min_value_x:.1f}x), "
            f"kept forced={int(is_forced.sum())}"
        )

    return df.loc[keep].reset_index(drop=True)


# -------------------------
# Variance fallback
# -------------------------
def build_sd_fallback(proj: np.ndarray) -> np.ndarray:
    sd = 0.33 * proj + 4.5
    sd = np.clip(sd, 6.0, 28.0)
    return sd.astype(float)


# -------------------------
# Optimizer helpers
# -------------------------
def _topk_from_scores(idx: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    if idx.size <= k:
        return idx
    vals = scores[idx]
    sel = np.argpartition(-vals, k - 1)[:k]
    return idx[sel]


def build_slot_indices(elig_lists: List[List[str]]) -> Dict[str, np.ndarray]:
    slots = {s: [] for s in SLOT_COUNTS.keys()}
    for i, elig in enumerate(elig_lists):
        for s in elig:
            if s in slots:
                slots[s].append(i)
    return {s: np.array(v, dtype=int) for s, v in slots.items()}


def _team_counts_ok(team_arr: np.ndarray, picked: List[int], max_per_team: int) -> bool:
    counts: Dict[str, int] = {}
    for i in picked:
        t = str(team_arr[int(i)]).upper().strip()
        counts[t] = counts.get(t, 0) + 1
        if counts[t] > max_per_team:
            return False
    return True


def _team_counts(team_arr: np.ndarray, picked: List[int]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for i in picked:
        t = str(team_arr[int(i)]).upper().strip()
        c[t] = c.get(t, 0) + 1
    return c


def _discourage_4_stack(team_arr: np.ndarray, picked: List[int]) -> float:
    # 0 penalty <=3, mild penalty for 4 (we do NOT want 4-stack)
    counts = _team_counts(team_arr, picked)
    pen = 0.0
    for v in counts.values():
        if v >= 4:
            pen += 6.0
    return pen


def _game_stack_bonus(fd_game_arr: np.ndarray, picked: List[int]) -> float:
    # Encourage correlation without forcing 4-stacks:
    # - bonus for 2+ players from SAME game
    # - extra bonus if it’s 2-2 or 3-2 style (more balanced bringback)
    games: Dict[str, int] = {}
    for i in picked:
        g = str(fd_game_arr[int(i)]).upper().strip()
        if not g:
            continue
        games[g] = games.get(g, 0) + 1

    bonus = 0.0
    for c in games.values():
        if c == 2:
            bonus += 0.8
        elif c == 3:
            bonus += 1.6
        elif c >= 4:
            # still allow game stacks, but not too hard
            bonus += 1.9
    return bonus


def optimize_lineup_one_sim_with_locks(
    rng: np.random.Generator,
    scores: np.ndarray,
    salary: np.ndarray,
    own: np.ndarray,
    stocks: np.ndarray,
    sd_eff: np.ndarray,
    fd_game_arr: np.ndarray,
    elig_idx: Dict[str, np.ndarray],
    locked_slot_idx: Dict[int, int],
    team_rule: np.ndarray,
    team_fd: np.ndarray,
    cap: int = FD_SALARY_CAP,
    min_sal: int = FD_MIN_SALARY,
    topk_per_slot: int = 55,
    attempts: int = 14,
    salary_bonus: float = 0.00110,
    chalk_penalty: float = 0.0,
    # new knobs
    ceiling_w: float = 0.18,      # pulls in volatile ceiling guys (sd_eff)
    stocks_w: float = 0.35,       # boosts stocks (STL+BLK) players
    game_stack_w: float = 1.0,    # encourages 2-2 type correlation
    team4_pen_w: float = 1.0,     # discourages 4 from same team
) -> Optional[List[int]]:
    """
    Fast greedy optimizer w/ repair.
    Objective ~ sum(scores) + salary_bonus*salary + ceiling_w*sd_eff + stocks_w*stocks
              + game_stack_w*game_stack_bonus - team4_pen_w*discourage_4_stack
              - chalk_penalty*sum(log_own)
    """
    best_obj = -1e18
    best_slots: Optional[List[int]] = None

    # IMPORTANT: topK should consider ceiling too, otherwise spiky guys get cut.
    slot_cands: Dict[str, np.ndarray] = {}
    pre_score = scores + ceiling_w * sd_eff + stocks_w * stocks
    for s in SLOT_COUNTS:
        idx = elig_idx.get(s, np.array([], dtype=int))
        if idx.size == 0:
            return None
        slot_cands[s] = _topk_from_scores(idx, pre_score, topk_per_slot)

    log_own = np.log(np.clip(own, 0.25, 80.0))

    for _ in range(attempts):
        roster_pick = [-1] * 9
        used = set()

        for j, pi in locked_slot_idx.items():
            roster_pick[j] = int(pi)
            used.add(int(pi))

        locked_only = [x for x in roster_pick if x != -1]
        if locked_only and not _team_counts_ok(team_rule, locked_only, FD_MAX_PER_TEAM):
            continue
        if ENFORCE_FD_TEAM_ALSO and locked_only and not _team_counts_ok(team_fd, locked_only, FD_MAX_PER_TEAM):
            continue

        positions = [j for j in range(9) if roster_pick[j] == -1]
        rng.shuffle(positions)

        feasible = True
        for j in positions:
            slot = ROSTER_SLOTS[j]
            cands = slot_cands[slot]

            if used:
                mask = np.ones(len(cands), dtype=bool)
                for k, pi in enumerate(cands.tolist()):
                    if pi in used:
                        mask[k] = False
                cands2 = cands[mask]
            else:
                cands2 = cands

            if cands2.size == 0:
                feasible = False
                break

            # current team counts (fast dict on tiny roster)
            current = [x for x in roster_pick if x != -1]
            counts_rule: Dict[str, int] = {}
            for pi0 in current:
                t = str(team_rule[int(pi0)]).upper().strip()
                counts_rule[t] = counts_rule.get(t, 0) + 1

            counts_fd: Dict[str, int] = {}
            if ENFORCE_FD_TEAM_ALSO:
                for pi0 in current:
                    t = str(team_fd[int(pi0)]).upper().strip()
                    counts_fd[t] = counts_fd.get(t, 0) + 1

            ok_idx: List[int] = []
            for pi in cands2.tolist():
                t_rule = str(team_rule[int(pi)]).upper().strip()
                if counts_rule.get(t_rule, 0) >= FD_MAX_PER_TEAM:
                    continue
                if ENFORCE_FD_TEAM_ALSO:
                    t_fd = str(team_fd[int(pi)]).upper().strip()
                    if counts_fd.get(t_fd, 0) >= FD_MAX_PER_TEAM:
                        continue
                ok_idx.append(int(pi))

            if not ok_idx:
                feasible = False
                break

            c3 = np.array(ok_idx, dtype=int)

            # tiny jitter to avoid identical greedy paths (still cheap)
            jitter = rng.normal(0.0, 0.55, size=len(c3))
            cand_obj = (
                scores[c3]
                + salary_bonus * salary[c3]
                + ceiling_w * sd_eff[c3]
                + stocks_w * stocks[c3]
                + jitter
            )
            pick = int(c3[int(np.argmax(cand_obj))])

            roster_pick[j] = pick
            used.add(pick)

        if not feasible or any(x == -1 for x in roster_pick):
            continue
        if len(set(roster_pick)) != 9:
            continue
        if not _team_counts_ok(team_rule, roster_pick, FD_MAX_PER_TEAM):
            continue
        if ENFORCE_FD_TEAM_ALSO and not _team_counts_ok(team_fd, roster_pick, FD_MAX_PER_TEAM):
            continue

        ssum = int(np.sum(salary[roster_pick]))
        if ssum > cap or ssum < min_sal:
            locked_positions = set(locked_slot_idx.keys())
            roster_pick_arr = np.array(roster_pick, dtype=int)

            repaired = False
            for _rep in range(22):
                ssum = int(np.sum(salary[roster_pick_arr]))
                if min_sal <= ssum <= cap:
                    repaired = True
                    break

                need_more = (ssum < min_sal)
                swap_positions = [j for j in range(9) if j not in locked_positions]
                rng.shuffle(swap_positions)

                swapped = False
                for j in swap_positions:
                    slot = ROSTER_SLOTS[j]
                    cur = int(roster_pick_arr[j])

                    cands = slot_cands[slot]
                    roster_set = set(roster_pick_arr.tolist())
                    # small list, comprehension is fine
                    cands2 = cands[np.array([i not in roster_set for i in cands], dtype=bool)]
                    if cands2.size == 0:
                        continue

                    if need_more:
                        cands2 = cands2[salary[cands2] > salary[cur]]
                    else:
                        cands2 = cands2[salary[cands2] < salary[cur]]
                    if cands2.size == 0:
                        continue

                    current = roster_pick_arr.tolist()
                    current[j] = -1
                    current_real = [x for x in current if x != -1]

                    counts_rule: Dict[str, int] = {}
                    for pi0 in current_real:
                        t = str(team_rule[int(pi0)]).upper().strip()
                        counts_rule[t] = counts_rule.get(t, 0) + 1

                    counts_fd: Dict[str, int] = {}
                    if ENFORCE_FD_TEAM_ALSO:
                        for pi0 in current_real:
                            t = str(team_fd[int(pi0)]).upper().strip()
                            counts_fd[t] = counts_fd.get(t, 0) + 1

                    ok_idx = []
                    for pi in cands2.tolist():
                        t_rule = str(team_rule[int(pi)]).upper().strip()
                        if counts_rule.get(t_rule, 0) >= FD_MAX_PER_TEAM:
                            continue
                        if ENFORCE_FD_TEAM_ALSO:
                            t_fd = str(team_fd[int(pi)]).upper().strip()
                            if counts_fd.get(t_fd, 0) >= FD_MAX_PER_TEAM:
                                continue
                        ok_idx.append(int(pi))

                    if not ok_idx:
                        continue

                    c3 = np.array(ok_idx, dtype=int)
                    obj = (
                        scores[c3]
                        + salary_bonus * salary[c3]
                        + ceiling_w * sd_eff[c3]
                        + stocks_w * stocks[c3]
                    )
                    rep = int(c3[np.argmax(obj)])

                    roster_pick_arr[j] = rep

                    if len(set(roster_pick_arr.tolist())) != 9:
                        roster_pick_arr[j] = cur
                        continue
                    if not _team_counts_ok(team_rule, roster_pick_arr.tolist(), FD_MAX_PER_TEAM):
                        roster_pick_arr[j] = cur
                        continue
                    if ENFORCE_FD_TEAM_ALSO and not _team_counts_ok(team_fd, roster_pick_arr.tolist(), FD_MAX_PER_TEAM):
                        roster_pick_arr[j] = cur
                        continue

                    swapped = True
                    break

                if not swapped:
                    break

            if not repaired:
                continue

            roster_pick = roster_pick_arr.tolist()
            ssum = int(np.sum(salary[roster_pick]))

        pts = float(np.sum(scores[roster_pick]))
        obj = pts + salary_bonus * ssum + float(np.sum(ceiling_w * sd_eff[roster_pick])) + float(np.sum(stocks_w * stocks[roster_pick]))

        if chalk_penalty > 0.0:
            obj -= chalk_penalty * float(np.sum(log_own[roster_pick]))

        # correlation encouragement + 4-stack discouragement
        obj += game_stack_w * _game_stack_bonus(fd_game_arr, roster_pick)
        obj -= team4_pen_w * _discourage_4_stack(team_rule, roster_pick)

        if obj > best_obj:
            best_obj = obj
            best_slots = roster_pick

    return best_slots


# -------------------------
# Exposure report
# -------------------------
def exposure_df(lineups: List[Dict[str, Any]], id_to_player: Dict[str, str], entries_you: int) -> pd.DataFrame:
    counts: Dict[str, int] = {}
    for lu in lineups:
        for pid in lu["fd_ids"]:
            counts[pid] = counts.get(pid, 0) + 1

    rows = []
    for pid, c in counts.items():
        rows.append(
            {"FD_ID": pid, "PLAYER": id_to_player.get(pid, pid), "Count": c, "Exposure": c / entries_you if entries_you else 0.0}
        )
    return pd.DataFrame(rows).sort_values(["Exposure", "Count"], ascending=[False, False]).reset_index(drop=True)


# -------------------------
# Lock logic
# -------------------------
def compute_locked_teams(team_lock_minutes: Optional[Dict[str, int]], current_lock_minute: Optional[int]) -> List[str]:
    if not team_lock_minutes or current_lock_minute is None:
        return []
    locked = []
    for t, m in team_lock_minutes.items():
        try:
            if int(m) <= int(current_lock_minute):
                locked.append(canon_team(str(t).upper().strip()))
        except Exception:
            continue
    return sorted(set(locked))


def _is_locked_team_for_pid(pid: str, id_to_team_canon: Dict[str, str], locked_teams: set[str]) -> bool:
    t = id_to_team_canon.get(pid, "")
    if not t:
        return False
    return canon_team(t).upper().strip() in locked_teams


# -------------------------
# Portfolio helpers
# -------------------------
def _lineup_key(ids: List[str]) -> Tuple[str, ...]:
    return tuple(sorted(ids))


def _count_needed(entries: int, frac: float) -> int:
    return int(math.ceil(entries * float(frac)))


def _normalize_exp_dict(d: Optional[Dict[str, float]]) -> Dict[str, float]:
    d = d or {}
    out: Dict[str, float] = {}
    for k, v in d.items():
        try:
            out[normalize_name(k)] = float(v)
        except Exception:
            continue
    return out


def _build_name_key_by_id(fd_id: np.ndarray, name_key: np.ndarray) -> Dict[str, str]:
    return {str(pid): str(nk) for pid, nk in zip(fd_id, name_key)}


# -------------------------
# Uploaded FD entries parsing
# -------------------------
def _read_uploaded_entries(path: str | Path) -> Tuple[pd.DataFrame, List[str]]:
    path = str(path)
    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        for r in reader:
            rows.append(r)

    if not rows:
        raise ValueError("uploaded_fd_csv is empty.")

    header = [c.strip() for c in rows[0]]

    def find_col(name: str) -> int:
        for i, c in enumerate(header):
            if c.strip().strip('"').strip("'") == name:
                return i
        return -1

    i_entry = find_col("entry_id")
    i_contest = find_col("contest_id")
    i_name = find_col("contest_name")
    i_fee = find_col("entry_fee")

    if min(i_entry, i_contest, i_name, i_fee) < 0:
        raise ValueError(
            "uploaded_fd_csv missing one of required columns: entry_id, contest_id, contest_name, entry_fee.\n"
            f"Header was: {header}"
        )

    pos_start = -1
    for i in range(i_fee + 1, len(header)):
        if header[i].strip().strip('"').strip("'") == "PG":
            pos_start = i
            break
    if pos_start < 0:
        raise ValueError(f"Could not find position columns starting at PG. Header was: {header}")

    pos_idx = list(range(pos_start, pos_start + 9))
    if pos_idx[-1] >= len(header):
        raise ValueError(f"uploaded_fd_csv header too short for 9 position columns. Header was: {header}")

    clean_rows = []
    for raw in rows[1:]:
        if len(raw) < (pos_idx[-1] + 1):
            raw = raw + [""] * ((pos_idx[-1] + 1) - len(raw))

        entry_id = _strip_quotes(raw[i_entry])
        if entry_id == "":
            break

        contest_id = _strip_quotes(raw[i_contest])
        contest_name = _strip_quotes(raw[i_name])
        entry_fee = _strip_quotes(raw[i_fee])

        slots_raw = [raw[j] if j < len(raw) else "" for j in pos_idx]
        slots = [_clean_player_cell_to_id(x) for x in slots_raw]

        clean_rows.append([entry_id, contest_id, contest_name, entry_fee] + slots)

    cols = ["entry_id", "contest_id", "contest_name", "entry_fee"] + SLOT_KEYS
    up_df = pd.DataFrame(clean_rows, columns=cols)
    return up_df, SLOT_KEYS


# -------------------------
# Writers for FanDuel CSV formats
# -------------------------
def _write_fd_upload_csv(path: Path, slot_rows: List[List[str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(FD_UPLOAD_HEADER)
        for ids9 in slot_rows:
            w.writerow([str(x).strip() for x in ids9])


def _write_fd_edit_csv(path: Path, out_df: pd.DataFrame) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        w.writerow(FD_EDIT_HEADER)
        for _, row in out_df.iterrows():
            w.writerow(
                [
                    str(row["entry_id"]).strip(),
                    str(row["contest_id"]).strip(),
                    str(row["contest_name"]).strip(),
                    str(row["entry_fee"]).strip(),
                    str(row["PG"]).strip(),
                    str(row["PG2"]).strip(),
                    str(row["SG"]).strip(),
                    str(row["SG2"]).strip(),
                    str(row["SF"]).strip(),
                    str(row["SF2"]).strip(),
                    str(row["PF"]).strip(),
                    str(row["PF2"]).strip(),
                    str(row["C"]).strip(),
                ]
            )


# -------------------------
# ROI evaluation (unchanged; still fast scoring-only)
# -------------------------
def _roi_eval_fast(
    rng: np.random.Generator,
    contest: Contest,
    proj: np.ndarray,
    sd_eff: np.ndarray,
    base_mult: np.ndarray,
    game_idx: np.ndarray,
    team_idx: np.ndarray,
    n_games: int,
    n_teams: int,
    portfolio_idx9: np.ndarray,
    field_idx9: np.ndarray,
    roi_sims: int = 80,
    game_sigma: float = 0.075,
    team_sigma: float = 0.045,
) -> Dict[str, float]:
    if contest.field_size <= 0 or contest.entry_fee <= 0:
        return {"roi_mean": 0.0, "roi_p50": 0.0, "roi_p90": 0.0, "roi_sims": 0, "field_pool": int(field_idx9.shape[0])}

    P = int(portfolio_idx9.shape[0])
    F = int(field_idx9.shape[0])
    if P == 0 or F == 0 or roi_sims <= 0:
        return {"roi_mean": 0.0, "roi_p50": 0.0, "roi_p90": 0.0, "roi_sims": 0, "field_pool": F}

    total_sim = F + P
    roi_vals = np.zeros(roi_sims, dtype=float)

    t0 = datetime.now()
    for s in range(roi_sims):
        g_shock = rng.normal(0.0, game_sigma, size=n_games)
        t_shock = rng.normal(0.0, team_sigma, size=n_teams)

        mult = 1.0 + g_shock[game_idx] + t_shock[team_idx]
        mult = np.clip(mult, 0.72, 1.35)

        mu = proj * base_mult * mult
        noise = rng.normal(0.0, 1.0, size=len(proj)) * sd_eff
        sim_scores = np.clip(mu + noise, 0.0, None)

        field_scores = sim_scores[field_idx9].sum(axis=1)
        port_scores = sim_scores[portfolio_idx9].sum(axis=1)

        all_scores = np.concatenate([field_scores, port_scores], axis=0)
        order = np.argsort(-all_scores)
        ranks = np.empty_like(order)
        ranks[order] = np.arange(1, total_sim + 1)

        port_ranks_sim = ranks[F:]

        denom = max(1, total_sim - 1)
        pct = (port_ranks_sim - 1) / denom
        contest_ranks = 1 + (pct * (max(1, contest.field_size - 1))).astype(int)

        pay = 0.0
        for rnk in contest_ranks.tolist():
            pay += payout_for_rank(contest, int(rnk))
        roi_vals[s] = (pay / (contest.entry_fee * P)) - 1.0

        if (s + 1) % 25 == 0:
            elapsed = (datetime.now() - t0).total_seconds()
            _print(f"[roi] {s+1}/{roi_sims} sims | elapsed={elapsed:.1f}s | field_pool={F}")

    return {
        "roi_mean": float(np.mean(roi_vals)),
        "roi_p50": float(np.quantile(roi_vals, 0.50)),
        "roi_p90": float(np.quantile(roi_vals, 0.90)),
        "roi_sims": int(roi_sims),
        "field_pool": int(F),
    }


# -------------------------
# Main API
# -------------------------
def run_slate_sim(
    clean_csv: str = "clean.csv",
    contest_pick: str = "piggy",
    team_lock_minutes: Optional[Dict[str, int]] = None,
    current_lock_minute: Optional[int] = None,
    seed: int = 7,
    entries_you: int = 150,
    exposure_min: Optional[Dict[str, float]] = None,
    exposure_max: Optional[Dict[str, float]] = None,
    output_dir: str = "output",
    uploaded_fd_csv: Optional[str] = None,

    # core sim generation
    num_sims: int = 2200,

    # optimizer tuning
    topk_per_slot: int = 52,
    attempts: int = 12,
    salary_bonus: float = 0.00105,
    chalk_penalty: float = 0.0,

    # variance model
    game_sigma: float = 0.075,
    team_sigma: float = 0.045,
    pace_alpha: float = 0.90,
    def_beta: float = 0.70,

    # NEW: spike knobs (kept cheap)
    heavy_tail_prob: float = 0.08,   # fraction of sims with heavier tails / chaos
    heavy_tail_scale: float = 1.35,  # multiply sd_eff in those sims
    t_df: int = 5,                   # t-dist df for heavy tail noise

    # NEW: optimizer ceiling knobs
    ceiling_w: float = 0.18,
    stocks_w: float = 0.35,
    game_stack_w: float = 1.0,
    team4_pen_w: float = 1.0,

    # ROI eval
    do_roi: bool = True,
    roi_sims: int = 80,
    field_pool_size: int = 1200,

    # selection blending (cheap ceiling proxy)
    sel_hit_w: float = 1.0,
    sel_proj_w: float = 0.35,
    sel_maxsim_w: float = 0.55,
) -> Dict[str, object]:
    _print("=" * 70)
    _print(f"[sim_prep] run_slate_sim start | {_ts()} | contest_pick={contest_pick} | seed={seed}")
    _print("=" * 70)

    rng = np.random.default_rng(seed)

    contest = load_contest(contest_pick)
    _print(f"[sim_prep] Contest (for summary only): {contest.name}")
    _print(f"[sim_prep] entries_you={entries_you} | cap={FD_SALARY_CAP} | min_sal={FD_MIN_SALARY}")

    _print(f"[sim_prep] Loading {clean_csv} ...")
    clean = load_clean(clean_csv)

    exposure_min = exposure_min or {}
    exposure_max = exposure_max or {}
    clean = _apply_punt_fade_filter(clean_df=clean, exposure_min=exposure_min)

    fd_id = clean["FD_ID"].to_numpy(dtype=str)
    player_full = clean["PLAYER_FULL"].to_numpy(dtype=str)
    name_key = clean["NAME_KEY"].to_numpy(dtype=str)

    team_fd_arr = clean["TEAM_FD"].to_numpy(dtype=str)
    team_canon_arr = clean["TEAM_CANON"].to_numpy(dtype=str)
    fd_game_arr = clean["FD_GAME"].to_numpy(dtype=str)

    salary = clean["SALARY"].to_numpy(dtype=int)
    proj = clean["PROJECTION"].to_numpy(dtype=float)
    own = clean["OWN_PCT"].to_numpy(dtype=float)
    elig_lists = clean["ELIG"].tolist()

    pace_raw = clean["PACE_RAW"].to_numpy(dtype=float)
    def_raw = clean["DEF_RAW"].to_numpy(dtype=float)

    stocks = clean["PROJ_STOCKS"].to_numpy(dtype=float)

    sd_in = clean["FD_SD"].to_numpy(dtype=float)
    if np.isnan(sd_in).all():
        sd = build_sd_fallback(proj)
        _print("[sim_prep] FD_SD not found/usable -> using fallback SD model.")
    else:
        sd = np.where(np.isfinite(sd_in) & (sd_in > 0), sd_in, build_sd_fallback(proj))
        _print("[sim_prep] Using FD_SD (fallback where missing).")

    # Environment multipliers
    base_mult = (np.clip(pace_raw, 0.80, 1.30) ** pace_alpha) * (np.clip(def_raw, 0.70, 1.40) ** def_beta)
    base_mult = np.clip(base_mult, 0.78, 1.28)

    # Make stocks contribute to volatility slightly (cheap and effective)
    # stocks_z in [0..~2], then bump sd_eff modestly
    stocks_z = (stocks - float(np.mean(stocks))) / float(np.std(stocks) + 1e-9)
    stocks_z = np.clip(stocks_z, -2.0, 2.5)
    sd_eff = sd * (1.0 + 0.10 * np.maximum(stocks_z, 0.0))

    id_to_player = {pid: nm for pid, nm in zip(fd_id, player_full)}
    id_to_team_canon = {pid: tm for pid, tm in zip(fd_id, team_canon_arr)}
    name_key_by_id = _build_name_key_by_id(fd_id, name_key)

    elig_idx = build_slot_indices(elig_lists)
    for s, need in SLOT_COUNTS.items():
        if elig_idx.get(s, np.array([], dtype=int)).size < need:
            raise ValueError(f"Not enough eligible players for slot={s}. eligible={elig_idx.get(s, np.array([])).size} need={need}")

    games_unique, game_idx = np.unique(fd_game_arr, return_inverse=True)
    team_unique, team_idx = np.unique(team_canon_arr, return_inverse=True)
    game_idx = game_idx.astype(int)
    team_idx = team_idx.astype(int)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path_lineups = out_dir / "report_lineups.csv"
    path_expo = out_dir / "report_exposures.csv"
    path_summary = out_dir / "summary.csv"
    path_upload = out_dir / "fd_upload.csv"
    path_edit = out_dir / "fd_edit.csv"
    path_roi = out_dir / "roi_summary.csv"

    locked_teams = set(compute_locked_teams(team_lock_minutes, current_lock_minute))

    def _expand_locked_teams_to_opponents(clean_df: pd.DataFrame, locked: set[str]) -> set[str]:
        out = set(locked)
        for g in clean_df["FD_GAME"].astype(str).str.upper().tolist():
            a, h = parse_fd_game(g)
            if not a or not h:
                continue
            a_can, h_can = canon_team(a), canon_team(h)
            if a_can in out or h_can in out:
                out.add(a_can)
                out.add(h_can)
        return out

    locked_teams = _expand_locked_teams_to_opponents(clean, locked_teams)

    if locked_teams:
        _print(f"[sim_prep] Locked cutoff minute={current_lock_minute} -> locked_teams={sorted(locked_teams)}")
    else:
        _print("[sim_prep] No locked teams.")

    # ---------------------------------------------------------
    # MODE A: Edit Entries
    # ---------------------------------------------------------
    if uploaded_fd_csv:
        _print(f"[sim_prep] Uploaded/edit mode ON | reading: {uploaded_fd_csv}")
        up_df, _slot_keys = _read_uploaded_entries(uploaded_fd_csv)
        n_rows = len(up_df)
        _print(f"[sim_prep] Uploaded usable rows: {n_rows}")
        if n_rows == 0:
            raise ValueError("No usable entry rows found (stopped at first blank entry_id).")

        protected_mask = up_df["contest_name"].map(_is_cash_protected_contest)
        _print(f"[sim_prep] PROTECTED rows: {int(protected_mask.sum())}/{n_rows} -> unchanged")

        fd_id_to_idx: Dict[str, int] = {str(pid): int(i) for i, pid in enumerate(fd_id.tolist())}
        up_slots = up_df[SLOT_KEYS].copy().fillna("").astype(str)

        missing_locked: set[str] = set()
        locked_slot_idxs: List[Dict[int, int]] = []

        for r in range(n_rows):
            if bool(protected_mask.iloc[r]):
                locked_slot_idxs.append({})
                continue

            locked_map: Dict[int, int] = {}
            row_ids = [_clean_player_cell_to_id(up_slots.iloc[r, j]) for j in range(9)]
            for j, pid in enumerate(row_ids):
                if not pid:
                    continue
                if locked_teams and _is_locked_team_for_pid(pid, id_to_team_canon, locked_teams):
                    if pid not in fd_id_to_idx:
                        missing_locked.add(pid)
                        continue
                    locked_map[j] = fd_id_to_idx[pid]
            locked_slot_idxs.append(locked_map)

        if missing_locked:
            raise ValueError(
                "Late swap error: LOCKED FD_IDs in uploaded file not present in clean.csv:\n"
                + "\n".join(sorted(missing_locked))
            )

        sims_per_row = 80  # keep fast
        _print(f"[sim_prep] Building edit lineups... num_rows={n_rows} | sims_per_row={sims_per_row}")

        rows_out_ids: List[List[str]] = []
        pretty_rows: List[Dict[str, Any]] = []
        loop_t0 = datetime.now()

        for r in range(n_rows):
            if bool(protected_mask.iloc[r]):
                ids_slot = [_clean_player_cell_to_id(up_slots.iloc[r, j]) for j in range(9)]
                rows_out_ids.append(ids_slot)
                pretty_rows.append(
                    {"Row": r + 1, "LockedSlots": "PROTECTED", "Salary": "", "SimPts": "", "Players": "PROTECTED", "FD_IDs": ",".join(ids_slot)}
                )
                continue

            locked_map = locked_slot_idxs[r]
            best = None
            best_obj = -1e18

            for _s in range(sims_per_row):
                g_shock = rng.normal(0.0, game_sigma, size=len(games_unique))
                t_shock = rng.normal(0.0, team_sigma, size=len(team_unique))

                mult = 1.0 + g_shock[game_idx] + t_shock[team_idx]
                mult = np.clip(mult, 0.72, 1.35)

                mu = proj * base_mult * mult

                # heavy tail mix (cheap)
                if rng.random() < heavy_tail_prob:
                    # t noise is heavier tail than normal; scale up
                    noise = rng.standard_t(df=t_df, size=len(proj)) * (sd_eff * heavy_tail_scale)
                else:
                    noise = rng.normal(0.0, 1.0, size=len(proj)) * sd_eff

                sim_scores = np.clip(mu + noise, 0.0, None)

                picked_idx = optimize_lineup_one_sim_with_locks(
                    rng=rng,
                    scores=sim_scores,
                    salary=salary,
                    own=own,
                    stocks=stocks,
                    sd_eff=sd_eff,
                    fd_game_arr=fd_game_arr,
                    elig_idx=elig_idx,
                    locked_slot_idx=locked_map,
                    team_rule=team_canon_arr,
                    team_fd=team_fd_arr,
                    cap=FD_SALARY_CAP,
                    min_sal=FD_MIN_SALARY,
                    topk_per_slot=topk_per_slot,
                    attempts=attempts,
                    salary_bonus=salary_bonus,
                    chalk_penalty=chalk_penalty,
                    ceiling_w=ceiling_w,
                    stocks_w=stocks_w,
                    game_stack_w=game_stack_w,
                    team4_pen_w=team4_pen_w,
                )
                if picked_idx is None:
                    continue

                ids_slot = [fd_id[i] for i in picked_idx]
                ok_lock = True
                for j, pi in locked_map.items():
                    if ids_slot[j] != fd_id[int(pi)]:
                        ok_lock = False
                        break
                if not ok_lock:
                    continue

                ssum = int(np.sum(salary[picked_idx]))
                pts = float(np.sum(sim_scores[picked_idx]))

                # same objective for selection
                obj = pts + salary_bonus * ssum
                obj += game_stack_w * _game_stack_bonus(fd_game_arr, picked_idx)
                obj -= team4_pen_w * _discourage_4_stack(team_canon_arr, picked_idx)

                if obj > best_obj:
                    best_obj = obj
                    best = (picked_idx, ids_slot, ssum, pts)

            if best is None:
                raise RuntimeError(
                    f"Edit mode failed: could not build a valid lineup for row {r+1}/{n_rows}. "
                    f"Locks + team/salary constraints likely impossible."
                )

            picked_idx, ids_slot, ssum, pts = best
            rows_out_ids.append(ids_slot)

            names = [player_full[i] for i in picked_idx]
            teams = [team_canon_arr[i] for i in picked_idx]
            pretty_rows.append(
                {
                    "Row": r + 1,
                    "LockedSlots": ",".join([SLOT_KEYS[j] for j in sorted(locked_map.keys())]) if locked_map else "",
                    "Salary": ssum,
                    "SimPts": round(float(pts), 2),
                    "Players": " | ".join([f"{nm} ({tm})" for nm, tm in zip(names, teams)]),
                    "FD_IDs": ",".join(ids_slot),
                }
            )

            if (r + 1) % 25 == 0 or (r + 1) == n_rows:
                elapsed = (datetime.now() - loop_t0).total_seconds()
                _print(f"[edit] built {r+1}/{n_rows} | {elapsed:.2f}s elapsed")

        out_df = up_df.copy()
        for j, k in enumerate(SLOT_KEYS):
            out_df[k] = [row[j] for row in rows_out_ids]

        lineups_df = pd.DataFrame(pretty_rows)
        exp_df = exposure_df([{"fd_ids": row, "key": _lineup_key(row)} for row in rows_out_ids], id_to_player=id_to_player, entries_you=len(rows_out_ids))

        summary_df = pd.DataFrame(
            [
                {"Metric": "Mode", "Value": "edit"},
                {"Metric": "Contest (for summary only)", "Value": contest.name},
                {"Metric": "Rows (uploaded usable)", "Value": n_rows},
                {"Metric": "Protected rows", "Value": int(protected_mask.sum())},
                {"Metric": "Locked cutoff minute", "Value": current_lock_minute if current_lock_minute is not None else ""},
                {"Metric": "Locked teams", "Value": " ".join(sorted(locked_teams)) if locked_teams else ""},
                {"Metric": "Min Salary", "Value": FD_MIN_SALARY},
                {"Metric": "Salary Cap", "Value": FD_SALARY_CAP},
                {"Metric": "Max players per team", "Value": FD_MAX_PER_TEAM},
                {"Metric": "Ceiling weight", "Value": ceiling_w},
                {"Metric": "Stocks weight", "Value": stocks_w},
                {"Metric": "Game-stack weight", "Value": game_stack_w},
                {"Metric": "4-stack penalty weight", "Value": team4_pen_w},
                {"Metric": "Wrote fd_edit.csv", "Value": str(path_edit)},
            ]
        )

        lineups_df.to_csv(path_lineups, index=False)
        exp_df.to_csv(path_expo, index=False)
        summary_df.to_csv(path_summary, index=False)
        _write_fd_upload_csv(path_upload, rows_out_ids)
        _write_fd_edit_csv(path_edit, out_df)

        roi_df = pd.DataFrame([{"Metric": "roi_skipped", "Value": "edit_mode"}])
        roi_df.to_csv(path_roi, index=False)

        _print(f"[OK] wrote {path_lineups}")
        _print(f"[OK] wrote {path_expo}")
        _print(f"[OK] wrote {path_summary}")
        _print(f"[OK] wrote {path_upload}")
        _print(f"[OK] wrote {path_edit}")
        _print(f"[OK] wrote {path_roi}")

        return {
            "lineups_df": lineups_df,
            "exposures_df": exp_df,
            "summary_df": summary_df,
            "paths": {
                "lineups_csv": str(path_lineups),
                "exposures_csv": str(path_expo),
                "summary_csv": str(path_summary),
                "roi_summary_csv": str(path_roi),
                "fd_upload_csv": str(path_upload),
                "fd_edit_csv": str(path_edit),
            },
            "locked_teams": sorted(locked_teams),
        }

    # ---------------------------------------------------------
    # MODE B: Normal mode
    # ---------------------------------------------------------
    _print("[sim_prep] Normal mode. Simulating + building lineups...")
    _print(f"[sim_prep] num_sims={num_sims} | prints every 250")

    hits: Dict[Tuple[str, ...], Dict[str, Any]] = {}
    field_pool: List[List[int]] = []

    loop_t0 = datetime.now()
    for s in range(num_sims):
        g_shock = rng.normal(0.0, game_sigma, size=len(games_unique))
        t_shock = rng.normal(0.0, team_sigma, size=len(team_unique))

        mult = 1.0 + g_shock[game_idx] + t_shock[team_idx]
        mult = np.clip(mult, 0.72, 1.35)

        mu = proj * base_mult * mult

        # heavy tail mix (cheap)
        if rng.random() < heavy_tail_prob:
            noise = rng.standard_t(df=t_df, size=len(proj)) * (sd_eff * heavy_tail_scale)
        else:
            noise = rng.normal(0.0, 1.0, size=len(proj)) * sd_eff

        sim_scores = np.clip(mu + noise, 0.0, None)

        picked_idx = optimize_lineup_one_sim_with_locks(
            rng=rng,
            scores=sim_scores,
            salary=salary,
            own=own,
            stocks=stocks,
            sd_eff=sd_eff,
            fd_game_arr=fd_game_arr,
            elig_idx=elig_idx,
            locked_slot_idx={},
            team_rule=team_canon_arr,
            team_fd=team_fd_arr,
            cap=FD_SALARY_CAP,
            min_sal=FD_MIN_SALARY,
            topk_per_slot=topk_per_slot,
            attempts=attempts,
            salary_bonus=salary_bonus,
            chalk_penalty=chalk_penalty,
            ceiling_w=ceiling_w,
            stocks_w=stocks_w,
            game_stack_w=game_stack_w,
            team4_pen_w=team4_pen_w,
        )
        if picked_idx is None:
            continue

        picked_ids_in_slot_order = [fd_id[i] for i in picked_idx]
        if len(picked_ids_in_slot_order) != 9 or len(set(picked_ids_in_slot_order)) != 9:
            continue
        if not _team_counts_ok(team_canon_arr, picked_idx, FD_MAX_PER_TEAM):
            continue
        if ENFORCE_FD_TEAM_ALSO and not _team_counts_ok(team_fd_arr, picked_idx, FD_MAX_PER_TEAM):
            continue

        # store field proxy
        if len(field_pool) < field_pool_size:
            field_pool.append(list(picked_idx))
        elif (s % 13) == 0:
            j = int(rng.integers(0, field_pool_size))
            field_pool[j] = list(picked_idx)

        sim_pts = float(np.sum(sim_scores[picked_idx]))

        key = _lineup_key(picked_ids_in_slot_order)
        if key not in hits:
            hits[key] = {
                "key": key,
                "hit_count": 0,
                "idx9": list(picked_idx),
                "fd_ids_in_slot_order": picked_ids_in_slot_order,
                "fd_ids": list(key),
                "proj_sum": float(np.sum(proj[picked_idx])),
                "salary_sum": int(np.sum(salary[picked_idx])),
                "max_sim_pts": sim_pts,     # CHEAP ceiling proxy
                "sum_sim_pts": sim_pts,
            }
        else:
            hits[key]["max_sim_pts"] = max(float(hits[key]["max_sim_pts"]), sim_pts)
            hits[key]["sum_sim_pts"] = float(hits[key]["sum_sim_pts"]) + sim_pts

        hits[key]["hit_count"] += 1

        if (s + 1) % 250 == 0:
            elapsed = (datetime.now() - loop_t0).total_seconds()
            _print(f"[sim] {s+1}/{num_sims} | unique={len(hits)} | field_pool={len(field_pool)} | {elapsed:.2f}s")

    if not hits:
        raise RuntimeError("No lineups generated. Something is wrong with eligibility or projections.")

    candidates = list(hits.values())
    _print(f"[sim_prep] Candidate unique lineups: {len(candidates)}")

    exp_min = _normalize_exp_dict(exposure_min)
    exp_max = _normalize_exp_dict(exposure_max)

    # Selection score: hit_count + proj + max_sim_pts
    # This is where we stop being “median-only” without extra expensive loops.
    max_hit = max(1, max(x["hit_count"] for x in candidates))
    max_proj = max(1e-9, max(x["proj_sum"] for x in candidates))
    max_maxsim = max(1e-9, max(x["max_sim_pts"] for x in candidates))

    for x in candidates:
        x["sel_score"] = (
            sel_hit_w * (float(x["hit_count"]) / max_hit)
            + sel_proj_w * (float(x["proj_sum"]) / max_proj)
            + sel_maxsim_w * (float(x["max_sim_pts"]) / max_maxsim)
        )

    cand_sorted = sorted(candidates, key=lambda x: (x["sel_score"], x["hit_count"], x["max_sim_pts"], x["proj_sum"]), reverse=True)

    selected: List[Dict[str, Any]] = []
    used_keys = set()
    counts: Dict[str, int] = {}

    def would_violate_max(lu: Dict[str, Any]) -> bool:
        if not exp_max:
            return False
        for pid in lu["fd_ids"]:
            nk = name_key_by_id.get(pid, "")
            if nk in exp_max:
                if (counts.get(nk, 0) + 1) / entries_you > exp_max[nk] + 1e-12:
                    return True
        return False

    for lu in cand_sorted:
        if len(selected) >= entries_you:
            break
        k = lu["key"]
        if k in used_keys:
            continue
        if would_violate_max(lu):
            continue
        selected.append(lu)
        used_keys.add(k)
        for pid in lu["fd_ids"]:
            nk = name_key_by_id.get(pid, "")
            counts[nk] = counts.get(nk, 0) + 1

    if len(selected) < entries_you:
        for lu in cand_sorted:
            if len(selected) >= entries_you:
                break
            k = lu["key"]
            if k in used_keys:
                continue
            selected.append(lu)
            used_keys.add(k)
            for pid in lu["fd_ids"]:
                nk = name_key_by_id.get(pid, "")
                counts[nk] = counts.get(nk, 0) + 1

    if exp_min:
        need = {nk: _count_needed(entries_you, frac) for nk, frac in exp_min.items()}

        def has_player(lu: Dict[str, Any], nk: str) -> bool:
            return any(name_key_by_id.get(pid, "") == nk for pid in lu["fd_ids"])

        for nk, req in need.items():
            while counts.get(nk, 0) < req:
                pool_add = [lu for lu in cand_sorted if (lu["key"] not in used_keys) and has_player(lu, nk)]
                if not pool_add:
                    break
                pool_add.sort(key=lambda x: (x["sel_score"], x["hit_count"], x["max_sim_pts"]), reverse=True)
                lu_in = pool_add[0]

                removable = [lu for lu in selected if not has_player(lu, nk)]
                if not removable:
                    break
                removable.sort(key=lambda x: (x["sel_score"], x["hit_count"], x["max_sim_pts"]))
                lu_out = removable[0]

                selected.remove(lu_out)
                used_keys.remove(lu_out["key"])
                for pid in lu_out["fd_ids"]:
                    nk2 = name_key_by_id.get(pid, "")
                    counts[nk2] = counts.get(nk2, 0) - 1

                selected.append(lu_in)
                used_keys.add(lu_in["key"])
                for pid in lu_in["fd_ids"]:
                    nk2 = name_key_by_id.get(pid, "")
                    counts[nk2] = counts.get(nk2, 0) + 1

    # Outputs
    pretty_rows = []
    slot_rows: List[List[str]] = []
    portfolio_idx9: List[List[int]] = []

    for i, lu in enumerate(selected, 1):
        ids_slot = lu["fd_ids_in_slot_order"]
        idx9 = lu["idx9"]
        portfolio_idx9.append(idx9)

        names = [player_full[j] for j in idx9]
        teams = [team_canon_arr[j] for j in idx9]
        pretty_rows.append(
            {
                "Lineup #": i,
                "SelScore": round(float(lu["sel_score"]), 4),
                "HitCount": lu["hit_count"],
                "Salary": lu["salary_sum"],
                "ProjSum": round(lu["proj_sum"], 2),
                "MaxSimPts": round(float(lu["max_sim_pts"]), 2),
                "Players": " | ".join([f"{nm} ({tm})" for nm, tm in zip(names, teams)]),
                "FD_IDs": ",".join(ids_slot),
            }
        )
        slot_rows.append(ids_slot)

    lineups_df = pd.DataFrame(pretty_rows).sort_values(["SelScore", "HitCount", "MaxSimPts", "ProjSum"], ascending=[False, False, False, False]).reset_index(drop=True)
    exp_df = exposure_df(selected, id_to_player=id_to_player, entries_you=len(selected))

    avg_salary = float(lineups_df["Salary"].mean()) if len(lineups_df) else 0.0
    min_salary = int(lineups_df["Salary"].min()) if len(lineups_df) else 0

    roi_metrics = {"roi_mean": 0.0, "roi_p50": 0.0, "roi_p90": 0.0, "roi_sims": 0, "field_pool": 0}
    if do_roi and field_pool and portfolio_idx9:
        _print(f"[sim_prep] ROI eval -> roi_sims={roi_sims}, field_pool={len(field_pool)}")
        roi_metrics = _roi_eval_fast(
            rng=rng,
            contest=contest,
            proj=proj,
            sd_eff=sd_eff,
            base_mult=base_mult,
            game_idx=game_idx,
            team_idx=team_idx,
            n_games=len(games_unique),
            n_teams=len(team_unique),
            portfolio_idx9=np.array(portfolio_idx9, dtype=int),
            field_idx9=np.array(field_pool[:field_pool_size], dtype=int),
            roi_sims=roi_sims,
            game_sigma=game_sigma,
            team_sigma=team_sigma,
        )
    else:
        _print("[sim_prep] ROI eval skipped.")

    summary_df = pd.DataFrame(
        [
            {"Metric": "Mode", "Value": "normal"},
            {"Metric": "Contest (for summary only)", "Value": contest.name},
            {"Metric": "Entries You", "Value": entries_you},
            {"Metric": "Unique Lineups Returned", "Value": len(selected)},
            {"Metric": "Salary Cap", "Value": FD_SALARY_CAP},
            {"Metric": "Min Salary", "Value": FD_MIN_SALARY},
            {"Metric": "Avg Salary (portfolio)", "Value": round(avg_salary, 1)},
            {"Metric": "Min Salary (portfolio)", "Value": min_salary},
            {"Metric": "Candidate Unique Lineups", "Value": len(candidates)},
            {"Metric": "Num Sims", "Value": num_sims},
            {"Metric": "Field Pool Size", "Value": len(field_pool)},
            {"Metric": "Max players per team", "Value": FD_MAX_PER_TEAM},
            {"Metric": "Ceiling weight", "Value": ceiling_w},
            {"Metric": "Stocks weight", "Value": stocks_w},
            {"Metric": "Game-stack weight", "Value": game_stack_w},
            {"Metric": "4-stack penalty weight", "Value": team4_pen_w},
            {"Metric": "Heavy-tail prob", "Value": heavy_tail_prob},
            {"Metric": "Heavy-tail scale", "Value": heavy_tail_scale},
            {"Metric": "ROI sims", "Value": roi_metrics.get("roi_sims", 0)},
            {"Metric": "ROI mean", "Value": round(float(roi_metrics.get("roi_mean", 0.0)), 4)},
            {"Metric": "ROI p50", "Value": round(float(roi_metrics.get("roi_p50", 0.0)), 4)},
            {"Metric": "ROI p90", "Value": round(float(roi_metrics.get("roi_p90", 0.0)), 4)},
        ]
    )

    roi_df = pd.DataFrame(
        [
            {"Metric": "roi_mean", "Value": roi_metrics.get("roi_mean", 0.0)},
            {"Metric": "roi_p50", "Value": roi_metrics.get("roi_p50", 0.0)},
            {"Metric": "roi_p90", "Value": roi_metrics.get("roi_p90", 0.0)},
            {"Metric": "roi_sims", "Value": roi_metrics.get("roi_sims", 0)},
            {"Metric": "field_pool", "Value": roi_metrics.get("field_pool", 0)},
        ]
    )

    lineups_df.to_csv(path_lineups, index=False)
    exp_df.to_csv(path_expo, index=False)
    summary_df.to_csv(path_summary, index=False)
    roi_df.to_csv(path_roi, index=False)
    _write_fd_upload_csv(path_upload, slot_rows)

    _print(f"[OK] wrote {path_lineups}")
    _print(f"[OK] wrote {path_expo}")
    _print(f"[OK] wrote {path_summary}")
    _print(f"[OK] wrote {path_roi}")
    _print(f"[OK] wrote {path_upload}")
    _print(f"[sim_prep] done | unique={len(selected)} | avg_salary={avg_salary:.1f} | min_salary={min_salary}")

    return {
        "lineups_df": lineups_df,
        "exposures_df": exp_df,
        "summary_df": summary_df,
        "roi_summary_df": roi_df,
        "paths": {
            "lineups_csv": str(path_lineups),
            "exposures_csv": str(path_expo),
            "summary_csv": str(path_summary),
            "roi_summary_csv": str(path_roi),
            "fd_upload_csv": str(path_upload),
        },
        "locked_teams": sorted(locked_teams),
    }
