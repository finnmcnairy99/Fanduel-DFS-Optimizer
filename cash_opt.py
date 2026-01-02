# cash_opt.py
# Exact cash optimal lineup (max projected) for FanDuel NBA Classic
# Reads clean.csv produced by model.py (or any similarly-formatted csv)
#
# Install:
#   pip install pulp
#
# Notebook usage:
#   from cash_opt import cash_optimal_lineup
#   lineup = cash_optimal_lineup("clean.csv", locks=["Nikola Jokic", "Jalen Brunson"])
#   lineup
#
# CLI usage:
#   python cash_opt.py --csv clean.csv --lock "Nikola Jokic" --lock "Jalen Brunson"

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import pulp
except ImportError as e:
    raise ImportError(
        "Missing dependency: pulp. Install with:\n\n"
        "  pip install pulp\n"
    ) from e


# -------------------------
# FanDuel NBA roster (classic)
# -------------------------
ROSTER_SLOTS = ["PG", "PG", "SG", "SG", "SF", "SF", "PF", "PF", "C"]
FD_SALARY_CAP = 60000
FD_MAX_PER_TEAM = 4


# -------------------------
# Team aliasing (FD <-> canon)
# -------------------------
FD_TO_CANON = {"GS": "GSW", "NO": "NOP", "NY": "NYK", "PHO": "PHX", "SA": "SAS"}
CANON_TO_FD = {v: k for k, v in FD_TO_CANON.items()}


# -------------------------
# Helpers
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
    s = re.sub(r"\b([a-z])\s+([a-z])\b", r"\1\2", s)
    suffixes = {"jr", "sr", "ii", "iii", "iv", "v"}
    toks = [t for t in s.split() if t not in suffixes]
    return " ".join(toks)


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


def _looks_like_fd_id(s: str) -> bool:
    s = str(s).strip()
    return bool(re.match(r"^\d+-\d+$", s))


# -------------------------
# Load / validate
# -------------------------
def load_clean(clean_csv: str | Path) -> pd.DataFrame:
    df = pd.read_csv(clean_csv)

    required = ["FD_ID", "PLAYER_FULL", "TEAM", "POS", "FD_GAME", "SALARY", "PROJECTION"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"clean.csv missing columns: {missing}\nfound: {list(df.columns)}")

    df = df.copy()
    df["FD_ID"] = df["FD_ID"].astype(str).str.strip()
    df["PLAYER_FULL"] = df["PLAYER_FULL"].astype(str).str.strip()

    if "NAME_KEY" not in df.columns:
        df["NAME_KEY"] = df["PLAYER_FULL"].map(normalize_name)
    else:
        df["NAME_KEY"] = df["NAME_KEY"].astype(str).map(normalize_name)

    df["TEAM"] = df["TEAM"].astype(str).str.upper().str.strip().map(canon_team)
    df["FD_GAME"] = df["FD_GAME"].astype(str).str.upper().str.strip()

    df["SALARY"] = df["SALARY"].map(as_float).fillna(0.0).astype(int)
    df["PROJECTION"] = df["PROJECTION"].map(as_float).fillna(0.0).astype(float)

    df["ELIG"] = df["POS"].map(split_pos)
    df["TEAM_FD"] = df.apply(lambda r: team_from_fd_game_for_player(r["TEAM"], r["FD_GAME"]), axis=1)
    df["TEAM_FD"] = df["TEAM_FD"].astype(str).str.upper().str.strip()

    # sanity filters
    df = df[df["FD_ID"].astype(str).str.len() > 0].copy()
    df = df[df["SALARY"] > 0].copy()
    df = df[df["PROJECTION"] > 0].copy()
    df = df[df["ELIG"].map(len) > 0].copy()
    df = df.reset_index(drop=True)

    return df


def resolve_locks(df: pd.DataFrame, locks: List[str] | None) -> List[int]:
    """
    Locks can be:
      - exact PLAYER_FULL string
      - name-ish string (matched via NAME_KEY normalize_name)
      - FD_ID-like "12345-67890"
    Must resolve to exactly one row each (no guessing).
    """
    if not locks:
        return []

    # maps -> list of indices (so we can detect ambiguity)
    namekey_to_idx: Dict[str, List[int]] = {}
    full_to_idx: Dict[str, List[int]] = {}
    fdid_to_idx: Dict[str, List[int]] = {}

    for i in range(len(df)):
        nk = str(df.loc[i, "NAME_KEY"]).strip()
        if nk:
            namekey_to_idx.setdefault(nk, []).append(i)

        pf = str(df.loc[i, "PLAYER_FULL"]).strip().lower()
        if pf:
            full_to_idx.setdefault(pf, []).append(i)

        fid = str(df.loc[i, "FD_ID"]).strip()
        if fid:
            fdid_to_idx.setdefault(fid, []).append(i)

    resolved: List[int] = []
    for raw in locks:
        raw_s = str(raw).strip()
        if not raw_s:
            continue

        candidates: List[int] = []
        if _looks_like_fd_id(raw_s):
            candidates = fdid_to_idx.get(raw_s, [])
        else:
            # try normalized
            nk = normalize_name(raw_s)
            candidates = namekey_to_idx.get(nk, [])
            if not candidates:
                # try exact PLAYER_FULL (case-insensitive)
                candidates = full_to_idx.get(raw_s.lower(), [])

        if len(candidates) == 0:
            raise ValueError(f"Lock not found in clean.csv: {raw_s}")

        if len(candidates) > 1:
            options = df.loc[candidates, ["PLAYER_FULL", "TEAM", "POS", "FD_GAME", "SALARY", "PROJECTION"]]
            raise ValueError(
                f"Lock matched multiple rows for: {raw_s}\n\n"
                f"{options.to_string(index=True)}\n\n"
                f"Be more specific (exact PLAYER_FULL) or lock by FD_ID."
            )

        resolved.append(candidates[0])

    # detect duplicates
    if len(set(resolved)) != len(resolved):
        raise ValueError("Duplicate locks detected (same player locked twice).")

    # quick feasibility sanity: if a lock has no eligible variable created, solver can't pick it anyway
    return resolved


# -------------------------
# Optimizer
# -------------------------
def cash_optimal_lineup(
    clean_csv: str | Path = "clean.csv",
    locks: List[str] | None = None,
    cap: int = FD_SALARY_CAP,
    max_per_team: int | None = FD_MAX_PER_TEAM,
) -> pd.DataFrame:
    """
    Returns a roster-ordered DataFrame with:
      Slot, PLAYER_FULL, TEAM, POS, FD_GAME, SALARY, PROJECTION, VALUE
    plus a TOTAL row.

    - locks: list of players to force into lineup (PLAYER_FULL / name / FD_ID)
    """
    df = load_clean(clean_csv)
    lock_idxs = resolve_locks(df, locks)

    n = len(df)
    slots = list(range(len(ROSTER_SLOTS)))  # 0..8

    # decision vars: x[i,j] = player i used in slot j
    x: Dict[Tuple[int, int], pulp.LpVariable] = {}
    for i in range(n):
        elig = set(df.loc[i, "ELIG"])
        for j in slots:
            slot_pos = ROSTER_SLOTS[j]
            if slot_pos in elig:
                x[(i, j)] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, upBound=1, cat="Binary")

    # If some slot has no eligible players, hard fail early with a readable message.
    for j in slots:
        if not any((i, j) in x for i in range(n)):
            raise RuntimeError(f"No eligible players found to fill slot {j} ({ROSTER_SLOTS[j]}).")

    prob = pulp.LpProblem("FD_Cash_Optimal", pulp.LpMaximize)

    proj = df["PROJECTION"].to_numpy(float)
    sal = df["SALARY"].to_numpy(int)

    # objective: maximize total projection
    prob += pulp.lpSum(proj[i] * x[(i, j)] for (i, j) in x.keys())

    # each roster slot filled exactly once
    for j in slots:
        prob += pulp.lpSum(x[(i, j)] for i in range(n) if (i, j) in x) == 1, f"fill_slot_{j}"

    # each player used at most once
    for i in range(n):
        prob += pulp.lpSum(x[(i, j)] for j in slots if (i, j) in x) <= 1, f"unique_player_{i}"

    # salary cap
    prob += pulp.lpSum(sal[i] * x[(i, j)] for (i, j) in x.keys()) <= int(cap), "salary_cap"

    # max per team (FanDuel team abbrev)
    if max_per_team is not None:
        for team in sorted(df["TEAM_FD"].unique().tolist()):
            idx = df.index[df["TEAM_FD"] == team].tolist()
            prob += (
                pulp.lpSum(x[(i, j)] for i in idx for j in slots if (i, j) in x) <= int(max_per_team)
            ), f"max_team_{team}"

    # locks: each locked player must be used in exactly one eligible slot
    for i in lock_idxs:
        elig_vars = [x[(i, j)] for j in slots if (i, j) in x]
        if not elig_vars:
            raise RuntimeError(
                f"Locked player has no eligible slots based on POS/ELIG: {df.loc[i, 'PLAYER_FULL']} ({df.loc[i, 'POS']})"
            )
        prob += pulp.lpSum(elig_vars) == 1, f"lock_player_{i}"

    # solve
    solver = pulp.PULP_CBC_CMD(msg=False)
    status = prob.solve(solver)
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"Cash optimizer failed. Solver status: {pulp.LpStatus[status]}")

    # extract chosen per slot
    chosen: List[Tuple[int, int]] = []
    for j in slots:
        picked_i = None
        for i in range(n):
            var = x.get((i, j))
            if var is not None and var.value() == 1:
                picked_i = i
                break
        if picked_i is None:
            raise RuntimeError(f"Solver returned no player for slot {j} ({ROSTER_SLOTS[j]})")
        chosen.append((j, picked_i))

    # output (no FD_ID)
    rows = []
    for j, i in sorted(chosen, key=lambda t: t[0]):
        rows.append(
            {
                "Slot": ROSTER_SLOTS[j],
                "PLAYER_FULL": df.loc[i, "PLAYER_FULL"],
                "TEAM": df.loc[i, "TEAM"],
                "POS": df.loc[i, "POS"],
                "FD_GAME": df.loc[i, "FD_GAME"],
                "SALARY": int(df.loc[i, "SALARY"]),
                "PROJECTION": float(df.loc[i, "PROJECTION"]),
            }
        )

    out = pd.DataFrame(rows)
    out["VALUE"] = out["PROJECTION"] / (out["SALARY"] / 1000.0)

    total_salary = int(out["SALARY"].sum())
    total_proj = float(out["PROJECTION"].sum())
    total_value = (total_proj / (total_salary / 1000.0)) if total_salary > 0 else 0.0

    out.loc["TOTAL"] = {
        "Slot": "",
        "PLAYER_FULL": "TOTAL",
        "TEAM": "",
        "POS": "",
        "FD_GAME": "",
        "SALARY": total_salary,
        "PROJECTION": total_proj,
        "VALUE": total_value,
    }

    return out


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="FD NBA Cash Optimal Lineup (max projected)")
    parser.add_argument("--csv", default="clean.csv", help="Path to clean.csv (default: clean.csv)")
    parser.add_argument("--cap", type=int, default=FD_SALARY_CAP, help="Salary cap (default: 60000)")
    parser.add_argument("--max-per-team", type=int, default=FD_MAX_PER_TEAM, help="Max players per team (default: 4)")
    parser.add_argument(
        "--lock",
        action="append",
        default=[],
        help='Lock a player (repeatable). Example: --lock "Nikola Jokic"',
    )
    args = parser.parse_args()

    lineup = cash_optimal_lineup(
        clean_csv=args.csv,
        locks=args.lock,
        cap=args.cap,
        max_per_team=args.max_per_team,
    )

    # print without index (TOTAL row still included)
    print(lineup.to_string(index=False))


if __name__ == "__main__":
    main()
