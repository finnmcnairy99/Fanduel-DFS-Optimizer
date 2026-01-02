import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional


OUT_DIR = Path("data") / "contests"


def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "contest"


@dataclass
class PayoutTier:
    rank_min: int
    rank_max: int
    payout: float

    def to_dict(self) -> Dict[str, Any]:
        return {"rank_min": self.rank_min, "rank_max": self.rank_max, "payout": self.payout}


def parse_rank_range(rank_str: str) -> tuple[int, int]:
    rank_str = rank_str.strip()
    if "-" in rank_str:
        a, b = rank_str.split("-", 1)
        a = int(a.strip())
        b = int(b.strip())
    else:
        a = b = int(rank_str)
    if a <= 0 or b <= 0:
        raise ValueError(f"Ranks must be positive: {rank_str}")
    if b < a:
        raise ValueError(f"Invalid rank range (max < min): {rank_str}")
    return a, b


def parse_payout_string(payout_str: str) -> List[PayoutTier]:
    """
    Example:
      "1:125,2:75,9-10:4,26-30:1.5"
    """
    if not payout_str or not payout_str.strip():
        raise ValueError("Empty --payout string")

    tiers: List[PayoutTier] = []
    chunks = [c.strip() for c in payout_str.split(",") if c.strip()]
    for chunk in chunks:
        if ":" not in chunk:
            raise ValueError(f"Bad payout chunk (missing ':'): {chunk}")
        rank_part, pay_part = chunk.split(":", 1)
        rmin, rmax = parse_rank_range(rank_part)
        payout = float(pay_part.strip())
        tiers.append(PayoutTier(rmin, rmax, payout))

    # sort by rank_min then rank_max
    tiers.sort(key=lambda t: (t.rank_min, t.rank_max))
    return tiers


def load_payout_file(path: Path) -> List[PayoutTier]:
    """
    Accepts a CSV with columns:
      rank_min,rank_max,payout
    """
    import csv

    if not path.exists():
        raise FileNotFoundError(f"Payout file not found: {path}")

    tiers: List[PayoutTier] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        needed = {"rank_min", "rank_max", "payout"}
        if not needed.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Payout CSV must have columns {sorted(needed)}; got {reader.fieldnames}")

        for row in reader:
            rmin = int(str(row["rank_min"]).strip())
            rmax = int(str(row["rank_max"]).strip())
            payout = float(str(row["payout"]).strip())
            tiers.append(PayoutTier(rmin, rmax, payout))

    tiers.sort(key=lambda t: (t.rank_min, t.rank_max))
    return tiers


def validate_payouts(tiers: List[PayoutTier], field_size: int) -> None:
    if not tiers:
        raise ValueError("No payout tiers provided")

    # sanity checks
    for t in tiers:
        if t.rank_min <= 0 or t.rank_max <= 0:
            raise ValueError(f"Invalid tier ranks: {t}")
        if t.rank_max < t.rank_min:
            raise ValueError(f"Invalid tier range: {t}")
        if t.payout < 0:
            raise ValueError(f"Negative payout not allowed: {t}")

    # check tiers don’t exceed field_size by default (FD sometimes pays deeper than expected; warn instead of fail)
    max_paid = max(t.rank_max for t in tiers)
    if max_paid > field_size:
        print(f"⚠️ warning: max paid rank ({max_paid}) > field size ({field_size}). Check your inputs.")


def build_contest_config(
    name: str,
    field_size: int,
    entry_fee: float,
    max_entries: int,
    entries_you: int,
    tiers: List[PayoutTier],
) -> Dict[str, Any]:
    paid_entries = max(t.rank_max for t in tiers)
    min_cash = None
    # min cash is the lowest rank range that has a positive payout
    positive = [t for t in tiers if t.payout > 0]
    if positive:
        min_cash = min(t.rank_min for t in positive)

    return {
        "name": name,
        "field_size": field_size,
        "entry_fee": entry_fee,
        "max_entries": max_entries,
        "entries_you": entries_you,
        "paid_entries": paid_entries,
        "min_cash_rank": min_cash,
        "payouts": [t.to_dict() for t in tiers],
    }


def main():
    parser = argparse.ArgumentParser(description="Create a contest config JSON for sim.ipynb.")
    parser.add_argument("--name", required=True, help="Contest name")
    parser.add_argument("--field-size", type=int, required=True, help="Total entries in contest (field size)")
    parser.add_argument("--entry-fee", type=float, required=True, help="Entry fee (e.g., 5.55)")
    parser.add_argument("--max-entries", type=int, required=True, help="Max entries per user (e.g., 150)")
    parser.add_argument("--entries-you", type=int, required=True, help="How many lineups you will enter")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--payout", help='Payout string like "1:100000,2:10000,9-10:1000,21-25:100"')
    group.add_argument("--payout-file", help="CSV path with columns rank_min,rank_max,payout")

    parser.add_argument("--out", default=None, help="Optional output json path (defaults to data/contests/<slug>.json)")

    args = parser.parse_args()

    if args.entries_you <= 0:
        raise ValueError("--entries-you must be >= 1")
    if args.entries_you > args.max_entries:
        print("⚠️ warning: entries-you > max-entries (per user). Check your inputs.")

    if args.payout_file:
        tiers = load_payout_file(Path(args.payout_file))
    else:
        tiers = parse_payout_string(args.payout)

    validate_payouts(tiers, args.field_size)

    config = build_contest_config(
        name=args.name,
        field_size=args.field_size,
        entry_fee=args.entry_fee,
        max_entries=args.max_entries,
        entries_you=args.entries_you,
        tiers=tiers,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.out:
        out_path = Path(args.out)
    else:
        slug = slugify(f"{args.name}_{args.entry_fee}")
        out_path = OUT_DIR / f"{slug}.json"

    out_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"✅ wrote {out_path}")


if __name__ == "__main__":
    main()
