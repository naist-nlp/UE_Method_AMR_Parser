#!/usr/bin/env python3
"""
fix_gold_alignment.py

Re-align gold_graph in sent_*.json files for baseline models.
The stored gold_graph values were misaligned (or empty); this script
replaces them with the correct AMR from gold.amr.txt keyed by sent_id.

Usage:
    python3 fix_gold_alignment.py [--dry_run]
"""

import json
import argparse
from pathlib import Path

_ROOT     = Path(__file__).resolve().parent.parent   # repository root (script lives in kbest_generation/)
GOLD_FILE = _ROOT / "data" / "k_top_AMR_parsers" / "gold.amr.txt"

BASELINE_MODELS = [
    "SPRING",
    "IBM_parser",
    "HMG-AMR-T5",
    "HMG-AMR-BART",
]
BEAM_SIZES = [3, 5, 8, 10]


def parse_gold_file(path: Path) -> dict:
    """Parse gold.amr.txt → {amr_id: graph_string}."""
    gold_map: dict = {}
    current_id    = None
    current_lines = []

    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("# ::id "):
            # Save previous entry
            if current_id and current_lines:
                gold_map[current_id] = "\n".join(current_lines).strip()
            # Start new entry
            rest = stripped[len("# ::id "):]
            current_id    = rest.split("::")[0].strip()
            current_lines = []
        elif stripped.startswith("#") or not stripped:
            continue
        else:
            current_lines.append(line)

    if current_id and current_lines:
        gold_map[current_id] = "\n".join(current_lines).strip()

    return gold_map


def _get_sent_dir(model: str, beam: int) -> Path:
    base = _ROOT / "data" / "k_top_AMR_parsers" / model / f"kbest_beam{beam}"
    for candidate in [
        base / "sentences",
        base / f"kbest_beam{beam}" / "sentences",
    ]:
        if candidate.exists():
            return candidate
    return None


def fix_model_beam(model: str, beam: int, gold_map: dict, dry_run: bool) -> dict:
    sent_dir = _get_sent_dir(model, beam)
    if sent_dir is None:
        return {"skipped": True}

    files = sorted(sent_dir.glob("sent_*.json"))
    stats = {"total": len(files), "fixed": 0, "missing_id": 0, "already_ok": 0, "no_gold_entry": 0}

    for f in files:
        d = json.loads(f.read_text())
        raw_sent_id = d.get("sent_id", "")
        amr_id = raw_sent_id.split("::")[0].strip() if "::" in raw_sent_id else raw_sent_id.strip()

        if not amr_id:
            stats["missing_id"] += 1
            continue

        correct_gold = gold_map.get(amr_id)
        if correct_gold is None:
            stats["no_gold_entry"] += 1
            continue

        current_gold = d.get("gold_graph", "").strip()
        if current_gold == correct_gold:
            stats["already_ok"] += 1
            continue

        d["gold_graph"] = correct_gold
        if not dry_run:
            f.write_text(json.dumps(d, ensure_ascii=False, indent=2))
        stats["fixed"] += 1

    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true", help="Report changes without writing")
    ap.add_argument("--models", nargs="+", default=BASELINE_MODELS)
    ap.add_argument("--beams",  nargs="+", type=int, default=BEAM_SIZES)
    args = ap.parse_args()

    print("Parsing gold.amr.txt ...")
    gold_map = parse_gold_file(GOLD_FILE)
    print(f"  {len(gold_map)} gold entries loaded")

    if args.dry_run:
        print("DRY RUN — no files will be written\n")

    for model in args.models:
        for beam in args.beams:
            stats = fix_model_beam(model, beam, gold_map, args.dry_run)
            if stats.get("skipped"):
                print(f"{model}/beam{beam}: SKIP (sentences dir not found)")
            else:
                action = "would fix" if args.dry_run else "fixed"
                print(
                    f"{model}/beam{beam}: total={stats['total']}  "
                    f"{action}={stats['fixed']}  "
                    f"already_ok={stats['already_ok']}  "
                    f"no_gold_entry={stats['no_gold_entry']}  "
                    f"missing_id={stats['missing_id']}"
                )

    print("\nDone.")


if __name__ == "__main__":
    main()
