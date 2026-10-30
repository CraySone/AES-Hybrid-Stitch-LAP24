"""Sweep the mixing weight alpha and report LSD at each value.

Shows that the optimum is broad: across a wide band around alpha = 0.45, LSD
moves by only thousandths of a dB, so the result does not rest on the exact
value.

Note that the minimum of this sweep is not how the paper's alpha was chosen.
Run over the three reported subjects, the curve bottoms out at alpha = 0.40;
the paper uses 0.45, selected by a grid search over the nine LAP24 test
subjects that are NOT reported on, so the evaluation subjects could not
influence it. Selecting on the reported subjects instead would give a headline
0.003 dB better -- which is precisely why it was not done that way.

    python -m hybrid.sweep
    python -m hybrid.sweep --alphas 0.0 1.0 --step 0.05 --csv results/sweep.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .core import ALPHA, REPO_ROOT, SUBJECTS, Prediction, aggregate

#: Rows within this many dB of the best are reported as the flat region.
FLAT_TOLERANCE_DB = 0.01


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alphas", type=float, nargs=2, default=(0.20, 0.60),
                        metavar=("LOW", "HIGH"), help="inclusive range (default 0.20 0.60)")
    parser.add_argument("--step", type=float, default=0.05, help="grid spacing (default 0.05)")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(7)))
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS))
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "sweep")
    parser.add_argument("--csv", type=Path, help="also write the table here")
    args = parser.parse_args()

    low, high = args.alphas
    steps = int(round((high - low) / args.step)) + 1
    alphas = [round(low + i * args.step, 4) for i in range(steps)]

    # Load each pair once; sweeping alpha then only re-mixes and re-scores.
    predictions = [Prediction.load(seed, subject)
                   for seed in args.seeds for subject in args.subjects]

    rows = []
    for alpha in alphas:
        per_seed: dict[int, list] = {}
        for prediction in predictions:
            per_seed.setdefault(prediction.seed, []).append(
                prediction.score_hybrid(alpha, args.out_dir)
            )
        summary = aggregate(per_seed)
        rows.append((alpha, summary["lsd_db_mean"], summary["lsd_db_ci95"]))
        print(f"alpha {alpha:.2f}   LSD {summary['lsd_db_mean']:.4f} "
              f"+/- {summary['lsd_db_ci95']:.4f}")

    best_alpha, best_lsd, _ = min(rows, key=lambda row: row[1])
    subjects = ", ".join(args.subjects)
    print(f"\nminimum of this sweep: alpha {best_alpha:.2f}, LSD {best_lsd:.4f} "
          f"(on {subjects})")

    flat = [(alpha, lsd) for alpha, lsd, _ in rows
            if lsd - best_lsd <= FLAT_TOLERANCE_DB]
    if len(flat) > 1:
        spread = max(lsd for _, lsd in flat) - min(lsd for _, lsd in flat)
        print(f"alpha {min(a for a, _ in flat):.2f} to {max(a for a, _ in flat):.2f} "
              f"all lie within {spread:.4f} dB")

    if set(args.subjects) == set(SUBJECTS):
        at_paper = next((lsd for alpha, lsd, _ in rows if abs(alpha - ALPHA) < 1e-9), None)
        if at_paper is not None:
            print(f"\nthe paper reports alpha {ALPHA} (LSD {at_paper:.4f}), chosen on the "
                  f"nine\nsubjects not evaluated here -- see this module's docstring.")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["alpha", "lsd_db_mean", "lsd_db_ci95"])
            for alpha, mean, ci95 in rows:
                writer.writerow([f"{alpha:.2f}", f"{mean:.4f}", f"{ci95:.4f}"])
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
