"""Reproduce the paper's headline comparison.

Scores RANF, EWED and the hybrid over 7 seed-matched pairs on the three
official LAP24 sparsity-5 test subjects.

    python -m hybrid.evaluate
    python -m hybrid.evaluate --alpha 0.40 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .core import ALPHA, REPO_ROOT, SUBJECTS, Prediction, aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--alpha", type=float, default=ALPHA,
                        help=f"RANF's share of the magnitude (default {ALPHA})")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(7)),
                        help="seeds to evaluate (default 0-6)")
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS),
                        help="subjects to evaluate")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "hybrid",
                        help="where the reconstructed SOFA files are written")
    args = parser.parse_args()

    scores: dict[str, dict[int, list]] = {"RANF": {}, "EWED": {}, "hybrid": {}}

    for seed in args.seeds:
        for subject in args.subjects:
            prediction = Prediction.load(seed, subject)
            results = {
                "RANF": prediction.score_ranf(),
                "EWED": prediction.score_ewed(),
                "hybrid": prediction.score_hybrid(args.alpha, args.out_dir),
            }
            for method, metrics in results.items():
                scores[method].setdefault(seed, []).append(metrics)
            print(f"seed {seed}  {subject}  "
                  + "  ".join(f"{m}: LSD {r.lsd_db:.3f}" for m, r in results.items()))

    print()
    print(f"{len(args.seeds)} seeds x {len(args.subjects)} subjects, alpha = {args.alpha}")
    print(f"{'method':<10}{'ITD (us)':>18}{'ILD (dB)':>18}{'LSD (dB)':>18}")
    for method, per_seed in scores.items():
        a = aggregate(per_seed)
        print(f"{method:<10}"
              f"{a['itd_us_mean']:>10.2f} +/-{a['itd_us_ci95']:<6.2f}"
              f"{a['ild_db_mean']:>10.3f} +/-{a['ild_db_ci95']:<6.3f}"
              f"{a['lsd_db_mean']:>10.3f} +/-{a['lsd_db_ci95']:<6.3f}")
    print("\nLower is better on all three. Intervals are 95% CIs across seeds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
