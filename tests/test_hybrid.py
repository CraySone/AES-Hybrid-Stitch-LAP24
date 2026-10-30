"""Tests that run without the SOFA data.

The full pipeline needs the SONICOM ground truth and both methods' predictions,
which are not redistributable. These cover the parts that can be checked on
their own: direction matching, the two-level aggregation, and the alpha grid.

    python -m pytest tests/
    python tests/test_hybrid.py     # also works without pytest
"""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hybrid.core import ALPHA, Metrics, aggregate, match_positions
from hybrid.sweep import FLAT_TOLERANCE_DB

# The published sweep over the three reported subjects, 7 seed-matched pairs.
PUBLISHED_SWEEP = [
    (0.20, 4.163), (0.25, 4.147), (0.30, 4.136), (0.35, 4.129),
    (0.40, 4.127), (0.45, 4.130), (0.50, 4.137), (0.55, 4.148), (0.60, 4.163),
]


def test_match_positions_realigns_a_shuffled_grid():
    rng = np.random.default_rng(0)
    grid = np.array([[az, el, 1.5]
                     for el in (-30.0, 0.0, 30.0, 60.0)
                     for az in np.arange(0, 360, 15.0)])
    shuffled = grid[rng.permutation(len(grid))]

    recovered = shuffled[match_positions(shuffled, grid)]

    assert np.allclose(recovered, grid)


def test_match_positions_wraps_azimuth():
    # 359 degrees is nearer to 0 than to 180, but must still match itself.
    candidates = np.array([[0.0, 0.0, 1.5], [359.0, 0.0, 1.5], [180.0, 0.0, 1.5]])
    target = np.array([[359.0, 0.0, 1.5]])

    assert match_positions(candidates, target)[0] == 1


def test_aggregate_reproduces_the_published_interval():
    """The paper's alpha=0.45 row: LSD 4.130, SD 0.098, 95% CI half-width 0.073."""
    spread = np.array([-1.5, -1.0, -0.4, 0.0, 0.4, 1.0, 1.5])
    seed_means = 4.130 + (spread - spread.mean()) / spread.std(ddof=1) * 0.098

    summary = aggregate({i: [Metrics(0.0, 0.0, float(v))]
                         for i, v in enumerate(seed_means)})

    assert summary["lsd_db_mean"] == pytest_approx(4.130, 1e-6)
    assert summary["lsd_db_sd"] == pytest_approx(0.098, 1e-6)
    assert summary["lsd_db_ci95"] == pytest_approx(0.073, 5e-4)


def test_aggregate_averages_subjects_before_seeds():
    """A flat mean over all observations would weight seeds unequally."""
    lopsided = {
        0: [Metrics(0, 0, 4.0), Metrics(0, 0, 4.0), Metrics(0, 0, 4.0)],
        1: [Metrics(0, 0, 5.0)],
    }

    assert aggregate(lopsided)["lsd_db_mean"] == pytest_approx(4.5, 1e-9)
    assert np.mean([4.0, 4.0, 4.0, 5.0]) == 4.25  # what a flat mean would give


def test_aggregate_handles_a_single_seed():
    """One seed has no spread to report, and must not divide by zero."""
    summary = aggregate({0: [Metrics(1.0, 2.0, 3.0)]})

    assert summary["lsd_db_mean"] == pytest_approx(3.0, 1e-9)
    assert summary["lsd_db_sd"] == 0.0
    assert summary["lsd_db_ci95"] == 0.0


def test_alpha_grid_is_inclusive_and_clean():
    def build(low, high, step):
        steps = int(round((high - low) / step)) + 1
        return [round(low + i * step, 4) for i in range(steps)]

    assert build(0.20, 0.60, 0.05) == [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    assert len(build(0.0, 1.0, 0.05)) == 21     # the selection grid
    assert len(build(0.10, 0.70, 0.02)) == 31
    assert build(0.45, 0.45, 0.05) == [0.45]    # single point, no zero division


def test_flat_region_matches_the_papers_sensitivity_claim():
    """The paper: values in [0.30, 0.50] lie within 0.010 dB of each other."""
    best = min(lsd for _, lsd in PUBLISHED_SWEEP)
    flat = [a for a, lsd in PUBLISHED_SWEEP if lsd - best <= FLAT_TOLERANCE_DB]
    spread = (max(lsd for a, lsd in PUBLISHED_SWEEP if a in flat)
              - min(lsd for a, lsd in PUBLISHED_SWEEP if a in flat))

    assert (min(flat), max(flat)) == (0.30, 0.50)
    assert spread <= 0.010


def test_the_sweep_minimum_is_not_the_papers_alpha():
    """Selecting on the reported subjects would give 0.40, not the paper's 0.45.

    The paper chose alpha on the nine subjects it does not report on, and pays
    0.003 dB for it. That gap is the evidence alpha was not tuned on the
    evaluation set, so it is worth pinning down.
    """
    minimum = min(PUBLISHED_SWEEP, key=lambda row: row[1])
    at_paper_alpha = next(lsd for a, lsd in PUBLISHED_SWEEP if a == ALPHA)

    assert minimum[0] == 0.40
    assert ALPHA == 0.45
    assert at_paper_alpha - minimum[1] == pytest_approx(0.003, 1e-9)


def test_sweep_csv_round_trips():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "nested" / "sweep.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["alpha", "lsd_db_mean", "lsd_db_ci95"])
            for alpha, lsd in PUBLISHED_SWEEP:
                writer.writerow([f"{alpha:.2f}", f"{lsd:.4f}", "0.0000"])

        rows = list(csv.DictReader(open(path, encoding="utf-8")))

    assert len(rows) == len(PUBLISHED_SWEEP)
    assert rows[4]["alpha"] == "0.40" and rows[4]["lsd_db_mean"] == "4.1270"


def pytest_approx(value, tolerance):
    """Tiny stand-in so this file also runs without pytest installed."""
    class Approx:
        def __eq__(self, other):
            return abs(other - value) <= tolerance

        def __repr__(self):
            return f"approx({value} +/- {tolerance})"

    return Approx()


if __name__ == "__main__":
    failures = 0
    for name, test in sorted(globals().items()):
        if name.startswith("test_") and callable(test):
            try:
                test()
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
            else:
                print(f"ok   {name}")
    print(f"\n{'all passed' if not failures else f'{failures} failed'}")
    raise SystemExit(1 if failures else 0)
