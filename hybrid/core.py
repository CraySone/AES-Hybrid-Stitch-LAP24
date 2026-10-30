"""Core of the RANF + EWED hybrid.

The hybrid takes the magnitude spectrum as a convex combination of the two
methods' predictions and the interaural time difference from RANF alone:

    |H_hybrid| = alpha * |H_RANF| + (1 - alpha) * |H_EWED|

A minimum-phase impulse response is then reconstructed from that magnitude and
RANF's ITD. Nothing is retrained: both methods' predictions are inputs.
"""

from __future__ import annotations

import contextlib
import copy
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sofar as sf
from spatialaudiometrics import hrtf_metrics as hm
from spatialaudiometrics import lap_challenge as lap

REPO_ROOT = Path(__file__).resolve().parents[1]

SR = 48000

#: Mixing weight used in the paper (RANF's share of the magnitude).
ALPHA = 0.45

#: The three official LAP24 sparsity-5 test subjects.
SUBJECTS = ("P0203", "P0207", "P0212")

TRUTH_DIR = REPO_ROOT / "data" / "2Compare"
RANF_DIR = REPO_ROOT / "ranf-hrtf" / "data" / "exp"
EWED_DIR = REPO_ROOT / "data" / "ewed_seeds"


def _minimum_phase():
    """Return RANF's reconstruction helper, importing it on first use.

    RANF is cloned beside this repository rather than vendored, so it stays the
    authors' code under their own licence. Importing lazily keeps `--help` and
    the test suite working before that clone exists, and turns a missing
    checkout into an explanatory message instead of an ImportError traceback.
    """
    ranf_root = REPO_ROOT / "ranf-hrtf"
    if str(ranf_root) not in sys.path:
        sys.path.insert(0, str(ranf_root))
    try:
        from ranf.utils.reconstruction import hrtf2hrir_minph
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"RANF not found at {ranf_root}. Clone it first:\n"
            "    git clone https://github.com/merlresearch/ranf-hrtf"
        ) from exc
    return hrtf2hrir_minph


def truth_path(subject: str) -> Path:
    return TRUTH_DIR / f"{subject}_FreeFieldCompMinPhase_48kHz.sofa"


def ranf_path(seed: int, subject: str) -> Path:
    return RANF_DIR / f"ranf_seed{seed}" / "log" / "eval" / f"pred_{subject.lower()}.sofa"


def ewed_path(seed: int, subject: str) -> Path:
    return EWED_DIR / f"Upsampled_seed{seed}" / f"{subject}_FreeFieldCompMinPhase_48kHz.sofa"


def match_positions(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Index `source` so its rows line up with `target`'s directions.

    The three SOFA files store the same directions in different orders, so the
    magnitudes cannot simply be added elementwise. Azimuth difference wraps at
    +/-180 degrees; nearest neighbour is exact here because the grids are
    identical up to ordering.
    """
    out = np.zeros(len(target), dtype=np.int64)
    for i, (az, el, _) in enumerate(target):
        d_az = (source[:, 0] - az + 180) % 360 - 180
        d_el = source[:, 1] - el
        out[i] = int(np.argmin(d_az * d_az + d_el * d_el))
    return out


@dataclass(frozen=True)
class Metrics:
    """LAP24 task-two errors. Lower is better on all three."""

    itd_us: float
    ild_db: float
    lsd_db: float

    @classmethod
    def evaluate(cls, truth: Path, prediction: Path) -> "Metrics":
        # This is the scoring used by the LAP24 challenge repository's own
        # task-two notebook, so the numbers are comparable with the
        # submissions'. It prints a per-file report to stdout.
        with contextlib.redirect_stdout(io.StringIO()):
            values, _, _ = lap.calculate_task_two_metrics(str(truth), str(prediction))
        return cls(float(values[0]), float(values[1]), float(values[2]))


@dataclass(frozen=True)
class Prediction:
    """One seed-matched (RANF-N, EWED-N) pair for one subject.

    Magnitudes and ITD are precomputed, so sweeping alpha costs one weighted
    sum and one reconstruction per value rather than re-reading the SOFAs.
    """

    subject: str
    seed: int
    truth_sofa: object
    ranf_mag: np.ndarray
    ewed_mag: np.ndarray
    ranf_itd: np.ndarray
    nfft: int

    @classmethod
    def load(cls, seed: int, subject: str) -> "Prediction":
        truth = sf.read_sofa(str(truth_path(subject)), verbose=False)
        ranf = sf.read_sofa(str(ranf_path(seed, subject)), verbose=False)
        ewed = sf.read_sofa(str(ewed_path(seed, subject)), verbose=False)

        directions = np.asarray(truth.SourcePosition)
        ranf_ir = np.asarray(ranf.Data_IR)[
            match_positions(np.asarray(ranf.SourcePosition), directions)
        ]
        ewed_ir = np.asarray(ewed.Data_IR)[
            match_positions(np.asarray(ewed.SourcePosition), directions)
        ]

        # Truncate to the shortest of the three so the FFTs share a length.
        nfft = min(
            np.asarray(truth.Data_IR).shape[-1], ranf_ir.shape[-1], ewed_ir.shape[-1]
        )
        ranf_ir = ranf_ir[..., :nfft]
        ewed_ir = ewed_ir[..., :nfft]

        _, itd_samples, _ = hm.itd_estimator_maxiacce(ranf_ir, np.array(SR))

        return cls(
            subject=subject,
            seed=seed,
            truth_sofa=truth,
            ranf_mag=np.abs(np.fft.rfft(ranf_ir, axis=-1)),
            ewed_mag=np.abs(np.fft.rfft(ewed_ir, axis=-1)),
            ranf_itd=np.asarray(itd_samples).reshape(-1, 1),
            nfft=nfft,
        )

    def mix(self, alpha: float = ALPHA) -> np.ndarray:
        """Reconstruct the hybrid impulse response at this alpha."""
        magnitude = alpha * self.ranf_mag + (1.0 - alpha) * self.ewed_mag
        return _minimum_phase()(magnitude, self.ranf_itd, nfft=self.nfft, fs=SR)

    def write(self, impulse_response: np.ndarray, path: Path) -> Path:
        """Save an impulse response as SOFA, reusing the truth file's metadata."""
        sofa = copy.deepcopy(self.truth_sofa)
        sofa.Data_IR = impulse_response.astype(np.float64)
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write_sofa(str(path), sofa)
        return path

    def score_hybrid(self, alpha: float, out_dir: Path) -> Metrics:
        """Build the hybrid at `alpha` and score it against the ground truth."""
        path = self.write(
            self.mix(alpha), out_dir / f"{self.subject}_seed{self.seed}_alpha{alpha:.2f}.sofa"
        )
        return Metrics.evaluate(truth_path(self.subject), path)

    def score_ranf(self) -> Metrics:
        return Metrics.evaluate(truth_path(self.subject), ranf_path(self.seed, self.subject))

    def score_ewed(self) -> Metrics:
        return Metrics.evaluate(truth_path(self.subject), ewed_path(self.seed, self.subject))


def aggregate(per_subject: dict[int, list[Metrics]]) -> dict[str, float]:
    """Two-level aggregation: mean over subjects per seed, then over seeds.

    Averaging in one pass over all (seed, subject) observations would treat
    correlated measurements as independent and understate the spread. The
    reported confidence interval is 1.96 * SD / sqrt(n_seeds).
    """
    seed_means = {
        "itd_us": [], "ild_db": [], "lsd_db": [],
    }
    for _, rows in sorted(per_subject.items()):
        seed_means["itd_us"].append(np.mean([r.itd_us for r in rows]))
        seed_means["ild_db"].append(np.mean([r.ild_db for r in rows]))
        seed_means["lsd_db"].append(np.mean([r.lsd_db for r in rows]))

    out: dict[str, float] = {}
    n = len(seed_means["lsd_db"])
    for key, values in seed_means.items():
        out[f"{key}_mean"] = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
        out[f"{key}_sd"] = sd
        out[f"{key}_ci95"] = 1.96 * sd / np.sqrt(n) if n > 1 else 0.0
    return out
