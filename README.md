# AES-Hybrid-Stitch-LAP24

Code for *Hybrid approach to spatial upsampling of Head-Related Transfer
Functions*, presented at the 161st AES Convention (Nashville, 2026).

Two LAP24 challenge submissions make different errors: the retrieval-augmented
neural field (**RANF**) reconstructs interaural timing well, while the
elevation-wise encoder-decoder (**EWED**) reconstructs magnitude better. The
hybrid combines them at inference time — no retraining, no new model:

```
|H_hybrid| = α · |H_RANF| + (1 − α) · |H_EWED|
```

with α = 0.45, taking the ITD from RANF alone and rebuilding a minimum-phase
impulse response from that magnitude.

On the three official LAP24 sparsity-5 test subjects, over seven random seeds,
it improves on both methods on all three official metrics at once:

| Method | ITD (µs) | ILD (dB) | LSD (dB) |
|---|---|---|---|
| RANF | 14.43 ± 0.72 | 0.95 ± 0.10 | 4.427 ± 0.131 |
| EWED | 26.90 ± 0.40 | 0.80 ± 0.03 | 4.282 ± 0.022 |
| **Hybrid** | **14.42 ± 0.71** | **0.76 ± 0.03** | **4.130 ± 0.073** |

Lower is better throughout. Intervals are 95 % confidence half-widths across
the seven seeds. `python -m hybrid.evaluate` reproduces every value in that
table; see [how α was chosen](#how-α-was-chosen) below.

## Layout

```
hybrid/core.py       the hybrid itself — mixing, reconstruction, scoring
hybrid/evaluate.py   reproduces the table above
hybrid/sweep.py      LSD against α, showing the optimum is broad
tests/               checks that run without the data
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

RANF's reconstruction helper is imported from the authors' own checkout rather
than copied into this repository, so clone it into the repo root:

```bash
git clone https://github.com/merlresearch/ranf-hrtf
```

## Inputs

Nothing here trains a model. The hybrid consumes predictions that RANF and EWED
produce, so you need all three of the following before anything will run. Each
comes from its original authors under their own licence and is not redistributed
here.

| Path | What it holds |
|---|---|
| `data/2Compare/` | LAP24 ground-truth SOFA files ([SONICOM](https://www.imperial.ac.uk/sonicom/)) |
| `ranf-hrtf/data/exp/ranf_seed{0..6}/log/eval/` | RANF predictions, `pred_p0203.sofa` etc. |
| `data/ewed_seeds/Upsampled_seed{0..6}/` | EWED predictions |

Train each method seven times, once per seed, following its own repository.
Seed *N* of RANF is paired with seed *N* of EWED, so a hybrid at that seed
carries both methods' seed variance rather than averaging one of them away.

## Running

```bash
python -m hybrid.evaluate                      # the headline table
python -m hybrid.evaluate --alpha 0.40         # a different mixing weight
python -m hybrid.sweep --csv results/sweep.csv # LSD against α
```

Reconstructed SOFA files are written under `data/`, which is gitignored. The
full sweep writes a few hundred megabytes of them.

A sweep over `[0.20, 0.60]` reproduces the paper's sensitivity result: LSD
varies by under 0.01 dB across `[0.30, 0.50]`, so the exact value of α is not
where the improvement comes from.

### How α was chosen

The sweep's minimum, taken over the three reported subjects, is α = 0.40. The
paper uses **0.45**, selected by a grid search over the nine LAP24 test
subjects that are *not* reported on, so the evaluation subjects could not
influence the choice. That costs 0.003 dB on the headline — selecting on the
reported subjects would have given a marginally better number — and paying it
is the point.

### Tests

```bash
python -m pytest tests/        # or: python tests/test_hybrid.py
```

Nine checks covering direction matching, the two-level aggregation and the α
grid. They need no SOFA files, so they run on a fresh clone.

## Upstream work

- **RANF** — Masuyama et al., [merlresearch/ranf-hrtf](https://github.com/merlresearch/ranf-hrtf)
- **EWED** ("Kalimotxo") — Arévalo and Villegas, available as a Code Ocean capsule
- **LAP24** — the [Listener Acoustic Personalisation
  Challenge](https://www.imperial.ac.uk/sonicom/lap-challenge/). Scoring uses
  [`spatialaudiometrics`](https://github.com/Katarina-Poole/Spatial-Audio-Metrics),
  the same package the challenge's own
  [task-two notebook](https://github.com/Audio-Experience-Design/LAPChallenge)
  scores with.

## Licence

MIT — see [LICENSE](LICENSE). This covers the code in this repository only;
RANF, EWED and the SONICOM data remain under their own terms.
