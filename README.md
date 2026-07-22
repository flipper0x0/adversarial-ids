# Adversarial Robustness of ML-Based Network Intrusion Detection

Code for the project *"Evaluating and Improving the Adversarial Robustness of
ML-Based Network Intrusion Detection Under Realistic Attacker Constraints."*

This repository trains baseline IDS models, attacks them with constrained
adversarial evasion, defends them, and reports an honest before/after
comparison — across two datasets, with a monitoring dashboard.

---

## Read this first (scope, honestly)

This is a **correct, runnable, tunable skeleton** — not a tuned, "maximum
accuracy" system. Accuracy comes from the real datasets plus hyperparameter
tuning on your own hardware, neither of which can be done in a sandbox. What
this code gives you:

- A clean, modular implementation of all ten proposal stages.
- The realistic-attacker **constraint model**, which is the methodological
  core of the project.
- Sensible, defensible defaults you will need to **justify and tune**, not
  accept blindly.

Three things you must own, because your supervisor and examiners will push on
them:

1. **The mutable/immutable feature map (`src/constraints.py`) is a defensible
   default, not ground truth.** Which traffic features an attacker can actually
   control is itself a research judgement. Defend your choices.
2. **Numbers in the report must come from real runs on CICIDS2017 / UNSW-NB15**,
   not from the synthetic demo, which only proves the pipeline works.
3. **Constrained attacks may produce weak adversarial examples.** If the attack
   success rate is low under constraints, that is a valid empirical finding —
   constraints make evasion harder — not a failure. Report it as such.

---

## What was verified, and what was not

- Verified by running the actual code: the metric formulas (ASR, Robustness
  Score, Defense Recovery Rate) and the constraint projection (freeze /
  in-range / increase-only). The full non-torch dependency stack (numpy,
  scipy, pandas, scikit-learn, xgboost, imbalanced-learn) installs and imports.
- **Not executed in the build sandbox:** the torch-backed white-box path
  (MLP + ART FGSM/PGD/JSMA). The CPU-only PyTorch wheel lives on an index the
  sandbox could not reach, and the CUDA wheel exceeds its disk quota. The ART
  calls in `src/attacks.py` follow standard ART 1.18 signatures and were
  reviewed, but you should run `python tests/smoke_test.py` on your own machine
  as the first thing you do (see below) to confirm the whole stack end-to-end.

---

## Setup

Requires **Python 3.10–3.12**.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins `numpy<2.0` and `adversarial-robustness-toolbox<1.19`
on purpose — newer numpy breaks this ART release. Keep the pins.

PyTorch: `pip install -r requirements.txt` installs a CPU build, which is fine.
If you have an NVIDIA GPU and want speed for adversarial training, install the
matching CUDA build of torch from pytorch.org instead.

### Confirm the install works (do this before anything else)

```bash
python tests/smoke_test.py
```

This runs the entire pipeline on tiny synthetic data in under a minute and
checks the metrics and constraints. If it passes, your environment is good. It
is **not** an experiment — it proves plumbing, nothing more.

---

## Get the datasets

The datasets are not included (too large — CICIDS2017 is ~1.1GB, UNSW-NB15
~380MB as CSVs) and must be downloaded. `data/raw/` and `results/` are both
git-ignored: GitHub hard-rejects files over 100MB, and the CICFlowMeter CSVs
blow well past that, so never commit them, even by accident with `git add -A`.

- **CICIDS2017 (primary).** Use the **corrected** release (Engelen et al. 2021),
  not the original — the original has labelling errors, infinity values, and
  duplicates. Put the CSV files in `data/raw/cicids2017/`.
- **UNSW-NB15 (secondary, cross-validation).** Put its CSV files in
  `data/raw/unsw_nb15/`.

See `data/README.md` for details. Then point `config.yaml` at them and set
`dataset: cicids2017`.

---

## File structure

```
adversarial-ids/
├── config.yaml              # ALL tunable settings live here (seeds, models, attacks, defenses)
├── requirements.txt
├── README.md
│
├── data/
│   └── README.md            # how/where to download the datasets
│
├── src/                     # the library — importable, no side effects on import
│   ├── config.py            # loads config.yaml; resolves result/processed paths
│   ├── utils.py             # seeding (reproducibility), logging, JSON IO
│   ├── constraints.py       # *** the realistic-attacker constraint model (core) ***
│   ├── dependencies.py      # *** base/derived feature model — naive-vs-consistent contrast (core) ***
│   ├── data_loaders.py      # CICIDS2017 / UNSW-NB15 / synthetic loaders
│   ├── preprocess.py        # clean, encode, MinMax-scale, (optional SMOTE), split
│   ├── models_classical.py  # Random Forest, XGBoost, SVM
│   ├── models_neural.py     # MLP + ART PyTorchClassifier wrapper (white-box target/surrogate)
│   ├── attacks.py           # constrained FGSM/PGD/JSMA (white-box) + HopSkipJump/ZOO/transfer (black-box)
│   ├── defenses.py          # adversarial training + input sanitization
│   ├── metrics.py           # clean metrics, ASR (3.1), Robustness Score (3.2), DRR (3.3)
│   └── pipeline.py          # chains the stages; writes results/<dataset>/summary.json
│
├── scripts/                 # thin command-line wrappers — run these
│   ├── _bootstrap.py        # puts repo root on sys.path
│   ├── 01_preprocess.py
│   ├── 02_train_baseline.py
│   ├── 03_evaluate_clean.py
│   ├── 04_generate_attacks.py
│   ├── 05_apply_defenses.py
│   ├── 06_reevaluate.py
│   ├── 07_cross_dataset.py  # repeats the full pipeline on the second dataset
│   └── run_all.py           # one command: everything, both datasets
│
├── dashboard/
│   └── app.py               # Streamlit dashboard (offline replay; reads summary.json)
│
├── tests/
│   └── smoke_test.py        # fast synthetic end-to-end check
│
└── results/                 # created at runtime: models, processed arrays, summary.json
```

---

## Execution sequence (how to run it)

All commands are run from the repository root, with the venv active.

**Fast path — run the whole study (both datasets):**

```bash
python scripts/run_all.py
streamlit run dashboard/app.py
```

**Step-by-step path — run one stage at a time** (recommended while developing;
each stage saves its output, so you can stop and inspect between steps):

```bash
python scripts/01_preprocess.py        # Stage 1-2: load, clean, scale, split
python scripts/02_train_baseline.py    # Stage 4:   train RF, XGBoost, SVM, MLP
python scripts/03_evaluate_clean.py    # Stage 5:   clean metrics (the "true" accuracy)
python scripts/04_generate_attacks.py  # Stage 6:   constrained attacks (naive + consistent) -> BEFORE numbers
python scripts/05_apply_defenses.py    # Stage 7:   adversarial training + sanitization
python scripts/06_reevaluate.py        # Stage 8:   re-attack hardened models; naive-vs-consistent sanitization contrast -> AFTER numbers
python scripts/07_cross_dataset.py     # Stage 9:   repeat the full pipeline on UNSW-NB15
streamlit run dashboard/app.py         # Stage 10:  visualize detections / attack impact / defense effect
```

Which dataset each script uses is controlled by `dataset:` (and
`cross_dataset:`) in `config.yaml`. To run the real experiment, set
`dataset: cicids2017` and re-run from step 01.

Every run writes/updates `results/<dataset>/summary.json`. The dashboard reads
that file — run the pipeline before launching the dashboard.

---

## Implementation / build sequence (the order it was built, and how to extend it)

Build and extend bottom-up. Each layer has a **verification gate** — do not move
on until it passes, or you will debug ten things at once.

1. **Config + utils** (`config.py`, `utils.py`). Fix the seed everywhere.
   *Gate:* config loads; `set_seed` makes two runs identical.
2. **Constraint model** (`constraints.py`). This is the contribution; build it
   early. *Gate:* projection freezes immutable features, clips to range, and
   enforces increase-only (the smoke test's constraint check).
3. **Data + preprocessing** (`data_loaders.py`, `preprocess.py`). MinMax-scale to
   `[0,1]` so attack budgets are interpretable; class weighting by default.
   *Gate:* processed arrays in `[0,1]`, label split sane, no leaked columns.
4. **Models** (`models_classical.py`, `models_neural.py`). The MLP exists to be a
   differentiable white-box target/surrogate; tree models have no usable
   gradient. *Gate:* all four train and score on clean data.
5. **Metrics** (`metrics.py`). *Gate:* ASR/R/DRR match hand-worked examples
   (the smoke test's metric check).
6. **Attacks** (`attacks.py`, `dependencies.py`). White-box on the MLP with an
   ART feature mask during search; black-box (HopSkipJump/ZOO) and transfer on
   the tree models. Each white-box call runs twice per Stage 6 — once
   *naive* (full mutable set, `attacks_before`) and once *consistency-
   preserving* (base features only, derived features recomputed via
   `dependencies.build_consistent_constraint_spec`, `attacks_before_consistent`).
   *Gate:* generated samples are constraint-valid after projection; the
   consistent spec perturbs strictly fewer features than the naive one
   (smoke test's dependency check).
7. **Defenses** (`defenses.py`). Adversarial training + sanitization, where
   sanitization now also recomputes derived features from base features. This
   is the naive-vs-consistent contrast (proposal's core finding): the same
   `sanitize()` call defeats the naive attack samples but not the consistent
   ones (`sanitization_MLP_naive` vs `sanitization_MLP_consistent` in
   `summary.json`; dashboard section "3.5"). *Gate:* hardened model still
   classifies clean traffic; AFTER >= BEFORE under attack.
8. **Pipeline** (`pipeline.py`) ties stages 3–7 into one run and writes
   `summary.json`. *Gate:* `tests/smoke_test.py` passes end-to-end.
9. **Cross-dataset** is just the pipeline re-run on UNSW-NB15 (`07_*`). The two
   feature spaces do not overlap, so this **retrains from scratch** — it does
   not transfer attacks across datasets. Scope your generalization claims to
   that fact.
10. **Dashboard** (`dashboard/app.py`) reads `summary.json` and replays recorded
    test traffic offline. It is the last thing to build because it only
    visualizes results that already exist.

---

## Key design decisions (and their trade-offs)

- **MinMax scaling, not standardization.** Keeps every feature in `[0,1]` so an
  L-infinity epsilon is a fixed fraction of each feature's range — directly
  interpretable. Standardization would make epsilon mean different things per
  feature.
- **Class weighting on by default; SMOTE off.** SMOTE interpolates new "flows"
  that may be physically invalid and, done wrong, leaks across the split. It is
  available (`preprocess.use_smote: true`) but train-split only — use with care.
- **Constraint map goes beyond the proposal:** timing and length features are
  **increase-only** (an attacker can add delay or padding, not make a flow
  shorter than what the attack already sends). This is more realistic than free
  perturbation. Be ready to defend it.
- **Black-box constraint handling is post-hoc projection + honest
  re-measurement**, not constraint-aware search. Projecting a black-box result
  can destroy its adversarial property — which correctly shows constraints make
  the attacker's job harder. This is a stated limitation, not a bug.
- **Kernel SVM is subsampled** (`models.svm.train_subsample`) because an RBF SVM
  cannot train on millions of rows. RF and XGBoost use the full data. Report the
  subsample size.
- **Dependent-feature recomputation (`src/dependencies.py`) is a defensible
  simplification, not a byte-for-byte CICFlowMeter/Argus reimplementation.**
  A handful of statistics (per-packet means, header length, load, inter-packet
  time) are recomputed from base totals/counts/duration; e.g.
  `Fwd Packet Length Mean = Total Length of Fwd Packets / Total Fwd Packets`.
  This is what the naive-vs-consistent contrast in `pipeline.stage_attacks` /
  `stage_reeval` and the dashboard's "3.5" section actually run: a naive
  attack perturbs a derived feature directly (inconsistent with its base),
  a consistent attack only perturbs base features and gets derived ones via
  `DependencySpec.recompute`. Defend the specific formulas, not the mechanism.

---

## Troubleshooting

- **ART import error / numpy errors.** You have numpy >= 2.0. Reinstall with the
  pins: `pip install "numpy<2.0"`. ART < 1.19 requires it.
- **`torch` install is huge or fails.** The default wheel may pull large CUDA
  packages. For a CPU-only machine, install the CPU build of torch from
  pytorch.org.
- **SVM training never finishes.** Lower `models.svm.train_subsample` in
  `config.yaml`.
- **Black-box attacks are too slow.** Lower `attacks.blackbox.n_samples` and the
  HopSkipJump iteration budget; keep ZOO disabled except for a small final run.
- **Dashboard shows nothing.** Run the pipeline first — it reads
  `results/<dataset>/summary.json`.
