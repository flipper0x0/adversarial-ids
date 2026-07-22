# Training on Kaggle (`kaggle_train.py`)

This guide explains how to run `kaggle_train.py` on Kaggle's free GPU to train the
four models (RandomForest, XGBoost, SVM, MLP) on CICIDS2017, then bring the trained
artifacts back to your local repo for the attack/defense stages.

> **Why this is needed:** `kaggle_train.py` is **not** a standalone file. It imports
> the project package (`from src.preprocess import ...`, etc.) and reads `config.yaml`.
> So you must upload the **whole repo**, not just this one script. The script adds the
> repo folder to Python's import path at runtime, which is what makes the imports work
> on Kaggle. `load_config()` locates `config.yaml` relative to the `src/` folder, so as
> long as `src/` and `config.yaml` stay together, nothing is tied to your local machine.

---

## What runs where

| Stage | Where | Why |
|-------|-------|-----|
| Preprocess + train 4 models + clean metrics | **Kaggle (GPU)** | The MLP wants a GPU; the rest is fast. |
| Attacks (FGSM/PGD/JSMA/black-box) + defenses | **Local** | `kaggle_train.py` deliberately does **not** run these. |

---

## Prerequisites

- A Kaggle account with **phone-verified GPU access** (Settings → Phone Verification).
- Your fixed local repo. The leak fix in `src/data_loaders.py` (dropping `id` and
  `Attempted Category`) **must** be in the code you upload — see the warning at the end.

---

## Step 1 — Package the code (exclude data/results)

The `data/` folder is ~1.5 GB, so keep **code** and **data** as two separate Kaggle
Datasets. From the repo root in **PowerShell**:

```powershell
cd "d:\major project\adversarial-ids"
$stage = "$env:TEMP\adversarial-ids"
Remove-Item $stage -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory $stage | Out-Null
Copy-Item src,scripts,config.yaml,requirements.txt,kaggle_train.py $stage -Recurse
Remove-Item "$stage\src\__pycache__" -Recurse -Force -ErrorAction SilentlyContinue
Compress-Archive "$stage\*" "$env:TEMP\adversarial-ids-code.zip" -Force
Write-Host "Created: $env:TEMP\adversarial-ids-code.zip"
```

This produces `adversarial-ids-code.zip` with `src/`, `config.yaml`, `scripts/`,
`requirements.txt`, and `kaggle_train.py` at the top level.

## Step 2 — Create two Kaggle Datasets

On kaggle.com → **Datasets → New Dataset**:

1. **Code dataset** — upload `adversarial-ids-code.zip`. Kaggle auto-unzips it.
   It will mount at `/kaggle/input/<code-slug>/...`.
2. **Data dataset** — upload the 5 CICIDS CSVs from `data/raw/cicids2017/`
   (`friday.csv`, `monday.csv`, `thursday.csv`, `tuesday.csv`, `wednesday.csv`).
   It will mount at `/kaggle/input/<data-slug>/`.

> If you re-upload later, create a **new version** of the same dataset so the slug
> stays stable.

## Step 3 — Create the notebook

**Code → New Notebook**, then:

- **Settings → Accelerator = GPU** (T4 ×2 or P100).
- **Settings → Internet = On** (only needed for cross-session resume / dataset push).
- **Add Input** → attach **both** datasets (code + data).
- Paste the entire contents of `kaggle_train.py` into the first cell
  (or **File → Import Notebook**).

## Step 4 — Find the exact mount paths

Kaggle slugs are lowercased/hyphenated from your dataset title, so confirm them.
Run this scratch cell **first**:

```python
import os
for root, dirs, files in os.walk("/kaggle/input"):
    if "config.yaml" in files or any(f.endswith(".csv") for f in files):
        print(root, "->", files[:5])
```

- The line printing `config.yaml` → that directory is your **`REPO_DIR`**.
- The line printing the CSVs → that directory is your **`DATA_PATH`**.

## Step 5 — Set the 3 constants at the top of `kaggle_train.py`

```python
REPO_DIR  = "/kaggle/input/<code-slug>/adversarial-ids"  # dir holding src/ AND config.yaml
DATASET   = "cicids2017"                                  # cicids2017 | unsw_nb15 | synthetic
DATA_PATH = "/kaggle/input/<data-slug>"                  # dir holding the 5 CSVs
```

Leave `WORKING = "/kaggle/working"` and `MAX_ROWS = None` as-is
(set `MAX_ROWS` to e.g. `100000` only for a quick debug run).

## Step 6 — Dependencies

Kaggle's default Python image already includes everything the script imports
(numpy, pandas, scikit-learn, xgboost, torch, joblib). **No `pip install` needed.**

The only exception: enabling SMOTE (`preprocess.use_smote: true` in `config.yaml`)
needs `imbalanced-learn`. It is **off by default** here (class weighting is used
instead), so you can ignore this unless you turn it on — then add a first cell:
`!pip install imbalanced-learn`.

## Step 7 — Run

- **Quick interactive check:** press **Run All**. In the logs you should see:

  ```
  Dropping identifier/leaky columns: ['Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'id', 'Attempted Category']
  ```

  and clean metrics with **ROC-AUC clearly below 1.0** (the leak fix working).

- **Real headless run:** **Save Version → Save & Run All (Commit)**. This runs up to
  ~12 h and preserves `/kaggle/working` as the notebook's Output. An interactive tab
  that idles out (~20 min) **wipes** `/kaggle/working` and saves nothing — always use
  Commit for the real run.

## Step 8 — Bring the models back

Download the notebook **Output**, then in your local repo copy:

```
run/cicids2017/models      ->  results/cicids2017/models
run/cicids2017/processed   ->  results/_processed/cicids2017
```

Then continue locally:

```powershell
python scripts/04_generate_attacks.py
# ...and the remaining 05..07 scripts
```

---

## Resuming across sessions (optional)

The script checkpoints every model and can resume. To continue in a **new** session:

1. Finish/interrupt a run via **Save & Run All (Commit)** so its Output is saved.
2. In the next notebook, **Add Input → Notebook Output → your notebook's latest version**.
3. **Run All** again — the script auto-detects the prior `run/<dataset>/` and skips
   finished work.

The models here are small (the whole thing usually finishes in one commit), so treat
resume as insurance against crashes, not a routine step.

---

## ⚠️ Critical: upload the *fixed* code

The data-leak fix lives in `src/data_loaders.py` (dropping `id` and
`Attempted Category`, which otherwise leak the label and produce a fake ~1.0 ROC-AUC).
**Zip and upload the repo *after* that fix is in your working tree** (the Step 1 command
copies your current files, so it will include it). If you previously uploaded an older
code dataset, push a **new version** of it — otherwise Kaggle will train on the leaky
code again and you'll get the same fake perfect scores.

Sanity check after training: if any model reports ROC-AUC ≥ ~0.999 on the clean test
set, treat it as a leak alarm, not a result.
