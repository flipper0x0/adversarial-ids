"""
================================================================================
 kaggle_train.py  —  TRAINING ONLY, built for Kaggle GPU, fully resumable.
================================================================================

What this does (and nothing more):
  Stage 1-2  preprocess the dataset           (reuses src/preprocess.py)
  Stage 4    train RandomForest, XGBoost, SVM, MLP   (with checkpoints)
  Stage 5    clean evaluation metrics on the test set (ASR/PGD/etc. are NOT here)

It deliberately does NOT run attacks or defenses. Train here on Kaggle's GPU,
download the trained models, then run scripts 04-07 locally for the
attack/defense experiments. The artifacts this writes are byte-compatible with
your local pipeline's expected layout (results/<dataset>/models + processed),
so you can drop them straight into your repo.

--------------------------------------------------------------------------------
 READ THIS ABOUT KAGGLE PERSISTENCE — it is the whole point of the checkpointing
--------------------------------------------------------------------------------
Kaggle facts you must internalise, or you will lose work and burn GPU quota:

1. /kaggle/working is the ONLY writable folder that can be saved. Everything
   else (including /kaggle/input) is read-only.

2. /kaggle/working is saved as the notebook's "output" ONLY when you run via
   "Save Version -> Save & Run All (Commit)" and the run finishes (or hits the
   time limit having saved). In a normal INTERACTIVE session, if the tab
   disconnects or idles out (~20 min idle), /kaggle/working is WIPED and
   nothing is saved. Checkpointing cannot rescue an idle-timed-out interactive
   session. For any long run: use Save & Run All (Commit), which runs headless
   up to the GPU limit (~12 h).

3. To RESUME in a brand-new session, the previous checkpoints must be mounted
   as INPUT. Two ways (this script auto-detects both):
     (a) Add Input -> "Notebook Output" -> your own notebook's latest version.
         It mounts at /kaggle/input/<your-notebook-slug>/.
     (b) Keep a Kaggle Dataset of checkpoints and attach it. (Permanent,
         decoupled from the notebook. Optional API snippet at the bottom.)

   On start, this script searches /kaggle/working first, then every
   /kaggle/input/*/ mount, for a previous run dir and copies it into
   /kaggle/working to continue.

Honest expectation for THIS project: the models are small. The MLP is the only
real GPU job and is short; RF/XGB/SVM are minutes (SVM is subsampled). The whole
thing very likely finishes inside ONE 12 h commit. So treat cross-session resume
as insurance against crashes / quota exhaustion, not a routine need. Do not
over-engineer your workflow around it.

--------------------------------------------------------------------------------
 SETUP ON KAGGLE (one-time)
--------------------------------------------------------------------------------
 1. Upload your repo so `src/` and `config.yaml` are importable. Easiest:
    zip the repo, create a Kaggle Dataset from it. It will mount at e.g.
    /kaggle/input/adversarial-ids-code/adversarial-ids/. Set REPO_DIR below.
    (Alternative: enable Internet and `!git clone <your repo>` in a first cell.)
 2. Attach the data as a Kaggle Dataset (CICIDS2017 corrected, and/or
    UNSW-NB15). Set DATA_PATH below to the folder of CSVs.
 3. Settings: Accelerator = GPU; for cross-session resume / dataset push,
    Internet = On.
 4. Put this file's contents in a notebook cell (or add it as a utility script)
    and Run All. To resume later, attach the previous output/dataset (see above)
    and Run All again — it will skip finished work.
================================================================================
"""

# ============================================================================ #
#  CONFIG — edit these, then Run All.                                          #
# ============================================================================ #
REPO_DIR  = "/kaggle/input/adversarial-ids-code/adversarial-ids"  # holds src/ + config.yaml
DATASET   = "cicids2017"          # "cicids2017" | "unsw_nb15" | "synthetic"
DATA_PATH = "/kaggle/input/cicids2017/cicids2017"  # folder of CSVs; ignored for "synthetic"

WORKING   = "/kaggle/working"     # do not change on Kaggle
MAX_ROWS  = None                  # cap rows (memory/debug). None = use all rows.

# Checkpoint granularity (how often progress is saved -> how finely you resume).
RF_TREES_PER_CKPT  = 50           # RandomForest: trees added per checkpoint
XGB_ROUNDS_PER_CKPT = 50          # XGBoost: boosting rounds added per checkpoint
MLP_CKPT_EVERY     = 1            # MLP: save every N epochs
# ============================================================================ #

import json
import os
import shutil
import sys
import time

# --- make src/ importable, then reuse the project's real logic --------------- #
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

import numpy as np

try:
    from src.config import load_config
    from src.data_loaders import load_dataset
    from src.preprocess import preprocess
    from src.models_classical import build_classical_models, fit_classical
    from src.models_neural import MLP
    from src import metrics as M
    from src.utils import set_seed
except ModuleNotFoundError as e:
    raise SystemExit(
        f"Could not import the project from REPO_DIR={REPO_DIR!r} ({e}).\n"
        "Fix REPO_DIR so that REPO_DIR/src/__init__.py and REPO_DIR/config.yaml exist.\n"
        "On Kaggle: upload the repo as a Dataset and point REPO_DIR at the mounted folder."
    )

import torch
import torch.nn as nn
import joblib

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================ #
#  Checkpoint store: one run directory holding processed arrays, models, and a #
#  manifest (state.json). Everything lives under /kaggle/working so a commit    #
#  saves it; resume copies a prior run dir back into /kaggle/working.           #
# ============================================================================ #
RUN_DIR = os.path.join(WORKING, "run", DATASET)
PROC_DIR = os.path.join(RUN_DIR, "processed")
MODEL_DIR = os.path.join(RUN_DIR, "models")
MANIFEST = os.path.join(RUN_DIR, "state.json")


def _empty_manifest(cfg):
    return {
        "dataset": DATASET,
        "models": {
            "RandomForest": {"trees_done": 0, "target": cfg["models"]["rf"]["n_estimators"], "done": False},
            "XGBoost":      {"rounds_done": 0, "target": cfg["models"]["xgb"]["n_estimators"], "done": False},
            "SVM":          {"done": False},
            "MLP":          {"epoch_done": 0, "target": cfg["models"]["mlp"]["epochs"], "done": False},
        },
    }


def load_manifest():
    with open(MANIFEST) as f:
        return json.load(f)


def _atomic_save_manifest(man):
    os.makedirs(RUN_DIR, exist_ok=True)
    tmp = MANIFEST + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, indent=2)
    os.replace(tmp, MANIFEST)


def discover_and_restore_prior_run():
    """If a previous run dir exists (in working, or mounted under /kaggle/input),
    copy it into /kaggle/working so we can continue writing to it."""
    if os.path.exists(MANIFEST):
        print(f"[resume] found checkpoints already in working: {RUN_DIR}")
        return True

    # Scan input mounts for a matching run dir: /kaggle/input/*/run/<dataset>/state.json
    candidates = []
    input_root = "/kaggle/input"
    if os.path.isdir(input_root):
        for entry in os.listdir(input_root):
            base = os.path.join(input_root, entry)
            for rel in (os.path.join("run", DATASET), os.path.join(entry, "run", DATASET)):
                cand = os.path.join(base, rel)
                if os.path.exists(os.path.join(cand, "state.json")):
                    candidates.append(cand)
    if not candidates:
        print("[resume] no prior checkpoints found -> starting fresh.")
        return False

    # Pick the most-recently-modified candidate.
    src = max(candidates, key=lambda p: os.path.getmtime(os.path.join(p, "state.json")))
    print(f"[resume] restoring prior run from input mount:\n          {src}\n      ->  {RUN_DIR}")
    os.makedirs(os.path.dirname(RUN_DIR), exist_ok=True)
    if os.path.exists(RUN_DIR):
        shutil.rmtree(RUN_DIR)
    shutil.copytree(src, RUN_DIR)
    return True


# ============================================================================ #
#  Stage 1-2: preprocessing (cached so resume does not redo it)                #
# ============================================================================ #
def get_data(cfg):
    xtr = os.path.join(PROC_DIR, "x_train.npy")
    if os.path.exists(xtr):
        print("[data] loading cached processed arrays (skipping preprocessing).")
        return {
            "x_train": np.load(os.path.join(PROC_DIR, "x_train.npy")),
            "x_test":  np.load(os.path.join(PROC_DIR, "x_test.npy")),
            "y_train": np.load(os.path.join(PROC_DIR, "y_train.npy")),
            "y_test":  np.load(os.path.join(PROC_DIR, "y_test.npy")),
            "feature_names": json.load(open(os.path.join(PROC_DIR, "feature_names.json"))),
        }

    print(f"[data] loading dataset '{DATASET}' from {DATA_PATH} ...")
    path = None if DATASET == "synthetic" else DATA_PATH
    df = load_dataset(DATASET, path, seed=cfg["seed"])

    if MAX_ROWS and len(df) > MAX_ROWS:
        print(f"[data] capping {len(df)} -> {MAX_ROWS} rows (MAX_ROWS set; for debug only).")
        df = df.sample(n=MAX_ROWS, random_state=cfg["seed"]).reset_index(drop=True)

    pre = preprocess(
        df,
        test_size=cfg["split"]["test_size"],
        use_smote=cfg["preprocess"]["use_smote"],
        seed=cfg["seed"],
    )
    os.makedirs(PROC_DIR, exist_ok=True)
    np.save(os.path.join(PROC_DIR, "x_train.npy"), pre.x_train)
    np.save(os.path.join(PROC_DIR, "x_test.npy"), pre.x_test)
    np.save(os.path.join(PROC_DIR, "y_train.npy"), pre.y_train)
    np.save(os.path.join(PROC_DIR, "y_test.npy"), pre.y_test)
    json.dump(pre.feature_names, open(os.path.join(PROC_DIR, "feature_names.json"), "w"))
    joblib.dump(pre.scaler, os.path.join(PROC_DIR, "scaler.joblib"))
    print(f"[data] features={len(pre.feature_names)} train={len(pre.x_train)} test={len(pre.x_test)}")
    return {
        "x_train": pre.x_train, "x_test": pre.x_test,
        "y_train": pre.y_train, "y_test": pre.y_test,
        "feature_names": pre.feature_names,
    }


# ============================================================================ #
#  Stage 4: training, one resumable routine per model.                         #
# ============================================================================ #
def train_random_forest(cfg, d, man):
    info = man["models"]["RandomForest"]
    if info["done"]:
        print("[RF] already complete -> skip.")
        return
    target = info["target"]
    path = os.path.join(MODEL_DIR, "RandomForest.joblib")

    # warm_start lets us GROW the forest in chunks and checkpoint between them.
    # With a fixed random_state the trees added are deterministic, so the final
    # forest is equivalent to fitting all `target` trees in one call.
    if info["trees_done"] > 0 and os.path.exists(path):
        rf = joblib.load(path)
        print(f"[RF] resuming at {info['trees_done']}/{target} trees.")
    else:
        rf = build_classical_models(cfg, int((d["y_train"] == 1).sum()), int((d["y_train"] == 0).sum()))["RandomForest"]
        info["trees_done"] = 0
    rf.set_params(warm_start=True)

    while info["trees_done"] < target:
        nxt = min(info["trees_done"] + RF_TREES_PER_CKPT, target)
        rf.set_params(n_estimators=nxt)
        t0 = time.time()
        rf.fit(d["x_train"], d["y_train"])
        info["trees_done"] = nxt
        os.makedirs(MODEL_DIR, exist_ok=True)
        joblib.dump(rf, path)
        _atomic_save_manifest(man)
        print(f"[RF] {nxt}/{target} trees  (+{time.time()-t0:.1f}s)  checkpoint saved.")

    info["done"] = True
    _atomic_save_manifest(man)
    print("[RF] done.")


def train_xgboost(cfg, d, man):
    info = man["models"]["XGBoost"]
    if info["done"]:
        print("[XGB] already complete -> skip.")
        return
    target = info["target"]
    booster_path = os.path.join(MODEL_DIR, "XGBoost.booster.json")
    final_path = os.path.join(MODEL_DIR, "XGBoost.joblib")

    xgb = build_classical_models(cfg, int((d["y_train"] == 1).sum()), int((d["y_train"] == 0).sum()))["XGBoost"]
    if DEVICE.type == "cuda":
        xgb.set_params(device="cuda")   # xgboost 2.x GPU; tree_method 'hist' supports it
        print("[XGB] using GPU (device=cuda).")

    # XGBoost continues from a saved booster via xgb_model=, so each chunk
    # appends rounds to the previous ones. The result is a complete, valid
    # `target`-round model. Note: because rows are subsampled per round
    # (subsample<1), a chunked run is NOT bit-identical to one uninterrupted
    # run (~99% prediction agreement in practice) — it is a fully-trained model,
    # not a byte-reproduction. Set subsample=1.0 in config if you need exactness.
    while info["rounds_done"] < target:
        add = min(XGB_ROUNDS_PER_CKPT, target - info["rounds_done"])
        xgb.set_params(n_estimators=add)
        prev = booster_path if info["rounds_done"] > 0 and os.path.exists(booster_path) else None
        t0 = time.time()
        xgb.fit(d["x_train"], d["y_train"], xgb_model=prev)
        os.makedirs(MODEL_DIR, exist_ok=True)
        xgb.get_booster().save_model(booster_path)  # cumulative booster
        info["rounds_done"] += add
        _atomic_save_manifest(man)
        print(f"[XGB] {info['rounds_done']}/{target} rounds  (+{time.time()-t0:.1f}s)  checkpoint saved.")

    # Save a full XGBClassifier (joblib) so the local pipeline can load it as-is.
    xgb.set_params(n_estimators=target)  # metadata matches actual tree count
    joblib.dump(xgb, final_path)
    info["done"] = True
    _atomic_save_manifest(man)
    print("[XGB] done.")


def train_svm(cfg, d, man):
    info = man["models"]["SVM"]
    path = os.path.join(MODEL_DIR, "SVM.joblib")
    if info["done"] and os.path.exists(path):
        print("[SVM] already complete -> skip.")
        return
    # HONEST LIMITATION: a kernel SVM has no incremental/warm-start fit. It is a
    # single atomic .fit(); if the session dies mid-fit it restarts from zero.
    # This is bounded only because the SVM trains on a stratified subsample
    # (config: models.svm.train_subsample). For the full data switch to
    # LinearSVC/SGDClassifier, which is incremental but weaker on non-linear data.
    print("[SVM] training (atomic; cannot resume mid-fit — subsample keeps it bounded) ...")
    svm = build_classical_models(cfg, int((d["y_train"] == 1).sum()), int((d["y_train"] == 0).sum()))["SVM"]
    t0 = time.time()
    fit_classical("SVM", svm, d["x_train"], d["y_train"], cfg)  # reuses repo's subsample logic
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(svm, path)
    info["done"] = True
    _atomic_save_manifest(man)
    print(f"[SVM] done (+{time.time()-t0:.1f}s).")


def train_mlp(cfg, d, man):
    info = man["models"]["MLP"]
    target = info["target"]
    ckpt = os.path.join(MODEL_DIR, "MLP_ckpt.pt")
    if info["done"]:
        print("[MLP] already complete -> skip.")
        return

    p = cfg["models"]["mlp"]
    in_dim = d["x_train"].shape[1]
    n_classes = int(d["y_train"].max()) + 1

    # Same setup as src/models_neural.train_mlp (so the trained model matches),
    # but with a per-epoch checkpoint loop wrapped around it.
    xt = torch.tensor(d["x_train"], dtype=torch.float32)
    yt = torch.tensor(d["y_train"], dtype=torch.long)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xt, yt), batch_size=p["batch_size"], shuffle=True
    )
    counts = np.bincount(d["y_train"], minlength=n_classes).astype(np.float32)
    weights = counts.sum() / (n_classes * np.maximum(counts, 1))
    weights_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    model = MLP(in_dim, n_classes, tuple(p["hidden"]), p["dropout"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    loss_fn = nn.CrossEntropyLoss(weight=weights_t)

    start_epoch = 0
    if info["epoch_done"] > 0 and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location=DEVICE)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch = state["epoch"]
        try:
            torch.set_rng_state(state["torch_rng"].cpu() if hasattr(state["torch_rng"], "cpu") else state["torch_rng"])
            np.random.set_state(state["numpy_rng"])
        except Exception:
            pass
        print(f"[MLP] resuming at epoch {start_epoch}/{target}.")
    else:
        print(f"[MLP] training from scratch, target {target} epochs, on {DEVICE}.")

    arch = {"in_dim": in_dim, "n_classes": n_classes, "hidden": p["hidden"], "dropout": p["dropout"]}

    for epoch in range(start_epoch, target):
        model.train()
        total, t0 = 0.0, time.time()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)

        done = epoch + 1
        if done % MLP_CKPT_EVERY == 0 or done == target:
            os.makedirs(MODEL_DIR, exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(), "epoch": done,
                 "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(), "arch": arch},
                ckpt,
            )
            info["epoch_done"] = done
            _atomic_save_manifest(man)
        print(f"[MLP] epoch {done}/{target}  loss={total/len(dl.dataset):.4f}  (+{time.time()-t0:.1f}s)")

    # Save in the exact format the local pipeline expects (state_dict + arch json).
    model.eval()
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "MLP.pt"))
    json.dump(arch, open(os.path.join(MODEL_DIR, "MLP_arch.json"), "w"))
    info["done"] = True
    _atomic_save_manifest(man)
    print("[MLP] done.")


# ============================================================================ #
#  Stage 5: clean evaluation metrics (no attacks — those run locally).         #
# ============================================================================ #
def evaluate(cfg, d):
    print("\n[eval] clean metrics on the held-out test set ...")
    summary = {"dataset": DATASET, "clean": {}}

    for name in ("RandomForest", "XGBoost", "SVM"):
        path = os.path.join(MODEL_DIR, f"{name}.joblib")
        if not os.path.exists(path):
            print(f"[eval] {name}: model missing, skipped.")
            continue
        model = joblib.load(path)
        pred = model.predict(d["x_test"])
        score = model.predict_proba(d["x_test"])[:, 1] if hasattr(model, "predict_proba") else None
        summary["clean"][name] = M.clean_metrics(d["y_test"], pred, score)

    # MLP
    arch_path = os.path.join(MODEL_DIR, "MLP_arch.json")
    mlp_path = os.path.join(MODEL_DIR, "MLP.pt")
    if os.path.exists(arch_path) and os.path.exists(mlp_path):
        arch = json.load(open(arch_path))
        mlp = MLP(arch["in_dim"], arch["n_classes"], tuple(arch["hidden"]), arch["dropout"]).to(DEVICE)
        mlp.load_state_dict(torch.load(mlp_path, map_location=DEVICE))
        mlp.eval()
        with torch.no_grad():
            logits = mlp(torch.tensor(d["x_test"], dtype=torch.float32, device=DEVICE))
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = logits.argmax(dim=1).cpu().numpy()
        summary["clean"]["MLP"] = M.clean_metrics(d["y_test"], pred, prob)

    out = os.path.join(RUN_DIR, "clean_summary.json")
    json.dump(summary, open(out, "w"), indent=2)
    print("\n================ CLEAN RESULTS ================")
    for name, m in summary["clean"].items():
        auc = m.get("roc_auc")
        auc_s = f"{auc:.4f}" if isinstance(auc, float) else "n/a"
        print(f"  {name:13s}  acc={m['accuracy']:.4f}  precision={m['precision']:.4f}  "
              f"recall={m['recall']:.4f}  f1={m['f1']:.4f}  roc_auc={auc_s}")
    print(f"\n[eval] written to {out}")
    return summary


# ============================================================================ #
#  Main                                                                        #
# ============================================================================ #
def main():
    print(f"Device: {DEVICE}"
          + (f"  ({torch.cuda.get_device_name(0)})" if DEVICE.type == "cuda" else "  (no GPU — training on CPU)"))

    cfg = load_config()                  # reads REPO_DIR/config.yaml
    set_seed(cfg["seed"])

    restored = discover_and_restore_prior_run()
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MANIFEST):
        _atomic_save_manifest(_empty_manifest(cfg))
    man = load_manifest()

    d = get_data(cfg)

    train_random_forest(cfg, d, man)
    train_xgboost(cfg, d, man)
    train_svm(cfg, d, man)
    train_mlp(cfg, d, man)

    evaluate(cfg, d)

    print("\n================ DONE ================")
    print(f"All artifacts are under: {RUN_DIR}")
    print("  processed/        scaled arrays + scaler + feature_names")
    print("  models/           RandomForest.joblib, XGBoost.joblib, SVM.joblib, MLP.pt (+ arch)")
    print("  clean_summary.json")
    print("\nTO PERSIST FOR A LATER SESSION (so resume works):")
    print("  - Commit the notebook (Save Version -> Save & Run All). The folder above")
    print("    becomes this version's OUTPUT.")
    print("  - Next time, before Run All: Add Input -> Notebook Output -> this notebook's")
    print("    latest version. This script will detect it and skip finished work.")
    print("\nTO USE THE MODELS LOCALLY:")
    print("  Download the output, copy run/<dataset>/models -> results/<dataset>/models")
    print("  and run/<dataset>/processed -> results/_processed/<dataset> in your repo,")
    print("  then run scripts/04_generate_attacks.py onward.")


if __name__ == "__main__":
    main()


# ============================================================================ #
#  OPTIONAL — permanent checkpoints via a Kaggle Dataset (needs Internet = On  #
#  and your kaggle.json token added as a Kaggle Secret / uploaded).            #
#  Run this at the END of a session to push the latest checkpoints; attach the #
#  dataset as input next time and this script will restore from it.            #
# ---------------------------------------------------------------------------- #
#  import os
#  os.makedirs("/kaggle/working/ckpt_upload", exist_ok=True)
#  import shutil
#  shutil.copytree(RUN_DIR, f"/kaggle/working/ckpt_upload/run/{DATASET}", dirs_exist_ok=True)
#  meta = {
#      "title": "adversarial-ids-checkpoints",
#      "id": "YOUR_KAGGLE_USERNAME/adversarial-ids-checkpoints",
#      "licenses": [{"name": "CC0-1.0"}],
#  }
#  import json
#  json.dump(meta, open("/kaggle/working/ckpt_upload/dataset-metadata.json", "w"))
#  # First time only: create the dataset
#  # !kaggle datasets create -p /kaggle/working/ckpt_upload --dir-mode zip
#  # Subsequent times: push a new version
#  # !kaggle datasets version -p /kaggle/working/ckpt_upload -m "resume checkpoint" --dir-mode zip
# ============================================================================ #
