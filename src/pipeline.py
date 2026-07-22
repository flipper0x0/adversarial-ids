"""End-to-end pipeline (proposal Figure 3.2, the ten stages).

Each stage is a function so the numbered scripts can run them independently
(NFR-4, modular stages). run_all() chains them. Cross-dataset validation is
just run_all() with a different dataset key (Stage 9).

Artifacts per dataset land in results/<dataset>/:
  models/        trained models + scaler + feature names
  processed/     scaled train/test arrays
  summary.json   every metric the dashboard reads
"""
import os

import joblib
import numpy as np
import torch

from . import metrics as M
from .attacks import generate_blackbox, generate_whitebox, transfer
from .config import processed_dir, results_dir
from .constraints import build_constraint_spec
from .data_loaders import load_dataset
from .defenses import adversarial_train, sanitize
from .dependencies import build_consistent_constraint_spec
from .models_classical import build_classical_models, fit_classical
from .models_neural import MLP, mlp_predict, train_mlp, wrap_art
from .preprocess import preprocess
from .utils import get_logger, load_json, save_json, set_seed

log = get_logger("pipeline")


# --------------------------------------------------------------------------- #
# Stage 1-2: data acquisition + preprocessing
# --------------------------------------------------------------------------- #
def stage_preprocess(cfg, dataset):
    set_seed(cfg["seed"])
    ds_cfg = cfg["datasets"][dataset]
    df = load_dataset(dataset, ds_cfg.get("path"), seed=cfg["seed"])
    pre = preprocess(
        df,
        test_size=cfg["split"]["test_size"],
        use_smote=cfg["preprocess"]["use_smote"],
        seed=cfg["seed"],
    )
    out = processed_dir(cfg, dataset)
    np.save(os.path.join(out, "x_train.npy"), pre.x_train)
    np.save(os.path.join(out, "x_test.npy"), pre.x_test)
    np.save(os.path.join(out, "y_train.npy"), pre.y_train)
    np.save(os.path.join(out, "y_test.npy"), pre.y_test)
    save_json(pre.feature_names, os.path.join(out, "feature_names.json"))
    joblib.dump(pre.scaler, os.path.join(out, "scaler.joblib"))
    log.info("Stage 1-2 done. Features=%d  train=%d  test=%d",
             len(pre.feature_names), len(pre.x_train), len(pre.x_test))
    return pre


# --------------------------------------------------------------------------- #
# Stage 3-4: baseline training (RF, XGBoost, SVM, MLP)
# --------------------------------------------------------------------------- #
def stage_train(cfg, dataset):
    set_seed(cfg["seed"])
    d = _load_processed(cfg, dataset)
    mdir = os.path.join(results_dir(cfg, dataset), "models")
    os.makedirs(mdir, exist_ok=True)

    n_pos = int((d["y_train"] == 1).sum())
    n_neg = int((d["y_train"] == 0).sum())

    classical = build_classical_models(cfg, n_pos, n_neg)
    for name, model in classical.items():
        fit_classical(name, model, d["x_train"], d["y_train"], cfg)
        joblib.dump(model, os.path.join(mdir, f"{name}.joblib"))

    mlp = train_mlp(d["x_train"], d["y_train"], cfg)
    torch.save(mlp.state_dict(), os.path.join(mdir, "MLP.pt"))
    in_dim = d["x_train"].shape[1]
    n_classes = int(d["y_train"].max()) + 1
    save_json({"in_dim": in_dim, "n_classes": n_classes,
               "hidden": cfg["models"]["mlp"]["hidden"],
               "dropout": cfg["models"]["mlp"]["dropout"]},
              os.path.join(mdir, "MLP_arch.json"))
    log.info("Stage 3-4 done. Models saved to %s", mdir)


# --------------------------------------------------------------------------- #
# Stage 5: clean evaluation
# --------------------------------------------------------------------------- #
def stage_clean_eval(cfg, dataset):
    d = _load_processed(cfg, dataset)
    models = _load_models(cfg, dataset)
    summary = _load_summary(cfg, dataset)
    summary["dataset"] = dataset
    summary.setdefault("clean", {})

    for name, model in models["classical"].items():
        pred = model.predict(d["x_test"])
        score = model.predict_proba(d["x_test"])[:, 1] if hasattr(model, "predict_proba") else None
        summary["clean"][name] = M.clean_metrics(d["y_test"], pred, score)

    mlp = models["mlp"]
    pred = mlp_predict(mlp, d["x_test"])
    score = _mlp_scores(mlp, d["x_test"])
    summary["clean"]["MLP"] = M.clean_metrics(d["y_test"], pred, score)

    _save_summary(cfg, dataset, summary)
    log.info("Stage 5 done. Clean accuracy: %s",
             {k: round(v["accuracy"], 4) for k, v in summary["clean"].items()})


# --------------------------------------------------------------------------- #
# Stage 6: constrained attack generation (BEFORE defense)
# --------------------------------------------------------------------------- #
def stage_attacks(cfg, dataset):
    d = _load_processed(cfg, dataset)
    models = _load_models(cfg, dataset)
    spec_naive = build_constraint_spec(d["feature_names"], dataset)
    spec_consistent, dep = build_consistent_constraint_spec(d["feature_names"], dataset)
    log.info("Constraint model: %d / %d features mutable (naive); %d base-only (consistent).",
              spec_naive.n_mutable, len(d["feature_names"]), spec_consistent.n_mutable)

    # Attack pool = malicious test samples (capped for runtime).
    mal = d["x_test"][d["y_test"] == 1]
    n = min(cfg["attacks"]["n_eval_samples"], len(mal))
    x_mal = mal[:n]
    y_mal = np.ones(n, dtype=int)

    art = models["art_mlp"]
    summary = _load_summary(cfg, dataset)
    summary.setdefault("attacks_before", {})
    summary.setdefault("attacks_before_consistent", {})

    # White-box on the MLP — NAIVE: perturbs any mutable feature independently,
    # including derived stats (e.g. a mean), without touching the base
    # features they are computed from. Internally inconsistent — this is
    # exactly what the sanitizer's dependency recompute (Stage 8) catches.
    wb = generate_whitebox(art, x_mal, spec_naive, cfg)
    pred_before = mlp_predict(models["mlp"], x_mal)
    acc_clean_mlp = M.accuracy_on_malicious(y_mal, pred_before)
    for atk, x_adv in wb.items():
        pred_after = mlp_predict(models["mlp"], x_adv)
        summary["attacks_before"].setdefault("MLP", {})[atk] = _attack_record(
            y_mal, pred_before, pred_after, acc_clean_mlp
        )

    # White-box on the MLP — CONSISTENT: only base/root features are
    # perturbed; every derived feature is recomputed from them afterward, so
    # the sample stays internally consistent (naive-vs-consistent contrast).
    wb_c = generate_whitebox(art, x_mal, spec_consistent, cfg, dep=dep, scaler=d["scaler"])
    for atk, x_adv in wb_c.items():
        pred_after = mlp_predict(models["mlp"], x_adv)
        summary["attacks_before_consistent"].setdefault("MLP", {})[atk] = _attack_record(
            y_mal, pred_before, pred_after, acc_clean_mlp
        )

    # Transfer: MLP's PGD adversarials replayed against the tree/SVM models.
    x_transfer = transfer(wb["PGD"])
    for name, model in models["classical"].items():
        pb = model.predict(x_mal)
        pa = model.predict(x_transfer)
        acc_clean = M.accuracy_on_malicious(y_mal, pb)
        summary["attacks_before"].setdefault(name, {})["Transfer(PGD)"] = _attack_record(
            y_mal, pb, pa, acc_clean
        )

    # Black-box directly on the deployed models (light; ZOO off by default).
    if cfg["attacks"]["blackbox"]["enabled"]:
        for name, model in models["classical"].items():
            art_tree = _wrap_sklearn(model, d["x_train"].shape[1])
            bb = generate_blackbox(art_tree, x_mal, spec_naive, cfg)
            n_bb = len(next(iter(bb.values())))
            pb = model.predict(x_mal[:n_bb])
            acc_clean = M.accuracy_on_malicious(y_mal[:n_bb], pb)
            for atk, x_adv in bb.items():
                pa = model.predict(x_adv)
                summary["attacks_before"].setdefault(name, {})[atk] = _attack_record(
                    y_mal[:n_bb], pb, pa, acc_clean
                )

    _save_summary(cfg, dataset, summary)
    log.info("Stage 6 done (BEFORE). ASR(MLP/PGD) naive=%.3f consistent=%.3f",
             summary["attacks_before"]["MLP"]["PGD"]["asr"],
             summary["attacks_before_consistent"]["MLP"]["PGD"]["asr"])


# --------------------------------------------------------------------------- #
# Stage 7-8: defenses + robustness re-evaluation (AFTER), DRR, unseen attack
# --------------------------------------------------------------------------- #
def stage_defenses(cfg, dataset):
    d = _load_processed(cfg, dataset)
    models = _load_models(cfg, dataset)
    spec = build_constraint_spec(d["feature_names"], dataset)
    mdir = os.path.join(results_dir(cfg, dataset), "models")

    hardened = adversarial_train(d["x_train"], d["y_train"], models["art_mlp"], spec, cfg)
    torch.save(hardened.state_dict(), os.path.join(mdir, "MLP_hardened.pt"))
    log.info("Stage 7 done. Hardened MLP saved.")


def stage_reeval(cfg, dataset):
    d = _load_processed(cfg, dataset)
    models = _load_models(cfg, dataset)
    spec_naive = build_constraint_spec(d["feature_names"], dataset)
    spec_consistent, dep = build_consistent_constraint_spec(d["feature_names"], dataset)
    scaler = d["scaler"]

    hardened = _load_hardened_mlp(cfg, dataset)
    art_hardened = wrap_art(hardened, d["x_train"].shape[1], int(d["y_train"].max()) + 1, cfg)

    mal = d["x_test"][d["y_test"] == 1]
    n = min(cfg["attacks"]["n_eval_samples"], len(mal))
    x_mal, y_mal = mal[:n], np.ones(min(n, len(mal)), dtype=int)

    summary = _load_summary(cfg, dataset)
    summary.setdefault("after", {})

    # Defended clean accuracy + clean-accuracy trade-off vs original MLP.
    clean_def = M.accuracy_on_malicious(y_mal, mlp_predict(hardened, x_mal))
    summary["after"]["clean_malicious_acc_hardened"] = clean_def
    summary["after"]["clean_malicious_acc_baseline"] = M.accuracy_on_malicious(
        y_mal, mlp_predict(models["mlp"], x_mal)
    )

    # Re-run white-box on the hardened model.
    wb = generate_whitebox(art_hardened, x_mal, spec_naive, cfg)
    pred_before = mlp_predict(hardened, x_mal)
    acc_clean = M.accuracy_on_malicious(y_mal, pred_before)
    for atk, x_adv in wb.items():
        pred_after = mlp_predict(hardened, x_adv)
        rec = _attack_record(y_mal, pred_before, pred_after, acc_clean)
        # DRR vs the undefended result recorded in Stage 6.
        undef = summary["attacks_before"]["MLP"][atk]
        rec["drr"] = M.defense_recovery_rate(
            acc_clean=undef["acc_clean_malicious"],
            acc_undef_attacked=undef["acc_under_attack"],
            acc_def_attacked=rec["acc_under_attack"],
        )
        summary["after"].setdefault("adv_training_MLP", {})[atk] = rec

    # --- Sanitization defense, applied to the ORIGINAL model at inference. ---
    # sanitize() = range clip + feature squeeze + recompute derived features
    # from base features (src/defenses.py). It is applied identically to both
    # attack variants below — the sanitizer has no notion of which kind of
    # attack it is looking at, so this is an honest test, not a rigged one.
    pred_before_san = mlp_predict(
        models["mlp"], sanitize(x_mal, spec_naive, cfg, dep=dep, scaler=scaler)
    )
    acc_clean_san = M.accuracy_on_malicious(y_mal, pred_before_san)

    # NAIVE attack samples through the sanitizer: expected to be defeated —
    # sanitize() recomputes the very derived features the naive attack
    # perturbed independently of their base features.
    summary["after"].setdefault("sanitization_MLP_naive", {})
    wb_naive = generate_whitebox(models["art_mlp"], x_mal, spec_naive, cfg)
    for atk, x_adv in wb_naive.items():
        x_san = sanitize(x_adv, spec_naive, cfg, dep=dep, scaler=scaler)
        pred_after = mlp_predict(models["mlp"], x_san)
        summary["after"]["sanitization_MLP_naive"][atk] = _attack_record(
            y_mal, pred_before_san, pred_after, acc_clean_san
        )

    # CONSISTENT attack samples through the SAME sanitizer: expected to
    # survive — their derived features already match their (perturbed) base
    # features, so sanitize()'s recompute step is a no-op on them.
    summary["after"].setdefault("sanitization_MLP_consistent", {})
    wb_consistent = generate_whitebox(
        models["art_mlp"], x_mal, spec_consistent, cfg, dep=dep, scaler=scaler
    )
    for atk, x_adv in wb_consistent.items():
        x_san = sanitize(x_adv, spec_consistent, cfg, dep=dep, scaler=scaler)
        pred_after = mlp_predict(models["mlp"], x_san)
        summary["after"]["sanitization_MLP_consistent"][atk] = _attack_record(
            y_mal, pred_before_san, pred_after, acc_clean_san
        )

    # Backward-compatible alias for older dashboards/scripts.
    summary["after"]["sanitization_MLP"] = summary["after"]["sanitization_MLP_naive"]

    # Unseen-attack generalisation: defense built on PGD, tested on JSMA.
    summary["after"]["unseen_attack_note"] = (
        "adv_training built on PGD; JSMA/black-box columns above are unseen attacks."
    )

    _save_summary(cfg, dataset, summary)
    log.info("Stage 8 done (AFTER). DRR(MLP/PGD)=%s  "
             "ASR after sanitize: naive/PGD=%.3f consistent/PGD=%.3f",
             summary["after"]["adv_training_MLP"]["PGD"]["drr"],
             summary["after"]["sanitization_MLP_naive"]["PGD"]["asr"],
             summary["after"]["sanitization_MLP_consistent"]["PGD"]["asr"])


# --------------------------------------------------------------------------- #
# Stage 9-10: cross-dataset + (dashboard is a separate app)
# --------------------------------------------------------------------------- #
def run_all(cfg, dataset):
    log.info("===== PIPELINE START: %s =====", dataset)
    stage_preprocess(cfg, dataset)
    stage_train(cfg, dataset)
    stage_clean_eval(cfg, dataset)
    stage_attacks(cfg, dataset)
    stage_defenses(cfg, dataset)
    stage_reeval(cfg, dataset)
    log.info("===== PIPELINE DONE: %s =====", dataset)


# ----------------------------- helpers ------------------------------------- #
def _attack_record(y_true, pred_before, pred_after, acc_clean):
    acc_attacked = M.accuracy_on_malicious(y_true, pred_after)
    return {
        "asr": M.attack_success_rate(y_true, pred_before, pred_after),
        "acc_clean_malicious": acc_clean,
        "acc_under_attack": acc_attacked,
        "robustness_score": M.robustness_score(acc_clean, acc_attacked),
    }


def _load_processed(cfg, dataset):
    out = processed_dir(cfg, dataset)
    return {
        "x_train": np.load(os.path.join(out, "x_train.npy")),
        "x_test": np.load(os.path.join(out, "x_test.npy")),
        "y_train": np.load(os.path.join(out, "y_train.npy")),
        "y_test": np.load(os.path.join(out, "y_test.npy")),
        "feature_names": load_json(os.path.join(out, "feature_names.json")),
        "scaler": joblib.load(os.path.join(out, "scaler.joblib")),
    }


def _load_models(cfg, dataset):
    mdir = os.path.join(results_dir(cfg, dataset), "models")
    classical = {}
    for name in ["RandomForest", "XGBoost", "SVM"]:
        p = os.path.join(mdir, f"{name}.joblib")
        if os.path.exists(p):
            classical[name] = joblib.load(p)
    mlp = _load_mlp(cfg, dataset, "MLP.pt")
    d = _load_processed(cfg, dataset)
    art_mlp = wrap_art(mlp, d["x_train"].shape[1], int(d["y_train"].max()) + 1, cfg)
    return {"classical": classical, "mlp": mlp, "art_mlp": art_mlp}


def _load_mlp(cfg, dataset, fname):
    mdir = os.path.join(results_dir(cfg, dataset), "models")
    arch = load_json(os.path.join(mdir, "MLP_arch.json"))
    model = MLP(arch["in_dim"], arch["n_classes"], tuple(arch["hidden"]), arch["dropout"])
    model.load_state_dict(torch.load(os.path.join(mdir, fname), map_location="cpu"))
    model.eval()
    return model


def _load_hardened_mlp(cfg, dataset):
    return _load_mlp(cfg, dataset, "MLP_hardened.pt")


def _mlp_scores(mlp, x):
    import torch.nn.functional as Fnn

    mlp.eval()
    with torch.no_grad():
        logits = mlp(torch.tensor(x, dtype=torch.float32))
        return Fnn.softmax(logits, dim=1)[:, 1].cpu().numpy()


def _wrap_sklearn(model, in_dim):
    from art.estimators.classification import SklearnClassifier

    return SklearnClassifier(model=model, clip_values=(0.0, 1.0))


def _load_summary(cfg, dataset):
    p = os.path.join(results_dir(cfg, dataset), "summary.json")
    return load_json(p) if os.path.exists(p) else {}


def _save_summary(cfg, dataset, summary):
    save_json(summary, os.path.join(results_dir(cfg, dataset), "summary.json"))
