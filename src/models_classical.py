"""Classical baseline detectors: Random Forest, XGBoost, SVM.

Honest scalability note baked into the code:
A kernel SVM does NOT scale to the full CICIDS2017 (~2.8M flows) — training is
roughly quadratic in samples and will not finish. We therefore train the SVM
on a stratified subsample (size set in config: models.svm.train_subsample).
If you need the SVM to see all data, switch to a linear SVM (LinearSVC /
SGDClassifier), which scales but is weaker on non-linear boundaries. RF and
XGBoost train on the full set.
"""
from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .utils import get_logger

log = get_logger("models_classical")


def build_classical_models(cfg, n_pos: int, n_neg: int) -> Dict:
    """Instantiate the three classical models with imbalance handling."""
    scale_pos_weight = max(1.0, n_neg / max(1, n_pos))  # XGBoost imbalance knob
    m = cfg["models"]

    rf = RandomForestClassifier(
        n_estimators=m["rf"]["n_estimators"],
        max_depth=m["rf"]["max_depth"],
        n_jobs=-1,
        class_weight="balanced",
        random_state=cfg["seed"],
    )
    xgb = XGBClassifier(
        n_estimators=m["xgb"]["n_estimators"],
        max_depth=m["xgb"]["max_depth"],
        learning_rate=m["xgb"]["learning_rate"],
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=cfg["seed"],
    )
    svm = SVC(
        C=m["svm"]["C"],
        kernel=m["svm"]["kernel"],
        gamma=m["svm"]["gamma"],
        class_weight="balanced",
        probability=True,   # needed for ROC-AUC and score-based black-box attacks
        random_state=cfg["seed"],
    )
    return {"RandomForest": rf, "XGBoost": xgb, "SVM": svm}


def fit_classical(name: str, model, x_train, y_train, cfg):
    """Fit one model, subsampling the training set for SVM if it is large."""
    if name == "SVM":
        cap = cfg["models"]["svm"]["train_subsample"]
        if cap and len(x_train) > cap:
            idx = _stratified_subsample(y_train, cap, cfg["seed"])
            log.warning("SVM: subsampling %d -> %d rows for tractable training.",
                        len(x_train), len(idx))
            x_train, y_train = x_train[idx], y_train[idx]
    log.info("Fitting %s on %d rows ...", name, len(x_train))
    model.fit(x_train, y_train)
    return model


def _stratified_subsample(y, cap, seed):
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(y))
    out = []
    for cls in np.unique(y):
        cls_idx = idx_all[y == cls]
        take = int(round(cap * len(cls_idx) / len(y)))
        take = min(take, len(cls_idx))
        out.append(rng.choice(cls_idx, size=take, replace=False))
    return np.concatenate(out)
