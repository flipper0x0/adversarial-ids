"""Evaluation metrics (proposal section 3.3).

Label convention everywhere: 1 = malicious/attack, 0 = benign/normal.
Evasion = a malicious sample (1) that the model now calls benign (0).

The three project-specific metrics are implemented exactly as defined in the
proposal:

  ASR  = (# malicious samples that evade after attack)
         / (# malicious samples correctly detected before attack)         (3.1)

  R    = accuracy under attack / accuracy on clean data                    (3.2)

  DRR  = (acc_defended_attacked - acc_undefended_attacked)
         / (acc_clean - acc_undefended_attacked)                          (3.3)
"""
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def clean_metrics(y_true, y_pred, y_score: Optional[np.ndarray] = None) -> Dict:
    """Standard classification metrics. ROC-AUC reported because the datasets
    are heavily imbalanced and accuracy alone is misleading (proposal 3.3.1)."""
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_score is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            out["roc_auc"] = None
    return out


def attack_success_rate(y_true, pred_before, pred_after) -> float:
    """Equation 3.1. Denominator = malicious samples the model got right
    pre-attack; numerator = those now misclassified as benign."""
    y_true = np.asarray(y_true)
    pred_before = np.asarray(pred_before)
    pred_after = np.asarray(pred_after)

    detected = (y_true == 1) & (pred_before == 1)
    denom = int(detected.sum())
    if denom == 0:
        return float("nan")
    evaded = int(((pred_after == 0) & detected).sum())
    return evaded / denom


def robustness_score(acc_clean: float, acc_under_attack: float) -> float:
    """Equation 3.2. Share of clean accuracy that survives the attack."""
    if acc_clean == 0:
        return float("nan")
    return acc_under_attack / acc_clean


def defense_recovery_rate(acc_clean, acc_undef_attacked, acc_def_attacked) -> float:
    """Equation 3.3. Fraction of attack-induced accuracy loss the defense
    recovers. Undefined (NaN) when the attack caused no loss to recover."""
    denom = acc_clean - acc_undef_attacked
    if abs(denom) < 1e-12:
        return float("nan")
    return (acc_def_attacked - acc_undef_attacked) / denom


def accuracy_on_malicious(y_true, y_pred) -> float:
    """Detection accuracy restricted to malicious traffic — the quantity that
    actually degrades under evasion. Used for R and DRR so the metrics are not
    diluted by the easy benign class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mal = y_true == 1
    if mal.sum() == 0:
        return float("nan")
    return float((y_pred[mal] == 1).mean())
