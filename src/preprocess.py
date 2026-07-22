"""Preprocessing (proposal section 3.2.2).

Decisions made here, with reasons you should be ready to defend:

* MinMax scaling to [0, 1] (not StandardScaler). For adversarial work this
  matters: an L-inf budget of eps means "eps of each feature's full range",
  which is interpretable and comparable across features. StandardScaler would
  make eps mean different things per feature.

* Class imbalance is handled by CLASS WEIGHTING by default, not SMOTE.
  SMOTE on network-flow data can synthesise records that violate the same
  feature dependencies we are trying to respect in the attack model, and if
  applied before the split it leaks. SMOTE is available via config but off by
  default, and is only ever fit on the training split.

* Time-aware split when a timestamp exists. A random split lets near-duplicate
  flows from the same burst land in both train and test, inflating accuracy.
  Chronological split (train = earlier, test = later) is closer to deployment.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder

from .utils import get_logger

log = get_logger("preprocess")

_TIMESTAMP_CANDIDATES = ["Timestamp", "timestamp", "Stime", "stime"]
_CATEGORICAL_CANDIDATES = ["proto", "service", "state", "Protocol"]


@dataclass
class Preprocessed:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    scaler: MinMaxScaler


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    use_smote: bool = False,
    seed: int = 42,
) -> Preprocessed:
    df = df.copy()

    # 1. Pull out and order by timestamp if present (for the split), then drop it.
    ts_col = next((c for c in _TIMESTAMP_CANDIDATES if c in df.columns), None)
    if ts_col is not None:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.sort_values(ts_col).reset_index(drop=True)

    # 2. Separate target and reference columns from features.
    y = df["Label"].astype(int).values
    drop_cols = ["Label", "attack_cat"]
    if ts_col is not None:
        drop_cols.append(ts_col)
    feat_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 3. Encode categorical columns ordinally (one column each -> easy to mark
    #    immutable in the constraint model). One-hot is an alternative but
    #    complicates feature masking.
    cat_cols = [c for c in _CATEGORICAL_CANDIDATES if c in feat_df.columns]
    cat_cols = [c for c in cat_cols if feat_df[c].dtype == object or str(feat_df[c].dtype).startswith("category")]
    # also catch any remaining object columns
    cat_cols = list(dict.fromkeys(cat_cols + [c for c in feat_df.columns if feat_df[c].dtype == object]))
    if cat_cols:
        log.info("Encoding categorical columns: %s", cat_cols)
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        feat_df[cat_cols] = enc.fit_transform(feat_df[cat_cols].astype(str))

    feat_df = feat_df.apply(pd.to_numeric, errors="coerce")

    # 4. Clean: drop rows with NaN/Inf (after inf->nan upstream), drop exact dupes.
    before = len(feat_df)
    mask_finite = np.isfinite(feat_df.values).all(axis=1)
    feat_df, y = feat_df[mask_finite], y[mask_finite]
    feat_df = feat_df.reset_index(drop=True)
    dedup_mask = ~feat_df.duplicated().values
    feat_df, y = feat_df[dedup_mask].reset_index(drop=True), y[dedup_mask]
    log.info("Cleaning: %d -> %d rows after NaN/Inf + duplicate removal", before, len(feat_df))

    feature_names = list(feat_df.columns)
    x = feat_df.values.astype(np.float32)

    # 5. Split. Time-aware (no shuffle) if we had a timestamp, else stratified.
    if ts_col is not None:
        split = int(len(x) * (1 - test_size))
        x_train, x_test = x[:split], x[split:]
        y_train, y_test = y[:split], y[split:]
        log.info("Time-aware split: train=%d test=%d", len(x_train), len(x_test))
    else:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=seed, stratify=y
        )
        log.info("Stratified random split: train=%d test=%d", len(x_train), len(x_test))

    # 6. Scale. Fit on train ONLY, then transform both. Clip test into [0,1]
    #    so unseen extremes do not fall outside the constraint bounds.
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_test = np.clip(scaler.transform(x_test), 0.0, 1.0).astype(np.float32)

    # 7. Optional SMOTE — train only, after scaling. Off by default.
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE

            log.warning("SMOTE enabled. Note: synthetic flows may not satisfy "
                        "feature dependencies; results should be sanity-checked.")
            x_train, y_train = SMOTE(random_state=seed).fit_resample(x_train, y_train)
            x_train = np.clip(x_train, 0.0, 1.0).astype(np.float32)
        except ImportError:
            log.error("imbalanced-learn not installed; skipping SMOTE.")

    return Preprocessed(x_train, x_test, y_train, y_test, feature_names, scaler)
