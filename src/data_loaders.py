"""Raw dataset loaders.

Each loader returns a pandas DataFrame with a binary 'Label' column
(0 = benign/normal, 1 = malicious/attack) plus, where available, an
'attack_cat' column for reference (not used as a feature).

You must download the real datasets yourself (see data/README.md) and point
config.yaml at them. The 'synthetic' loader needs no download and lets you
run the whole pipeline immediately to verify your install.
"""
import glob
import os
from typing import Optional

import numpy as np
import pandas as pd

from .utils import get_logger

log = get_logger("data_loaders")

# Columns that are identifiers or leak the label and must never be features.
_CICIDS_DROP = [
    "Flow ID", "Source IP", "Src IP", "Destination IP", "Dst IP",
    "Source Port", "Src Port", "Fwd Header Length.1",
]
_CICIDS_TIMESTAMP = ["Timestamp"]


def load_synthetic(n_samples: int = 20000, n_features: int = 30, seed: int = 42) -> pd.DataFrame:
    """Class-imbalanced synthetic tabular data shaped like network flows.

    Not a substitute for real data — only for plumbing/install verification
    and the demo dashboard.
    """
    from sklearn.datasets import make_classification

    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=18,
        n_redundant=6,
        n_classes=2,
        weights=[0.85, 0.15],   # mimic IDS imbalance: mostly benign
        class_sep=1.2,
        random_state=seed,
    )
    cols = [f"f{i}" for i in range(n_features)]
    df = pd.DataFrame(x, columns=cols)
    df["Label"] = y.astype(int)
    return df


def load_cicids2017(path: str) -> pd.DataFrame:
    """Load CICIDS2017 from a directory of CSVs (or a single CSV).

    Handles the well-known quirks: leading/trailing spaces in column names,
    Infinity values in the rate columns, and the duplicate header column.
    Use the corrected/improved release (Engelen et al. 2021) if you have it.
    """
    files = _csv_files(path)
    log.info("CICIDS2017: reading %d file(s)", len(files))
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)

    df.columns = [c.strip() for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan)

    label_col = _find_col(df, ["Label"])
    if label_col is None:
        raise ValueError("CICIDS2017: no 'Label' column found after stripping names.")

    df["attack_cat"] = df[label_col].astype(str).str.strip()
    df["Label"] = (df["attack_cat"].str.upper() != "BENIGN").astype(int)
    if label_col != "Label":
        df = df.drop(columns=[label_col])

    df = _drop_if_present(df, _CICIDS_DROP)
    return df


def load_unsw_nb15(path: str) -> pd.DataFrame:
    """Load UNSW-NB15 from CSV(s).

    Expects the standard training/testing CSVs that already carry named
    columns including 'label' (binary) and 'attack_cat'. Categorical columns
    (proto, service, state) are returned as-is; preprocessing encodes them.
    """
    files = _csv_files(path)
    log.info("UNSW-NB15: reading %d file(s)", len(files))
    frames = [pd.read_csv(f, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df.columns = [c.strip() for c in df.columns]

    label_col = _find_col(df, ["label", "Label"])
    if label_col is None:
        raise ValueError("UNSW-NB15: no 'label' column found.")
    df["Label"] = df[label_col].astype(int)
    if label_col != "Label":
        df = df.drop(columns=[label_col])

    if "attack_cat" not in df.columns:
        df["attack_cat"] = np.where(df["Label"] == 1, "attack", "normal")

    df = _drop_if_present(df, ["id", "srcip", "dstip", "sport", "dsport", "stime", "ltime"])
    return df


def load_dataset(name: str, path: Optional[str], seed: int = 42) -> pd.DataFrame:
    name = name.lower()
    if name == "synthetic":
        return load_synthetic(seed=seed)
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset '{name}' path does not exist: {path!r}. "
            f"Download it (see data/README.md) and set paths in config.yaml, "
            f"or use dataset: synthetic to test the pipeline."
        )
    if name == "cicids2017":
        return load_cicids2017(path)
    if name == "unsw_nb15":
        return load_unsw_nb15(path)
    raise ValueError(f"Unknown dataset '{name}'.")


# ----------------------------- helpers ------------------------------------- #
def _csv_files(path: str):
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files in directory {path}")
        return files
    return [path]


def _find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _drop_if_present(df: pd.DataFrame, cols):
    present = [c for c in cols if c in df.columns]
    if present:
        log.info("Dropping identifier/leaky columns: %s", present)
    return df.drop(columns=present)
