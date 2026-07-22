"""Shared utilities: seeding, logging, IO."""
import json
import logging
import os
import random

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    """Seed every RNG we touch. This is what makes NFR-1 (reproducibility) real.

    Note: full bit-for-bit determinism on GPU is not guaranteed even with this;
    on CPU it is effectively deterministic for our models.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # torch optional at import time
        pass


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, default=_json_default)


def load_json(path: str):
    with open(path, "r") as handle:
        return json.load(handle)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
