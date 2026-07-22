"""Defenses (proposal section 3.2.6).

1. Adversarial training: retrain the MLP on clean + constrained-adversarial
   samples. We craft the adversarial training set with PGD (the strongest
   white-box attack) so the model learns a margin against the worst case.

2. Input sanitization at inference, two parts:
   - constraint projection: clip incoming vectors into valid ranges. (Note:
     at inference there is no "original" to freeze against, so this is range
     clipping + feature squeezing, NOT the freeze step used during attack
     generation.)
   - feature squeezing (Xu et al.): reduce feature precision to b bits, which
     collapses small adversarial perturbations onto the same quantised value
     as the clean input.

Defended models are evaluated against attacks NOT used to build the defense
(handled in the pipeline) — otherwise you measure memorisation, not robustness.
"""
import numpy as np

from .attacks import generate_whitebox
from .constraints import ConstraintSpec
from .models_neural import train_mlp
from .utils import get_logger

log = get_logger("defenses")


def adversarial_train(x_train, y_train, art_surrogate, spec: ConstraintSpec, cfg):
    """Return a hardened MLP trained on clean + adversarial samples.

    Only malicious samples are perturbed (an attacker has no reason to perturb
    benign traffic). Their adversarial versions keep the malicious label so the
    model learns to still flag them.
    """
    mal_idx = np.where(y_train == 1)[0]
    cap = cfg["defenses"]["adv_train"]["n_adv_samples"]
    if len(mal_idx) > cap:
        rng = np.random.default_rng(cfg["seed"])
        mal_idx = rng.choice(mal_idx, size=cap, replace=False)

    x_mal = x_train[mal_idx]
    log.info("Adversarial training: crafting %d adversarial malicious samples (PGD).", len(x_mal))
    advs = generate_whitebox(art_surrogate, x_mal, spec, cfg)
    x_adv = advs["PGD"]
    y_adv = np.ones(len(x_adv), dtype=int)

    log.info("Retraining MLP on clean + adversarial set.")
    return train_mlp(x_train, y_train, cfg, x_extra=x_adv, y_extra=y_adv)


def feature_squeeze(x: np.ndarray, bits: int) -> np.ndarray:
    """Quantise each feature to `bits` bits over the [0,1] range."""
    levels = float(2 ** bits - 1)
    return np.round(np.clip(x, 0.0, 1.0) * levels) / levels


def sanitize(x: np.ndarray, spec: ConstraintSpec, cfg, dep=None, scaler=None) -> np.ndarray:
    """Inference-time input sanitization: range clip + feature squeezing +
    (when `dep`/`scaler` are given) recomputing derived features from base
    features.

    That third step is the mechanism behind the naive-vs-consistent contrast:
    it overwrites whatever a naive attack put in a derived column (e.g. a
    mean it perturbed independently of the total/count it should equal) with
    the value consistent with that sample's base features — destroying the
    naive attack's perturbation there. A consistency-preserving attack never
    touched the derived columns directly, so this step is a no-op on it and
    the attack survives. The sanitizer applies the same steps either way; it
    has no notion of which kind of attack it is looking at.
    """
    x = np.clip(x, spec.lower, spec.upper)
    bits = cfg["defenses"]["sanitization"]["squeeze_bits"]
    x = feature_squeeze(x, bits).astype(np.float32)
    if dep is not None and dep.has_dependencies:
        x = dep.recompute(x, scaler)
    return x
