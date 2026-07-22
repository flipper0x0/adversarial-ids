"""The realistic-attacker constraint model.

This is the methodological core of the project (proposal section 3.2.3).
A perturbation is only allowed to touch features an attacker can actually
shape at packet-generation time. Everything else is frozen to its original
value.

We go one step beyond the proposal: timing and length features are marked
INCREASE-ONLY, because a real attacker can add delay or pad payload, but
cannot make a flow shorter or a packet smaller than what the attack already
sends. This is more realistic than free perturbation and you should defend
this choice in the viva.

IMPORTANT — read before trusting these lists:
The mutable/immutable split below is a DEFENSIBLE DEFAULT grounded in the
CICFlowMeter / Argus feature semantics and the constraint literature
(Han et al. 2021; Apruzzese et al. 2022). It is NOT ground truth. Which
features are attacker-controllable is itself a research judgement and your
supervisor will (correctly) push on it. Treat these lists as a starting
point you must justify, not as a settled fact.
"""
from dataclasses import dataclass
from typing import List

import numpy as np

# --------------------------------------------------------------------------- #
# CICIDS2017 (CICFlowMeter feature names, after whitespace stripping).
# Attacker controls the FORWARD (client->server) direction: it can add inter-
# arrival delay, pad packet length, and extend flow/active/idle time.
# It does NOT control backward (server) features, protocol-level flag counts,
# the destination port, or defender-computed ratios.
# --------------------------------------------------------------------------- #
CICIDS2017_MUTABLE_INCREASE_ONLY: List[str] = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Length of Fwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Fwd Header Length",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "Subflow Fwd Bytes",
    "Subflow Fwd Packets",
]
CICIDS2017_MUTABLE_FREE: List[str] = []  # none by default; document if you add

# --------------------------------------------------------------------------- #
# UNSW-NB15 (Argus/Bro feature names). Source-side features are attacker-
# controllable; destination-side, protocol/service/state, and ct_* connection
# counts are not.
# --------------------------------------------------------------------------- #
UNSW_NB15_MUTABLE_INCREASE_ONLY: List[str] = [
    "dur",
    "spkts",
    "sbytes",
    "sload",
    "sinpkt",
    "sjit",
    "smean",
    "stcpb",
    "trans_depth",
    "response_body_len",
]
UNSW_NB15_MUTABLE_FREE: List[str] = []

# Synthetic dataset: used for the out-of-the-box demo / smoke test.
SYNTHETIC_MUTABLE_INCREASE_ONLY = [f"f{i}" for i in range(0, 10)]
SYNTHETIC_MUTABLE_FREE = [f"f{i}" for i in range(10, 14)]

_REGISTRY = {
    "cicids2017": (CICIDS2017_MUTABLE_INCREASE_ONLY, CICIDS2017_MUTABLE_FREE),
    "unsw_nb15": (UNSW_NB15_MUTABLE_INCREASE_ONLY, UNSW_NB15_MUTABLE_FREE),
    "synthetic": (SYNTHETIC_MUTABLE_INCREASE_ONLY, SYNTHETIC_MUTABLE_FREE),
}


@dataclass
class ConstraintSpec:
    """Holds the masks/bounds and enforces the constraint set."""

    feature_names: List[str]
    mutable_mask: np.ndarray        # 1.0 where the attacker may perturb
    increase_only_mask: np.ndarray  # 1.0 where value may only increase
    lower: np.ndarray               # per-feature lower bound (post-scaling)
    upper: np.ndarray               # per-feature upper bound (post-scaling)

    @property
    def n_mutable(self) -> int:
        return int(self.mutable_mask.sum())

    def art_mask(self) -> np.ndarray:
        """Feature mask for ART gradient attacks (1 = perturbable)."""
        return self.mutable_mask.astype(np.float32)

    def project(self, x_adv: np.ndarray, x_orig: np.ndarray) -> np.ndarray:
        """Project a candidate adversarial batch back into the legal set.

        Steps, in order:
          1. Reset immutable features to their original value (freeze).
          2. Clip every feature into [lower, upper] (valid range).
          3. Enforce increase-only: marked features cannot drop below original.

        After this, x is something a constrained attacker could actually send.
        """
        x = np.array(x_adv, dtype=np.float32, copy=True)
        x0 = np.asarray(x_orig, dtype=np.float32)

        immutable = self.mutable_mask == 0.0
        x[:, immutable] = x0[:, immutable]

        x = np.clip(x, self.lower, self.upper)

        inc = self.increase_only_mask == 1.0
        if inc.any():
            x[:, inc] = np.maximum(x[:, inc], x0[:, inc])
        return x

    def is_valid(self, x_adv: np.ndarray, x_orig: np.ndarray, atol: float = 1e-5,
                 exempt_idx=None) -> np.ndarray:
        """Per-sample boolean: does this perturbation respect the constraints?

        Used by the integrity check (FR-5 / NFR-2): non-compliant samples are
        logged and discarded rather than silently scored.

        `exempt_idx`: column indices to skip in the freeze check — used for
        derived features under a consistency-preserving attack, which are
        legitimately recomputed (not frozen) from perturbed base features.
        """
        x = np.asarray(x_adv, dtype=np.float32)
        x0 = np.asarray(x_orig, dtype=np.float32)

        immutable = self.mutable_mask == 0.0
        if exempt_idx:
            immutable = immutable.copy()
            for i in exempt_idx:
                immutable[i] = False
        froze = np.all(np.abs(x[:, immutable] - x0[:, immutable]) <= atol, axis=1)

        in_range = np.all((x >= self.lower - atol) & (x <= self.upper + atol), axis=1)

        inc = self.increase_only_mask == 1.0
        if inc.any():
            increased = np.all(x[:, inc] >= x0[:, inc] - atol, axis=1)
        else:
            increased = np.ones(len(x), dtype=bool)

        return froze & in_range & increased


def build_constraint_spec(
    feature_names: List[str],
    dataset: str,
    lower: float = 0.0,
    upper: float = 1.0,
) -> ConstraintSpec:
    """Build a ConstraintSpec aligned to the actual post-preprocessing columns.

    Names in the registry that are not present (e.g. dropped during feature
    selection) are simply skipped, so this is robust to column changes.
    Bounds default to [0, 1] because preprocessing scales features with
    MinMax — that makes L-inf / L-2 budgets directly interpretable.
    """
    key = dataset.lower()
    if key not in _REGISTRY:
        raise ValueError(f"No constraint map for dataset '{dataset}'. Known: {list(_REGISTRY)}")

    inc_names, free_names = _REGISTRY[key]
    inc_names = set(inc_names)
    free_names = set(free_names)

    n = len(feature_names)
    mutable = np.zeros(n, dtype=np.float32)
    increase_only = np.zeros(n, dtype=np.float32)

    for i, name in enumerate(feature_names):
        if name in inc_names:
            mutable[i] = 1.0
            increase_only[i] = 1.0
        elif name in free_names:
            mutable[i] = 1.0

    low = np.full(n, lower, dtype=np.float32)
    up = np.full(n, upper, dtype=np.float32)

    return ConstraintSpec(
        feature_names=list(feature_names),
        mutable_mask=mutable,
        increase_only_mask=increase_only,
        lower=low,
        upper=up,
    )
