"""The feature-dependency model — mechanism behind the naive-vs-consistent
contrast (proposal sections 3.2.5 / 3.2.6, "the core finding, not a side
experiment").

Some mutable features in `constraints.py` are not independent: they are
statistics *computed from* other mutable features by the flow exporter
(CICFlowMeter / Argus). E.g. `Fwd Packet Length Mean` is not something an
attacker sets directly — it falls out of how many forward packets were sent
and how many bytes they carried. This module marks each dataset's mutable
set into:

  * BASE features    — the attacker's actual primitives (packet/byte counts,
                        duration, timing totals). Perturbing these is what a
                        real packet-crafting attacker does.
  * DERIVED features  — statistics recomputed from base features. A "naive"
                        feature-space attack perturbs these directly (easy
                        for gradient descent, but physically meaningless: the
                        sample now claims a mean that doesn't match its own
                        total/count). A "consistency-preserving" attack never
                        touches them directly — it only perturbs base
                        features and recomputes derived ones afterward.

The formulas are DEFENSIBLE SIMPLIFICATIONS of the real CICFlowMeter/Argus
computations (e.g. "Subflow Fwd Bytes" is treated as equal to "Total Length
of Fwd Packets" under a single-subflow assumption), not a byte-for-byte
reimplementation of those tools. Same status as constraints.py's mutable
lists: a defensible default you must be ready to justify, not ground truth.

Formulas operate on RAW (pre-scaling) values because ratios like mean =
total / count do not survive independent per-feature MinMax scaling —
`recompute()` unscales, applies the formula, and rescales.
"""
from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

# --------------------------------------------------------------------------- #
# CICIDS2017: base = totals/counts/duration an attacker chooses directly.
# derived = per-packet/per-interval statistics computed from those totals.
# --------------------------------------------------------------------------- #
_HEADER_BYTES_PER_PACKET = 20.0  # typical TCP/IP header; approximation


def _fwd_pkt_len_mean(total_len_fwd, total_fwd_pkts):
    return total_len_fwd / np.maximum(total_fwd_pkts, 1.0)


def _fwd_iat_mean(fwd_iat_total, total_fwd_pkts):
    return fwd_iat_total / np.maximum(total_fwd_pkts - 1.0, 1.0)


def _flow_iat_mean(flow_duration, total_fwd_pkts):
    return flow_duration / np.maximum(total_fwd_pkts - 1.0, 1.0)


def _fwd_header_length(total_fwd_pkts):
    return total_fwd_pkts * _HEADER_BYTES_PER_PACKET


def _identity(x):
    return x


@dataclass
class DerivedFeature:
    """One derived feature and the formula that recomputes it from base cols."""

    name: str
    bases: List[str]
    raw_formula: Callable[..., np.ndarray]


CICIDS2017_DERIVED: List[DerivedFeature] = [
    DerivedFeature("Fwd Packet Length Mean",
                   ["Total Length of Fwd Packets", "Total Fwd Packets"], _fwd_pkt_len_mean),
    DerivedFeature("Fwd IAT Mean", ["Fwd IAT Total", "Total Fwd Packets"], _fwd_iat_mean),
    DerivedFeature("Flow IAT Mean", ["Flow Duration", "Total Fwd Packets"], _flow_iat_mean),
    DerivedFeature("Fwd Header Length", ["Total Fwd Packets"], _fwd_header_length),
    # Single-subflow approximation: subflow totals equal flow totals.
    DerivedFeature("Subflow Fwd Bytes", ["Total Length of Fwd Packets"], _identity),
    DerivedFeature("Subflow Fwd Packets", ["Total Fwd Packets"], _identity),
]

# --------------------------------------------------------------------------- #
# UNSW-NB15 (Argus): base = duration/packet-count/byte-count/tcp-seq/depth.
# derived = mean packet size, load (bits/sec), inter-packet time.
# --------------------------------------------------------------------------- #


def _smean(sbytes, spkts):
    return sbytes / np.maximum(spkts, 1.0)


def _sload(sbytes, dur):
    return sbytes * 8.0 / np.maximum(dur, 1e-6)


def _sinpkt(dur, spkts):
    return (dur * 1000.0) / np.maximum(spkts - 1.0, 1.0)


UNSW_NB15_DERIVED: List[DerivedFeature] = [
    DerivedFeature("smean", ["sbytes", "spkts"], _smean),
    DerivedFeature("sload", ["sbytes", "dur"], _sload),
    DerivedFeature("sinpkt", ["dur", "spkts"], _sinpkt),
]

# --------------------------------------------------------------------------- #
# Synthetic: no real semantics, but wires the same base->derived plumbing so
# the smoke test and demo exercise the naive-vs-consistent contrast end-to-end.
# --------------------------------------------------------------------------- #
SYNTHETIC_DERIVED: List[DerivedFeature] = [
    DerivedFeature("f7", ["f0", "f1", "f2"], lambda f0, f1, f2: (f0 + f1 + f2) / 3.0),
    DerivedFeature("f8", ["f3", "f4"], lambda f3, f4: f3 + f4),
    DerivedFeature("f9", ["f5", "f6"], lambda f5, f6: f5 / np.maximum(np.abs(f6), 1e-3)),
]

_REGISTRY = {
    "cicids2017": CICIDS2017_DERIVED,
    "unsw_nb15": UNSW_NB15_DERIVED,
    "synthetic": SYNTHETIC_DERIVED,
}


@dataclass
class DependencySpec:
    """Holds, for the columns actually present, which are derived and how to
    recompute them from their base columns."""

    feature_names: List[str]
    derived_idx: List[int] = field(default_factory=list)
    derived_names: List[str] = field(default_factory=list)
    base_idx: List[List[int]] = field(default_factory=list)
    formulas: List[Callable] = field(default_factory=list)

    @property
    def has_dependencies(self) -> bool:
        return len(self.derived_idx) > 0

    def recompute(self, x_scaled: np.ndarray, scaler) -> np.ndarray:
        """Overwrite every derived column with formula(base columns), in raw
        space, then rescale. A no-op if this dataset has no known dependencies
        or none of them survived preprocessing (columns dropped etc.)."""
        if not self.has_dependencies:
            return np.asarray(x_scaled, dtype=np.float32)

        raw = scaler.inverse_transform(np.asarray(x_scaled, dtype=np.float64))
        for d_idx, b_idxs, formula in zip(self.derived_idx, self.base_idx, self.formulas):
            base_cols = [raw[:, i] for i in b_idxs]
            raw[:, d_idx] = formula(*base_cols)

        rescaled = scaler.transform(raw)
        return np.clip(rescaled, 0.0, 1.0).astype(np.float32)


def build_dependency_spec(feature_names: List[str], dataset: str) -> DependencySpec:
    """Build a DependencySpec aligned to the actual post-preprocessing columns.

    Like build_constraint_spec, entries whose name (or whose bases) are not
    present in feature_names are silently skipped — robust to column changes.
    """
    key = dataset.lower()
    if key not in _REGISTRY:
        raise ValueError(f"No dependency map for dataset '{dataset}'. Known: {list(_REGISTRY)}")

    name_to_idx = {name: i for i, name in enumerate(feature_names)}
    spec = DependencySpec(feature_names=list(feature_names))

    for d in _REGISTRY[key]:
        if d.name not in name_to_idx:
            continue
        if not all(b in name_to_idx for b in d.bases):
            continue
        spec.derived_idx.append(name_to_idx[d.name])
        spec.derived_names.append(d.name)
        spec.base_idx.append([name_to_idx[b] for b in d.bases])
        spec.formulas.append(d.raw_formula)

    return spec


def build_consistent_constraint_spec(feature_names: List[str], dataset: str):
    """Build the restricted ConstraintSpec used by consistency-preserving
    attacks: identical to the naive spec (constraints.build_constraint_spec)
    except every derived feature is removed from the mutable set, since a
    consistent attacker only manipulates base features directly and gets
    derived ones "for free" via DependencySpec.recompute().

    Returns (consistent_spec, dependency_spec).
    """
    from .constraints import ConstraintSpec, build_constraint_spec

    naive_spec = build_constraint_spec(feature_names, dataset)
    dep = build_dependency_spec(feature_names, dataset)

    base_mask = naive_spec.mutable_mask.copy()
    for idx in dep.derived_idx:
        base_mask[idx] = 0.0

    consistent_spec = ConstraintSpec(
        feature_names=naive_spec.feature_names,
        mutable_mask=base_mask,
        increase_only_mask=naive_spec.increase_only_mask.copy(),
        lower=naive_spec.lower,
        upper=naive_spec.upper,
    )
    return consistent_spec, dep
