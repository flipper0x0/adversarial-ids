"""Adversarial attack generation (proposal sections 3.2.5).

Every attack output is forced into the legal set via ConstraintSpec.project:
only mutable features move, values stay in range, increase-only features
cannot drop. For gradient attacks we ALSO pass ART a feature mask so the
perturbation is constrained during the search, not just clipped after.

How the constraint is enforced per attack family:
  * FGSM / PGD : ART `mask` during search  +  projection after.  (clean)
  * JSMA       : projection after (ART's saliency attack takes no mask).
  * HopSkipJump / ZOO (black-box) : projection after generation, then the
    caller RE-MEASURES success on the projected sample. This is honest, not
    constraint-aware search — projecting a black-box result can destroy its
    adversarial property, which correctly shows that constraints make the
    attacker's job harder. Document this as a limitation.
  * Transfer   : reuse the (already-projected) MLP adversarials against the
    tree models — naturally constraint-respecting.

All attacks are untargeted-in-spirit (push malicious -> benign). JSMA is
targeted at the benign class because ART's saliency method requires a target.
"""
from typing import Dict

import numpy as np

from .constraints import ConstraintSpec
from .utils import get_logger

log = get_logger("attacks")

BENIGN = 0
MALICIOUS = 1


def generate_whitebox(art_clf, x: np.ndarray, spec: ConstraintSpec, cfg,
                       dep=None, scaler=None) -> Dict[str, np.ndarray]:
    """Constrained FGSM, PGD, JSMA on a differentiable classifier.

    `dep` / `scaler` are optional. When both are given (a DependencySpec from
    `.dependencies` and the fitted preprocessing scaler), every derived
    feature is recomputed from its base features after projection — this is
    what makes an attack "consistency-preserving" rather than "naive" (see
    `.dependencies` module docstring). Pass a `spec` whose mutable_mask
    already excludes derived features (`build_consistent_constraint_spec`)
    when using this; otherwise the recompute silently overwrites whatever the
    optimizer put in the derived columns.
    """
    from art.attacks.evasion import (
        FastGradientMethod,
        ProjectedGradientDescent,
        SaliencyMapMethod,
    )

    a = cfg["attacks"]
    mask = spec.art_mask()
    out: Dict[str, np.ndarray] = {}

    # FGSM — single-step, weak baseline.
    fgsm = FastGradientMethod(estimator=art_clf, eps=a["fgsm"]["eps"], norm=np.inf)
    x_fgsm = fgsm.generate(x=x, mask=mask)
    out["FGSM"] = spec.project(x_fgsm, x)

    # PGD — iterative, strong worst-case white-box bound.
    pgd = ProjectedGradientDescent(
        estimator=art_clf,
        eps=a["pgd"]["eps"],
        eps_step=a["pgd"]["eps_step"],
        max_iter=a["pgd"]["max_iter"],
        norm=np.inf,
        verbose=False,
    )
    x_pgd = pgd.generate(x=x, mask=mask)
    out["PGD"] = spec.project(x_pgd, x)

    # JSMA — sparse, few-feature; matches "only a few features controllable".
    from art.utils import to_categorical

    jsma = SaliencyMapMethod(
        classifier=art_clf,
        theta=a["jsma"]["theta"],
        gamma=a["jsma"]["gamma"],
        verbose=False,
    )
    target = to_categorical(np.full(len(x), BENIGN), nb_classes=2)
    x_jsma = jsma.generate(x=x, y=target)
    out["JSMA"] = spec.project(x_jsma, x)

    if dep is not None and dep.has_dependencies:
        out = {name: dep.recompute(x_adv, scaler) for name, x_adv in out.items()}

    _log_validity(out, x, spec, dep=dep)
    return out


def generate_blackbox(art_clf, x: np.ndarray, spec: ConstraintSpec, cfg,
                       dep=None, scaler=None) -> Dict[str, np.ndarray]:
    """Gradient-free attacks on the deployed model (query/decision based).

    Subsampled because these are expensive — ZOO especially. Sizes in config.
    See `generate_whitebox` for what `dep`/`scaler` do.
    """
    from art.attacks.evasion import HopSkipJump, ZooAttack

    a = cfg["attacks"]["blackbox"]
    n = min(a["n_samples"], len(x))
    xs = x[:n]
    out: Dict[str, np.ndarray] = {}

    log.info("HopSkipJump on %d samples ...", n)
    hsj = HopSkipJump(
        classifier=art_clf,
        targeted=False,
        norm=2,
        max_iter=a["hopskipjump"]["max_iter"],
        max_eval=a["hopskipjump"]["max_eval"],
        init_eval=a["hopskipjump"]["init_eval"],
        verbose=False,
    )
    out["HopSkipJump"] = spec.project(hsj.generate(x=xs), xs)

    if a["zoo"]["enabled"]:
        log.info("ZOO on %d samples (slow) ...", n)
        zoo = ZooAttack(
            classifier=art_clf,
            max_iter=a["zoo"]["max_iter"],
            nb_parallel=a["zoo"]["nb_parallel"],
            binary_search_steps=a["zoo"]["binary_search_steps"],
            verbose=False,
        )
        out["ZOO"] = spec.project(zoo.generate(x=xs), xs)

    if dep is not None and dep.has_dependencies:
        out = {name: dep.recompute(x_adv, scaler) for name, x_adv in out.items()}

    _log_validity(out, xs, spec, dep=dep)
    return out


def transfer(adv_from_surrogate: np.ndarray) -> np.ndarray:
    """Transfer attack = reuse surrogate adversarials against the real model.

    Already projected on the surrogate, so still constraint-valid. The caller
    feeds these straight into the tree model's predict and measures evasion.
    """
    return adv_from_surrogate


def _log_validity(adv_dict, x, spec, dep=None):
    exempt = set(dep.derived_idx) if dep is not None and dep.has_dependencies else None
    for name, x_adv in adv_dict.items():
        valid = spec.is_valid(x_adv, x, exempt_idx=exempt)
        if not valid.all():
            log.warning("%s: %d/%d samples violated constraints post-projection "
                        "(should be ~0).", name, int((~valid).sum()), len(valid))
