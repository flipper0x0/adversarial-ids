"""End-to-end smoke test on synthetic data.

This is NOT a robustness experiment. It verifies that every component imports,
trains, attacks, defends, and produces metrics without error — the plumbing.
Run:  python tests/smoke_test.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np

from src.config import load_config
from src.constraints import build_constraint_spec
from src.dependencies import build_consistent_constraint_spec, build_dependency_spec
from src import metrics as M
from src.pipeline import (
    stage_attacks,
    stage_clean_eval,
    stage_defenses,
    stage_preprocess,
    stage_reeval,
    stage_train,
)
from src.utils import load_json


def main():
    cfg = load_config()
    # Make the smoke test fast and cheap.
    cfg["attacks"]["n_eval_samples"] = 200
    cfg["attacks"]["blackbox"]["n_samples"] = 40
    cfg["attacks"]["blackbox"]["hopskipjump"]["max_iter"] = 10
    cfg["attacks"]["blackbox"]["hopskipjump"]["max_eval"] = 200
    cfg["attacks"]["blackbox"]["hopskipjump"]["init_eval"] = 20
    cfg["attacks"]["pgd"]["max_iter"] = 10
    cfg["models"]["mlp"]["epochs"] = 5
    cfg["models"]["svm"]["train_subsample"] = 4000
    cfg["defenses"]["adv_train"]["n_adv_samples"] = 500

    dataset = "synthetic"

    print("\n[1/7] metric unit checks ...")
    _check_metrics()

    print("\n[2/7] constraint projection check ...")
    _check_constraints()

    print("\n[3/7] dependency / naive-vs-consistent check ...")
    _check_dependencies()

    print("\n[4/7] preprocess + train ...")
    stage_preprocess(cfg, dataset)
    stage_train(cfg, dataset)

    print("\n[5/7] clean eval ...")
    stage_clean_eval(cfg, dataset)

    print("\n[6/7] attacks (before, naive + consistent) ...")
    stage_attacks(cfg, dataset)

    print("\n[7/7] defenses + reeval (after, naive + consistent sanitization) ...")
    stage_defenses(cfg, dataset)
    stage_reeval(cfg, dataset)

    from src.config import results_dir
    summary = load_json(os.path.join(results_dir(cfg, dataset), "summary.json"))
    assert "clean" in summary and "attacks_before" in summary and "after" in summary
    assert "attacks_before_consistent" in summary, \
        "naive-vs-consistent contrast: consistent attack stage did not run"
    assert "sanitization_MLP_naive" in summary["after"] and \
        "sanitization_MLP_consistent" in summary["after"], \
        "naive-vs-consistent contrast: sanitization comparison did not run"
    print("\nSMOKE TEST PASSED. Summary keys:", list(summary.keys()))
    print("Clean accuracies:", {k: round(v["accuracy"], 3) for k, v in summary["clean"].items()})
    print("MLP PGD ASR (before, naive):", round(summary["attacks_before"]["MLP"]["PGD"]["asr"], 3))
    print("MLP PGD ASR (before, consistent):",
          round(summary["attacks_before_consistent"]["MLP"]["PGD"]["asr"], 3))
    print("MLP PGD DRR (after):", summary["after"]["adv_training_MLP"]["PGD"]["drr"])
    print("MLP PGD ASR after sanitize — naive:",
          round(summary["after"]["sanitization_MLP_naive"]["PGD"]["asr"], 3),
          " consistent:",
          round(summary["after"]["sanitization_MLP_consistent"]["PGD"]["asr"], 3))


def _check_metrics():
    y_true = np.array([1, 1, 1, 1, 0, 0])
    pred_before = np.array([1, 1, 1, 0, 0, 0])  # 3 of 4 malicious detected
    pred_after = np.array([0, 0, 1, 0, 0, 0])   # 2 of those 3 now evade
    asr = M.attack_success_rate(y_true, pred_before, pred_after)
    assert abs(asr - 2 / 3) < 1e-9, asr
    assert abs(M.robustness_score(0.8, 0.4) - 0.5) < 1e-9
    drr = M.defense_recovery_rate(acc_clean=1.0, acc_undef_attacked=0.2, acc_def_attacked=0.6)
    assert abs(drr - 0.5) < 1e-9, drr
    print("    metrics OK (ASR=2/3, R=0.5, DRR=0.5)")


def _check_constraints():
    names = [f"f{i}" for i in range(30)]
    spec = build_constraint_spec(names, "synthetic")
    x0 = np.full((5, 30), 0.5, dtype=np.float32)
    x_adv = x0 + 0.3  # push everything up
    x_adv[:, 0] -= 0.6  # try to DECREASE an increase-only feature
    proj = spec.project(x_adv, x0)
    immutable = spec.mutable_mask == 0
    assert np.allclose(proj[:, immutable], x0[:, immutable]), "immutable not frozen"
    inc = spec.increase_only_mask == 1
    assert np.all(proj[:, inc] >= x0[:, inc] - 1e-6), "increase-only violated"
    assert proj.max() <= 1.0 + 1e-6 and proj.min() >= 0.0 - 1e-6, "out of range"
    assert spec.is_valid(proj, x0).all()
    print(f"    constraints OK ({spec.n_mutable}/{len(names)} mutable, freeze+range+increase enforced)")


def _check_dependencies():
    from sklearn.preprocessing import MinMaxScaler

    names = [f"f{i}" for i in range(30)]
    dep = build_dependency_spec(names, "synthetic")
    assert dep.has_dependencies, "synthetic dataset should define derived features"

    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 10, size=(20, 30))
    scaler = MinMaxScaler().fit(raw)
    x_scaled = scaler.transform(raw).astype(np.float32)

    consistent = dep.recompute(x_scaled, scaler)
    raw_back = scaler.inverse_transform(consistent)

    base_idx = [i for i in range(30) if i not in dep.derived_idx]
    assert np.allclose(raw_back[:, base_idx], raw[:, base_idx], atol=1e-3), \
        "recompute must leave base features untouched"

    for d_idx, b_idxs, formula in zip(dep.derived_idx, dep.base_idx, dep.formulas):
        expected = formula(*[raw_back[:, i] for i in b_idxs])
        assert np.allclose(raw_back[:, d_idx], expected, atol=1e-3), \
            "recomputed derived feature must satisfy its own formula"

    consistent_spec, _ = build_consistent_constraint_spec(names, "synthetic")
    naive_spec = build_constraint_spec(names, "synthetic")
    assert consistent_spec.n_mutable < naive_spec.n_mutable, (
        "the consistent spec must perturb strictly fewer (base-only) features "
        "than the naive spec, or there is no contrast to measure"
    )
    print(f"    dependencies OK ({len(dep.derived_idx)} derived features recomputed from base; "
          f"consistent spec mutable={consistent_spec.n_mutable} < naive mutable={naive_spec.n_mutable})")


if __name__ == "__main__":
    main()
