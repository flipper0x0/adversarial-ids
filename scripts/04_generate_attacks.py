"""Stage 6: generate constrained adversarial examples (the BEFORE result).
White-box FGSM/PGD/JSMA on the MLP; black-box / transfer on RF, XGBoost, SVM.
All perturbations are projected onto the mutable-feature constraint set."""
import _bootstrap  # noqa: F401

from src.config import load_config
from src.pipeline import stage_attacks
from src.utils import set_seed


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    stage_attacks(cfg, cfg["dataset"])


if __name__ == "__main__":
    main()
