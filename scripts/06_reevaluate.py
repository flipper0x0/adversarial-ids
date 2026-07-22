"""Stage 8: re-run the attacks against the hardened models (the AFTER result),
including UNSEEN attacks, and record recovery + clean-accuracy trade-off."""
import _bootstrap  # noqa: F401

from src.config import load_config
from src.pipeline import stage_reeval
from src.utils import set_seed


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    stage_reeval(cfg, cfg["dataset"])


if __name__ == "__main__":
    main()
