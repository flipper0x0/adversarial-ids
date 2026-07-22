"""Stage 7: harden the models with adversarial training and build the input
sanitization (constraint projection + feature squeezing) defense."""
import _bootstrap  # noqa: F401

from src.config import load_config
from src.pipeline import stage_defenses
from src.utils import set_seed


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    stage_defenses(cfg, cfg["dataset"])


if __name__ == "__main__":
    main()
