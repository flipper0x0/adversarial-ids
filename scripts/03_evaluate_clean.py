"""Stage 5: evaluate every trained model on clean (unattacked) test data.
Writes clean metrics into results/<dataset>/summary.json."""
import _bootstrap  # noqa: F401

from src.config import load_config
from src.pipeline import stage_clean_eval
from src.utils import set_seed


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    stage_clean_eval(cfg, cfg["dataset"])


if __name__ == "__main__":
    main()
