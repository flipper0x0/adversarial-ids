"""Stage 4: train the baseline detectors (Random Forest, XGBoost, SVM, MLP)
on the processed training data and save them."""
import _bootstrap  # noqa: F401

from src.config import load_config
from src.pipeline import stage_train
from src.utils import set_seed


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    stage_train(cfg, cfg["dataset"])


if __name__ == "__main__":
    main()
