"""Stage 1-2: load the configured dataset, clean/encode/scale, split, and
cache the processed arrays under results/<dataset>/processed/."""
import _bootstrap  # noqa: F401  (sets sys.path)

from src.config import load_config
from src.pipeline import stage_preprocess
from src.utils import set_seed


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    stage_preprocess(cfg, cfg["dataset"])


if __name__ == "__main__":
    main()
