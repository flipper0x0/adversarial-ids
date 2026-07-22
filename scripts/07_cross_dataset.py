"""Stage 9: repeat the FULL pipeline on the secondary dataset (retrain from
scratch, not transfer) to check the findings are not tied to one dataset.

The two datasets have non-overlapping feature spaces, so this re-runs the
whole pipeline on cross_dataset rather than transferring attacks across.
"""
import _bootstrap  # noqa: F401

from src.config import load_config
from src.pipeline import run_all
from src.utils import set_seed


def main():
    cfg = load_config()
    set_seed(cfg["seed"])
    cross = cfg["cross_dataset"]
    print(f"[cross-dataset] running full pipeline on secondary dataset: {cross}")
    run_all(cfg, cross)


if __name__ == "__main__":
    main()
