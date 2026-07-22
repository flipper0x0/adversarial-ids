"""Run the entire study end-to-end: the full pipeline on the primary dataset,
then the full pipeline again on the cross-validation dataset (Stage 9).

This is the one-command path. For step-by-step control and verification gates,
run scripts 01 -> 06 individually, then 07 for cross-dataset.
"""
import _bootstrap  # noqa: F401

from src.config import load_config
from src.pipeline import run_all
from src.utils import set_seed


def main():
    cfg = load_config()
    set_seed(cfg["seed"])

    primary = cfg["dataset"]
    run_all(cfg, primary)

    cross = cfg.get("cross_dataset")
    if cross and cross != primary:
        print(f"\n[cross-dataset] repeating full pipeline on: {cross}")
        run_all(cfg, cross)

    print("\nAll stages complete. Launch the dashboard with:")
    print("    streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
