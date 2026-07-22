"""Central configuration loader.

Everything tunable lives in config.yaml so runs are reproducible (NFR-1).
Nothing in the code hard-codes a hyperparameter that the config also sets.
"""
import os
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "config.yaml")


class Config(dict):
    """Dict that also supports attribute access (cfg.seed as well as cfg['seed'])."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        if isinstance(value, dict):
            return Config(value)
        return value


def load_config(path: str = None) -> Config:
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "r") as handle:
        raw = yaml.safe_load(handle)
    cfg = Config(raw)
    cfg["repo_root"] = REPO_ROOT
    return cfg


def results_dir(cfg: Config, dataset: str) -> str:
    """Per-dataset output directory; created on demand."""
    out = os.path.join(REPO_ROOT, cfg["paths"]["results"], dataset)
    os.makedirs(out, exist_ok=True)
    return out


def processed_dir(cfg: Config, dataset: str) -> str:
    out = os.path.join(REPO_ROOT, cfg["paths"]["processed"], dataset)
    os.makedirs(out, exist_ok=True)
    return out
