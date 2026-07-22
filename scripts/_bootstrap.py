"""Put the repo root on sys.path so `import src...` works when a script is
run directly (e.g. `python scripts/01_preprocess.py`)."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
