"""Global paths & reproducibility settings."""
from pathlib import Path

# Root directory two levels up from this file
ROOT = Path(__file__).resolve().parent.parent

# I/O folders
DATA_DIR = ROOT / "data"
GRAPH_DIR = ROOT / "graphs"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Misc.
SEED = 42