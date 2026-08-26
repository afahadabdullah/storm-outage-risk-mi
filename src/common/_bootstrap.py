"""Numbered pipeline scripts cannot be imported as modules, so they run as
scripts and import this first to put the repo root on sys.path."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
