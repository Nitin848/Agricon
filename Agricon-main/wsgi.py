import os
import sys

# Same directory as app.py — add this folder only (no nested Agricon-main path).
_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _BASE)

from app import app  # noqa: E402
