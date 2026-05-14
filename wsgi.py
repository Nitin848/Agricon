"""
WSGI entry when the Git repository root is one level above the app folder.

Gunicorn cannot load ``Agricon-main.app:app`` — ``Agricon-main`` is not a valid
Python module name (hyphens). Use ``gunicorn wsgi:app`` from this directory
instead, or set Render **Root Directory** to ``Agricon-main`` and use
``gunicorn app:app``.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
_APP = os.path.join(_ROOT, "Agricon-main")
if not os.path.isfile(os.path.join(_APP, "app.py")):
    raise RuntimeError(
        f"Expected Flask app at {_APP}/app.py. Clone layout should keep code in Agricon-main/."
    )
sys.path.insert(0, _APP)

from app import app  # noqa: E402
