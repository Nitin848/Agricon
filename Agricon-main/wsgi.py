import os
import sys

# Render/Gunicorn needs an importable module path. The project code lives in a
# directory named "Agricon-main" (hyphen), which can't be imported as a Python
# package. Add it to sys.path so `from app import app` works.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_DIR = os.path.join(BASE_DIR, "Agricon-main")
sys.path.insert(0, APP_DIR)

from app import app  # noqa: E402

