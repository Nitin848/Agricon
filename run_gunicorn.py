"""
Production entry for Render (and similar hosts).

Use as the start command:  python run_gunicorn.py

Never use:  gunicorn Agricon-main.app:app
Python cannot import a module named "Agricon-main" (hyphen in the name).
"""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> int:
    root = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.join(root, "Agricon-main")
    if not os.path.isfile(os.path.join(app_dir, "app.py")):
        sys.stderr.write(
            f"ERROR: Expected {app_dir}/app.py — wrong working directory or missing Agricon-main.\n"
        )
        return 1
    port = os.environ.get("PORT", "5000")
    cmd = [
        sys.executable,
        "-m",
        "gunicorn",
        "app:app",
        "--chdir",
        app_dir,
        "--bind",
        f"0.0.0.0:{port}",
        "--workers",
        "1",
        "--threads",
        "2",
        "--timeout",
        "180",
        "--graceful-timeout",
        "30",
    ]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
