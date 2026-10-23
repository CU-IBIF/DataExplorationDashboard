"""Desktop entry point (FeatureScope.exe).

Double-clicked with no arguments it starts the dashboard on a free local port,
keeps projects in ``Documents/FeatureScope`` and opens the browser.  With
arguments it behaves exactly like the ``featurescope`` command line
(``FeatureScope.exe run <project> --pipeline imaging``, ``... selftest``).
"""
from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path


def _user_dirs() -> tuple[Path, Path]:
    home = Path.home()
    docs = home / "Documents"
    workspace = Path(os.environ.get("FEATURESCOPE_WORKSPACE", (docs if docs.exists() else home) / "FeatureScope"))
    cache = Path(os.environ.get("LOCALAPPDATA", home / ".cache")) / "FeatureScope"
    return workspace, cache


def main() -> int:
    multiprocessing.freeze_support()   # required for joblib/loky workers in a frozen app
    workspace, cache = _user_dirs()
    cache.mkdir(parents=True, exist_ok=True)
    # numba (UMAP / pynndescent) writes its JIT cache next to modules unless told otherwise;
    # inside an installed app that folder is read-only.
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache / "numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")

    from .cli import main as cli_main
    if len(sys.argv) > 1:
        return cli_main(sys.argv[1:]) or 0
    workspace.mkdir(parents=True, exist_ok=True)
    from .selftest import free_port
    port = free_port()
    print("=" * 64, " FeatureScope", f" Dashboard : http://127.0.0.1:{port}/", f" Projects  : {workspace}",
          " Close this window to stop the dashboard.", "=" * 64, sep="\n", flush=True)
    return cli_main(["serve", "--workspace", str(workspace), "--port", str(port), "--open"]) or 0


if __name__ == "__main__":
    sys.exit(main())
