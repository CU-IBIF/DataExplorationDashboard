# Entry script for PyInstaller (absolute import so the frozen app resolves the package).
import sys

from featurescope.launcher import main

if __name__ == "__main__":
    sys.exit(main())
