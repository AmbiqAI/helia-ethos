"""Run the standalone package's required development checks."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Check the lock, source quality and complete package test suite."""
    root = Path(__file__).resolve().parents[1]
    commands = [
        ["uv", "lock", "--check"],
        ["uv", "run", "--frozen", "ruff", "check", "."],
        ["uv", "run", "--frozen", "ty", "check"],
        [sys.executable, "-m", "pytest", "-q"],
    ]
    for command in commands:
        subprocess.run(command, cwd=root, check=True)


if __name__ == "__main__":
    main()
