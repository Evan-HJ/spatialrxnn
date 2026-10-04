"""Compatibility entry point for the documented Grambow experiment."""

import sys

from scripts.run_experiment import main


if __name__ == "__main__":
    raise SystemExit(main(["--config", "configs/grambow_2023.json", *sys.argv[1:]]))
