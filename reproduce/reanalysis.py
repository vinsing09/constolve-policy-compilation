#!/usr/bin/env python3
"""Run experiments/duplicate-reanalysis/reanalyse.py unchanged, with the offline git check. No model calls.

    python -B reproduce/reanalysis.py && git diff --exit-code experiments/duplicate-reanalysis/results.json

reanalyse.py loads both rule-removal runs in registered mode, whose git provenance checks refer
to the private development repo (see offline_git.py). It rewrites results.json; an unchanged
file in git means the re-analysis reproduced exactly.
"""
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offline_git  # noqa: E402

offline_git.install()
runpy.run_path(str(offline_git.ROOT / "experiments/duplicate-reanalysis/reanalyse.py"), run_name="__main__")
