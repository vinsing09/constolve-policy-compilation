#!/usr/bin/env python3
"""Score a new bench.py run (a folder of traces) with scripts/score.py's loader and metrics.

    python -B reproduce/rescore_bench.py code-fixed/logs/bench [--cases 24] [--repeats 5]

Every trace is checked against the suite (criterion ids and rules, solved/actions consistent)
and the exact set of (arm, repeat, case) must be present, as for the recorded runs. No manifest,
seal or provenance checks: a new run has none of those.
"""
import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import score as sc  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", type=Path)
    ap.add_argument("--cases", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    ids = list(sc.suite_cases(a.cases))
    rows = sc.load(a.traces, ids, a.repeats)
    s = sc.score(rows, a.repeats)
    print(sc.show({"run": a.traces.name, "copy": "new run", "conversations": len(rows), **s}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
