#!/usr/bin/env python3
"""Section 5.1: determinism replays, checked against the archived replay reports.

    python -B reproduce/determinism.py

records/determinism/phase8_determinism_{telecom,airline,retail}.json are the reports the private
scripts/phase8_determinism.py wrote. That script replays each scenario's reference sequence
repeatedly with no agent, in one process and across processes, and keeps only counts of
scenarios whose final state, predicate values and (in-process only) tool return values were
identical on every replay. No per-replay record was kept, so this checks the stored summaries;
it does not recompute them. Exit 1 on any difference from the paper.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REC = Path(__file__).resolve().parents[1] / "records/determinism"
PAPER = {"telecom": (30, 3), "airline": (7, 4), "retail": (4, 4)}


def load() -> dict[str, tuple[dict, dict]]:
    out = {}
    for dom in PAPER:
        r = json.loads((REC / f"phase8_determinism_{dom}.json").read_text())
        out[dom] = r, {**{f"in-process {x}": v for x, v in r["in_process"].items()},
                       **{f"cross-process {x}": v for x, v in r["cross_process"].items()}}
    return out


def check() -> list[str]:
    bad = []
    for dom, (r, same) in load().items():
        n, k = PAPER[dom]
        if (r["tasks"], r["repeats"]) != (n, k):
            bad.append(f"{dom}: {r['tasks']}x{r['repeats']}, paper {n}x{k}")
        if r["diverged"] or any(v != n for v in same.values()) or "returns" not in r["in_process"]:
            bad.append(f"{dom}: divergence or missing comparison")
    return bad


def main() -> int:
    for dom, (r, same) in load().items():
        print(f"{dom:8s} {r['tasks']} scenarios x {r['repeats']} replays; identical: "
              + ", ".join(f"{x} {v}" for x, v in same.items()) + f"; diverged {r['diverged']}")
    bad = check()
    for b in bad:
        print("MISMATCH", b)
    print("determinism: archived reports match the paper (no divergence)" if not bad
          else f"determinism: {len(bad)} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
