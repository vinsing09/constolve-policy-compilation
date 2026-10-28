#!/usr/bin/env python3
"""Recompute the rule-removal results (main run and registered follow-up) from stored traces. No model calls.

    python -B reproduce/ablation.py

Runs the experiments' own loaders and scorers unchanged, in registered mode: seal committed and
matching every file, pinned files clean in git and identical to the recording commit's
(reproduce/offline_git.py), manifest equal to the pre-registered design, policy hashes, every
clean conversation's criteria against the suite. The scores must equal each run's stored
score.json (its scorer_commit field, a private commit id, is not compared).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offline_git  # noqa: E402

offline_git.install()
import score_ablation as sa  # noqa: E402
import score_followup as sf  # noqa: E402
from rules import ties as rule_ties  # noqa: E402

MAIN = offline_git.ROOT / "experiments" / "rule-ablation"
FOLLOW = offline_git.ROOT / "experiments" / "rule-ablation-followup"


def same(got, want, what: str) -> None:
    if json.loads(json.dumps(got)) != want:
        raise SystemExit(f"{what}: recomputed result differs from the stored score.json")


def main() -> int:
    ties = json.loads((MAIN / "ties.json").read_text())
    if ties != rule_ties():
        raise SystemExit("ties.json differs from rules.tied on the suite")
    m, rows, n = sa.load(MAIN / "results" / "main")
    out = sa.score(rows, ties)
    stored = json.loads((MAIN / "results" / "main" / "score.json").read_text())
    same(out, stored["rules"], "rule-ablation main")
    print(f"RULE REMOVAL, main run: valid blocks {n} of {m['summary']['blocks']}, "
          f"cost ${m['summary']['cost_usd']}")
    print(sa.table(out))
    for v, o in out["pair"].items():
        print(f"pair {v}: own-only drop {o['own_only'].get('drop', 0):+.1f} "
              f"p={o['own_only'].get('p', 1):.4f}; partner-only drop "
              f"{o['partner_only'].get('drop', 0):+.1f} p={o['partner_only'].get('p', 1):.4f}"
              f"; lands on right rule: {o['lands_on_right_rule']}")
    print("  identical to the stored score.json\n")

    fm, frows, fn = sf.load(FOLLOW / "results" / "main")
    tied = sf.ties()
    fout = sf.score(frows, tied)
    pool = sf.pooled(rows, frows, tied)
    fstored = json.loads((FOLLOW / "results" / "main" / "score.json").read_text())
    same(fout, fstored["rules"], "follow-up primary")
    same(pool, fstored["pooled_secondary"]["rules"], "follow-up pooled secondary")
    same({r: sorted(map(list, t)) for r, t in tied.items()}, fstored["ties"], "follow-up ties")
    print(f"RULE REMOVAL, registered follow-up: valid blocks {fn} of {fm['summary']['blocks']}, "
          f"cost ${fm['summary']['cost_usd']}")
    print("PRIMARY (follow-up alone)\n" + sa.table(fout))
    print("SECONDARY (main + follow-up pooled, functional duplicates merged)\n" + sa.table(pool))
    print("  identical to the stored score.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
