#!/usr/bin/env python3
"""Amendment 3: cut R5 from 10 to 8 scenarios (the brief's rule when the cap would be passed).

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/rule-ablation-followup/reduce_r5.py

Keeps all 6 R3 scenarios and a seeded sample of 8 of the 10 narrated, certified R5 ones
(no outcome data), in their original order. Reads suite_followup_16.json, writes
suite_followup.json. No model calls.
"""
from __future__ import annotations

import json
import random
import sys

sys.dont_write_bytecode = True
from design import HERE, SEED, SELECTION, SUITE, functional_key  # noqa: E402

FULL = HERE / "suite_followup_16.json"
FULL_SHA = "6253fd033ab336ca152b920ba0413e7dbca11ca81793f2c6626e3f2269a3db20"
KEEP_R5 = 8


def reduce(full: list[dict], selection: dict) -> list[dict]:
    rule = {x["functional_key"]: x["rule"] for x in selection["cases"]}
    r5 = [i for i, c in enumerate(full) if rule[functional_key(c)] == "R5_escalate_unfixable"]
    keep = set(random.Random(f"{SEED}:R5_escalate_unfixable:reduce{KEEP_R5}").sample(r5, KEEP_R5))
    return [c for i, c in enumerate(full) if i not in r5 or i in keep]


def main() -> int:
    import hashlib
    if hashlib.sha256(FULL.read_bytes()).hexdigest() != FULL_SHA:
        raise SystemExit("suite_followup_16.json is not the amendment-2 file")
    out = reduce(json.loads(FULL.read_text()), json.loads(SELECTION.read_text()))
    SUITE.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"kept {len(out)} scenarios; sha256 {hashlib.sha256(SUITE.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
