#!/usr/bin/env python3
"""Replay certificate for the selected scenarios on the engine the conversations run on.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/rule-ablation-followup/certify.py

The certificate is case_gate.check (the gate every suite scenario passed, p15:585): the
reference solution, replayed on the scenario's own world in code-fixed's 2,337-line
generated/env_telecom.py, raises nothing and leaves nothing broken (repair), or hands off
on a world that still resists repair (escalation). prefix_gate.check (every step still
needed when it is taken) is recorded beside it, for the suite too, as information.
Failing scenarios are listed and left out of narration. No model calls.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from design import CERTIFICATE, MAIN, PROJECT, SELECTION, sha_file  # noqa: E402

ENGINE = PROJECT / "code-fixed"
sys.path[:0] = [str(PROJECT / "scripts"), str(MAIN)]
from runlog import tree_hash  # noqa: E402
import rules  # noqa: E402

SAME = ["generated/env_telecom.py", "services/case_gate.py", "services/prefix_gate.py"]


def main() -> int:
    if tree_hash(ENGINE) != rules.COPY_TREE:
        raise SystemExit("code-fixed is not the registered copy")
    for f in SAME:
        if sha_file(ENGINE / f) != sha_file(PROJECT / "code-original" / f):
            raise SystemExit(f"{f} differs between the copies")
    if len((ENGINE / SAME[0]).read_text().splitlines()) != 2337:
        raise SystemExit("engine is not the 2,337-line build")
    sys.path[:0] = [str(ENGINE), str(ENGINE / "third_party" / "tau3-bench" / "src")]
    mod = importlib.import_module("generated.env_telecom")
    case_gate = importlib.import_module("services.case_gate")
    prefix_gate = importlib.import_module("services.prefix_gate")
    if Path(mod.__file__).resolve() != ENGINE / SAME[0]:
        raise SystemExit(f"engine loaded from {mod.__file__}")
    sel = json.loads(SELECTION.read_text())
    rows = []
    for x in sel["cases"]:
        gate = case_gate.check(mod, x["case"])
        rows.append({"rule": x["rule"], "walk_index": x["walk_index"],
                     "certified": bool(gate["solvable"]), "gate": gate,
                     "prefix_gate": prefix_gate.check(mod, x["case"])})
    suite_prefix = [prefix_gate.check(mod, c)["ok"] for c in rules.suite()]
    if tree_hash(ENGINE) != rules.COPY_TREE:
        raise SystemExit("code-fixed changed during the check")
    out = {"engine": "code-fixed/generated/env_telecom.py", "engine_sha256": sha_file(ENGINE / SAME[0]),
           "selection_sha256": sha_file(SELECTION), "cases": rows,
           "dropped": [r["walk_index"] for r in rows if not r["certified"]],
           "suite_prefix_gate_ok": suite_prefix}
    CERTIFICATE.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"certified {sum(r['certified'] for r in rows)} of {len(rows)}; dropped {out['dropped']}; "
          f"prefix_gate ok {sum(r['prefix_gate']['ok'] for r in rows)} of {len(rows)} "
          f"(suite {sum(suite_prefix)} of {len(suite_prefix)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
