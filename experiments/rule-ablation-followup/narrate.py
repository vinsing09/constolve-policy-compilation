#!/usr/bin/env python3
"""Write the customer brief for each certified scenario the way the suite did, plus one line.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/rule-ablation-followup/narrate.py

Same path as p15's narrate_one (p15:833-864) and roaming note (p15:880-890), on
code-original: meta_user.build writes reason_for_call (with design.NARRATOR_MODEL, passed
explicitly; prereg amendment 1); everything else is assembled. Then WEBCHAT is appended to known_info, the
same for every scenario and arm. Paid: every model call goes through the follow-up ledger
guard under the $45 cap. Writes suite_followup.json; refuses to overwrite it.
"""
from __future__ import annotations

import asyncio
import contextvars
import importlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
from design import (CAP_USD, CERTIFICATE, LEDGER, NARRATOR_MODEL, PROJECT, SELECTION,  # noqa: E402
                    SUITE, WEBCHAT, sha_file)
import ledger as lg  # noqa: E402

GEN = PROJECT / "code-original"
sys.path.insert(0, str(PROJECT / "scripts"))
from runlog import tree_hash  # noqa: E402

TREE_ORIGINAL = (PROJECT / "scripts" / "tree-original.txt").read_text().split()[0]
ROAM = {"enable_roaming", "toggle_roaming"}
CASE = contextvars.ContextVar("case", default=None)


class Budget:
    def __init__(self, hard: float):
        self.hard, self.spent, self.reserved, self.run, self.stopped = hard, 0.0, 0.0, "narrate", set()


def finish(case: dict, sc: dict) -> dict:
    """narrate_one's output, the roaming note, then the web-chat line."""
    ins = sc.get("instructions") or {}
    if not (ins.get("reason_for_call") or "").strip():
        raise SystemExit(f"no reason_for_call for {case['source_task_id']}")
    out = dict(case)
    out["scenario"] = sc
    out["goal"] = ins["reason_for_call"]
    out["known_facts"] = {**(case.get("known_facts") or {}),
                          "known_info": ins.get("known_info") or ""}
    out["task_instructions"] = ins.get("task_instructions") or ""
    out["wont_volunteer"] = []
    out["will_reveal_if_asked"] = []
    if {x["param"].get("tool") for x in out["success_criteria"]} & ROAM:
        out["will_reveal_if_asked"].insert(0, "I am travelling outside my home network at the moment.")
        out["known_facts"] = dict(out["known_facts"], travelling="outside home network")
    info = f"{out['known_facts']['known_info']} {WEBCHAT}".strip()
    out["known_facts"]["known_info"] = info
    sc["instructions"]["known_info"] = info
    return out


async def narrate(p15, mod, cases: list[dict]) -> list[dict]:
    graph = json.loads((GEN / "generated" / "policy_graph_telecom.json").read_text())
    branches = json.loads((GEN / "generated" / "branches_telecom.json").read_text())

    async def one(i, case):
        CASE.set(f"narrate|{i}")
        sc = await p15.meta_user.build(mod=mod, domain="telecom", case=case, graph=graph,
                                       branches=branches, model=NARRATOR_MODEL)
        return finish(case, sc)

    return list(await asyncio.gather(*[one(i, c) for i, c in enumerate(cases)]))


def main() -> int:
    if SUITE.exists():
        raise SystemExit(f"{SUITE.name} exists; narration is not redone silently")
    cert = json.loads(CERTIFICATE.read_text())
    if cert["selection_sha256"] != sha_file(SELECTION):
        raise SystemExit("certificate.json is not for this selection.json")
    ok = {r["walk_index"] for r in cert["cases"] if r["certified"]}
    cases = [x["case"] for x in json.loads(SELECTION.read_text())["cases"] if x["walk_index"] in ok]
    if tree_hash(GEN) != TREE_ORIGINAL:
        raise SystemExit("code-original does not match scripts/tree-original.txt")
    held = lg.lock(LEDGER)
    if held is None:
        raise SystemExit("another paid process holds the follow-up ledger lock")
    sys.path[:0] = [str(GEN), str(GEN / "third_party" / "tau3-bench" / "src")]
    spec = importlib.util.spec_from_file_location("phase15", GEN / "scripts" / "phase15_graph_suite.py")
    p15 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p15)
    mod = importlib.import_module("generated.env_telecom")
    import litellm
    budget = Budget(CAP_USD - lg.prior_spend(LEDGER))
    lg.install_guard(litellm, budget, LEDGER, {NARRATOR_MODEL}, CASE.get)
    try:
        out = asyncio.run(narrate(p15, mod, cases))
    finally:
        held.close()
    if tree_hash(GEN) != TREE_ORIGINAL:
        raise SystemExit("code-original changed while narrating")
    SUITE.write_text(json.dumps(out, indent=1, default=str) + "\n")
    print(f"narrated {len(out)}; spent ${budget.spent:.4f}; suite sha256 {sha_file(SUITE)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
