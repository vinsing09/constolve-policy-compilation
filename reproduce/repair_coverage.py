#!/usr/bin/env python3
"""Section 5.2: fault-type match and repair-call-set coverage, recomputed from released records.

    python -B reproduce/repair_coverage.py

Inputs, all in this repository:
  records/coverage/emitted_repair_sets.json.xz   the repair-call set of each of the 92,184
                                                 scenarios the generator emitted (distinct sets
                                                 with counts)
  records/coverage/tasks_small.json              tau3-bench's 20 single-fault telecom tasks
  experiments/rule-ablation-followup/inputs/phase3_atoms_telecom.json   the 28 discovered atoms
  code-fixed/third_party/tau3-bench/data/tau2/domains/telecom/{tasks,split_tasks}.json

The method is that of the private scripts/phase4_coverage.py: a benchmark fault's repair is the
set of actions of its single-fault task; a benchmark task's repair-call set is the union over its
faults, without transfer_to_human_agents; the task is covered when some emitted scenario has the
same set. As in that script, an atom's repair is its first repairing call (`fix.tool`); the
script also prints the count when every call of each atom's `fix_sequence` is counted, under
which the overdue-bill fault matches completely. The result is also compared with the archived report the private script wrote.
Exit 1 on any difference from the paper or from the archived report.
"""
from __future__ import annotations

import json
import lzma
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REC = ROOT / "records/coverage"
TAU = ROOT / "code-fixed/third_party/tau3-bench/data/tau2/domains/telecom"
ATOMS = ROOT / "experiments/rule-ablation-followup/inputs/phase3_atoms_telecom.json"
TRANSFER = "transfer_to_human_agents"
PAPER = {"emitted": 92184, "base tasks": 114, "denominator": 113, "covered": 105,
         "repair-graded": 94, "repair-graded covered": 90, "escalation-graded covered": 15,
         "uncovered": 8, "fault types": 19, "agent-repairable": 17, "complete": 16,
         "partial": 1, "escalation-only": 2, "partial fault": ["overdue_bill_suspension"]}


def _fault(task_id: str) -> str:
    return re.sub(r"\[PERSONA:.*$", "", re.sub(r"^\[[a-z_]+\]", "", task_id))


def _xz(p: Path):
    with lzma.open(p, "rt") as f:
        return json.load(f)


def fault_repairs() -> dict[str, frozenset]:
    out: dict[str, set] = {}
    for t in json.loads((REC / "tasks_small.json").read_text()):
        acts = (t.get("evaluation_criteria") or {}).get("actions") or []
        out.setdefault(_fault(t["id"]), set()).update(a["name"] for a in acts)
    return {k: frozenset(v) for k, v in out.items()}


def compute() -> dict:
    gt = fault_repairs()
    emitted = _xz(REC / "emitted_repair_sets.json.xz")
    sets = {frozenset(s) for _, s in emitted["sets"]}
    n_emitted = sum(n for n, _ in emitted["sets"])

    base = set(json.loads((TAU / "split_tasks.json").read_text())["base"])
    tasks = [t for t in json.loads((TAU / "tasks.json").read_text()) if t["id"] in base]

    def sig(t) -> frozenset:
        calls: set = set()
        for f in filter(None, _fault(t["id"]).split("|")):
            calls |= gt[f]
        return frozenset(calls - {TRANSFER})

    def escalation_graded(t) -> bool:
        return {a["name"] for a in t["evaluation_criteria"].get("actions") or []} == {TRANSFER}

    scored = [t for t in tasks if sig(t)]
    covered = [t["id"] for t in scored if sig(t) in sets]
    missing = [t["id"] for t in scored if sig(t) not in sets]
    repair = [t for t in scored if not escalation_graded(t)]
    task_sigs = {sig(t) for t in tasks}

    atoms = json.loads(ATOMS.read_text())["atoms"]
    fix_tools = {a["fix"]["tool"] for a in atoms}
    seq_tools = fix_tools | {s["tool"] for a in atoms for s in a.get("fix_sequence", [])}
    kinds = {}
    for f, r in gt.items():
        r = r - {TRANSFER}
        kinds[f] = ("escalation-only" if not r else "complete" if r <= fix_tools
                    else "partial" if r & fix_tools else "unmatched")
    return {
        "emitted": n_emitted, "base tasks": len(tasks), "denominator": len(scored),
        "excluded": [t["id"] for t in tasks if not sig(t)],
        "covered": len(covered), "uncovered": len(missing),
        "repair-graded": len(repair),
        "repair-graded covered": sum(t["id"] in covered for t in repair),
        "escalation-graded covered": sum(escalation_graded(t) for t in scored if t["id"] in covered),
        "fault types": len(gt), "agent-repairable": sum(k != "escalation-only" for k in kinds.values()),
        "complete": sum(k == "complete" for k in kinds.values()),
        "partial": sum(k == "partial" for k in kinds.values()),
        "escalation-only": sum(k == "escalation-only" for k in kinds.values()),
        "partial fault": sorted(f for f, k in kinds.items() if k == "partial"),
        "complete by fix sequence": sum(bool(r - {TRANSFER}) and r - {TRANSFER} <= seq_tools
                                        for r in gt.values()),
        "_lists": {"covered": covered, "missing": missing,
                   "margin": sorted(sorted(s) for s in sets if s not in task_sigs)},
        "_kinds": kinds,
    }


def check() -> list[str]:
    r = compute()
    bad = [f"{k}: got {r[k]!r}, paper {v!r}" for k, v in PAPER.items() if r[k] != v]
    arch = _xz(REC / "archived/phase4_coverage_telecom.json.xz")
    for k in ("covered", "missing"):
        if r["_lists"][k] != arch[k]:
            bad.append(f"{k} list differs from the archived report")
    if r["_lists"]["margin"] != sorted(arch["margin"]):
        bad.append("emitted sets with no benchmark task differ from the archived report")
    if arch["n_ours"] != r["emitted"] or len(arch["atom_correspondence"]) != r["agent-repairable"]:
        bad.append("archived n_ours or atom correspondence differs")
    return bad


def main() -> int:
    r = compute()
    print(f"emitted scenarios {r['emitted']:,} ({len(r['_lists']['margin']):,} distinct repair "
          f"sets match no benchmark task)")
    print(f"benchmark tasks {r['base tasks']}; excluded (repair set empty): {r['excluded']}")
    print(f"covered {r['covered']} of {r['denominator']}; repair-graded {r['repair-graded covered']} "
          f"of {r['repair-graded']}; escalation-graded tasks among the covered "
          f"{r['escalation-graded covered']}; uncovered {r['uncovered']}")
    print(f"fault types {r['fault types']}: agent-repairable {r['agent-repairable']} "
          f"(complete {r['complete']}, partial {r['partial']} {r['partial fault']}), "
          f"escalation-only {r['escalation-only']}")
    print(f"  (counting every call of each atom's fix sequence: {r['complete by fix sequence']} "
          f"complete)")
    bad = check()
    for b in bad:
        print("MISMATCH", b)
    print("coverage: all values match the paper and the archived report" if not bad
          else f"coverage: {len(bad)} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
