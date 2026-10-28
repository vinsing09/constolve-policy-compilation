#!/usr/bin/env python3
"""Rebuild the scenario pool the 24-scenario suite came from, and pick new scenarios by rule tag.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/rule-ablation-followup/select_scenarios.py

No model calls. Runs code-original's own generator (phase15_graph_suite: atom_audit.refile,
policy_graph.walks max_len 4, build_case, whose case_gate.check is the replay certificate)
with the atoms file as it was at 7257eda687. Refuses unless all 24 suite scenarios come out
byte-identical. Writes pool_index.json (every distinct case: walk, keys, tags) and
selection.json (the picked cases, not yet narrated).
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from design import HERE, MAIN, PER_RULE, PROJECT, RULES, SEED, SELECTION, functional_key, tied  # noqa: E402

GEN = PROJECT / "code-original"
ATOMS = HERE / "inputs" / "phase3_atoms_telecom.json"
ATOMS_GIT_BLOB = "2066b1a6a62acc82eba0b7c483971167c6247994"  # 7257eda687:reports/phase3_atoms_telecom.json
ENGINE_LINES, MAX_LEN = 2337, 4
sys.path[:0] = [str(PROJECT / "scripts"), str(MAIN)]
from runlog import tree_hash  # noqa: E402
import rules  # noqa: E402

TREE_ORIGINAL = (PROJECT / "scripts" / "tree-original.txt").read_text().split()[0]


def git_blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def exact_key(case: dict) -> str:
    keep = {k: case.get(k) for k in ("kind", "success_criteria", "reference_solution")}
    keep["world"] = (case.get("world_setup") or {}).get("entities")
    return hashlib.sha256(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()


def tags(case: dict) -> dict[str, list[str]]:
    """Criteria tied to each follow-up rule (design.tied: R5 strict)."""
    return {r: [c["id"] for c in case["success_criteria"] if tied(r, c, case)] for r in RULES}


def build() -> tuple[list[dict], dict]:
    """Every distinct case build_case emits over the suite's walks, in walk order."""
    if git_blob(ATOMS.read_bytes()) != ATOMS_GIT_BLOB:
        raise SystemExit("atoms file is not the one at 7257eda687")
    sys.path[:0] = [str(GEN), str(GEN / "third_party" / "tau3-bench" / "src")]
    spec = importlib.util.spec_from_file_location("phase15", GEN / "scripts" /
                                                  "phase15_graph_suite.py")
    p15 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p15)
    mod = importlib.import_module("generated.env_telecom")
    if Path(mod.__file__).resolve() != GEN / "generated" / "env_telecom.py" or \
            len(Path(mod.__file__).read_text().splitlines()) != ENGINE_LINES:
        raise SystemExit("engine is not code-original's 2,337-line build")
    graph = json.loads((GEN / "generated" / "policy_graph_telecom.json").read_text())
    atoms, _ = p15.atom_audit.refile(mod, json.loads(ATOMS.read_text())["atoms"])
    envmod = importlib.import_module("tau2.domains.telecom.environment")
    deps = p15.policy_graph.prerequisites(envmod.get_environment().get_policy() or "",
                                          [g["family"] for g in graph])
    walks = p15.policy_graph.walks(graph, max_len=MAX_LEN, deps=deps)
    p15.build_case.rejected, p15.build_case.pruned = [], []
    pool, seen = [], set()
    for i, w in enumerate(walks):
        c = p15.build_case(w, mod, atoms, i)
        if c is None:
            continue
        if not p15.case_gate.check(mod, c)["solvable"]:
            raise SystemExit(f"walk {i}: build_case emitted a case its own gate rejects")
        k = exact_key(c)
        if k not in seen:
            seen.add(k)
            pool.append({"walk_index": i, "exact_key": k, "functional_key": functional_key(c),
                         "kind": c["kind"], "tags": tags(c), "case": c})
    return pool, {"walks": len(walks), "built_distinct": len(pool),
                  "rejected_at_gate": len(p15.build_case.rejected)}


def select(pool: list[dict], suite: list[dict]) -> tuple[dict, dict]:
    """Per rule: tagged, not functionally a suite case, one per functional key (lowest
    walk), then a seeded sample of PER_RULE. Never looks at any run's outcome."""
    suite_keys = {functional_key(c) for c in suite}
    picked, record = {}, {}
    for r in RULES:
        reps = {}
        for p in pool:
            if p["tags"][r] and p["functional_key"] not in suite_keys:
                reps.setdefault(p["functional_key"], p)
        cands = sorted(reps.values(), key=lambda p: p["walk_index"])
        chosen = sorted(random.Random(f"{SEED}:{r}").sample(cands, min(PER_RULE, len(cands))),
                        key=lambda p: p["walk_index"])
        picked[r] = chosen
        record[r] = {"tagged": sum(bool(p["tags"][r]) for p in pool),
                     "tagged_functional_suite_duplicates": sum(
                         bool(p["tags"][r]) and p["functional_key"] in suite_keys for p in pool),
                     "candidates": len(cands), "picked": len(chosen),
                     "walk_indices": [p["walk_index"] for p in chosen]}
    both = {p["functional_key"] for p in picked[RULES[0]]} & \
        {p["functional_key"] for p in picked[RULES[1]]}
    if both:
        raise SystemExit("a scenario was picked for both rules")
    return picked, record


def main() -> int:
    if tree_hash(GEN) != TREE_ORIGINAL:
        raise SystemExit("code-original does not match scripts/tree-original.txt")
    pool, stats = build()
    if tree_hash(GEN) != TREE_ORIGINAL:
        raise SystemExit("code-original changed while building")
    suite = rules.suite()
    exact = {p["exact_key"]: p["walk_index"] for p in pool}
    missing = [i for i, c in enumerate(suite) if exact_key(c) not in exact]
    if missing:
        raise SystemExit(f"suite scenarios the generator does not reproduce: {missing}")
    stats["suite_walks"] = [exact[exact_key(c)] for c in suite]
    stats["suite_functional_distinct"] = len({functional_key(c) for c in suite})
    picked, record = select(pool, suite)
    (HERE / "pool_index.json").write_text(json.dumps(
        {"stats": stats, "pool": [{k: v for k, v in p.items() if k != "case"} for p in pool]},
        indent=0) + "\n")
    SELECTION.write_text(json.dumps(
        {"seed": SEED, "per_rule": PER_RULE, "atoms_git_blob": ATOMS_GIT_BLOB,
         "max_len": MAX_LEN, "generator_tree": TREE_ORIGINAL, "stats": stats,
         "record": record,
         "cases": [{"rule": r, "walk_index": p["walk_index"],
                    "functional_key": p["functional_key"], "tied": p["tags"][r],
                    "case": p["case"]} for r in RULES for p in picked[r]]},
        indent=1, default=str) + "\n")
    print(json.dumps({"stats": stats, "record": record}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
