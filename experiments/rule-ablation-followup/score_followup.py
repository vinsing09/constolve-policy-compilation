#!/usr/bin/env python3
"""Score the follow-up run against prereg.md, then the secondary pooled analysis with the main run.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/rule-ablation-followup/score_followup.py \
        experiments/rule-ablation-followup/results/main

Primary (follow-up alone): per removed rule, tied-criteria pass %, control vs variant, over
(repeat, scenario) blocks where all three arms are clean; one-sided paired sign-flip over
scenarios (../rule-ablation/score_ablation.compare, unchanged); untied change as the
false-alarm check. Secondary: main run + follow-up pooled per rule, scenarios with the same
functional key counted as one. Writes <run>/score.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import design as D  # noqa: E402
import run_followup as rf  # noqa: E402
import run_manifest as rm  # noqa: E402
import score as sc  # noqa: E402
import score_ablation as sa  # noqa: E402
import run_ablation as ra  # noqa: E402
import rules  # noqa: E402
from runlog import conv_key  # noqa: E402

Refused = sa.Refused
MAIN_RUN = D.MAIN / "results" / "main"
TEST_ONLY = {"--copy", "--results", "--cases", "--repeats", "--arms"}


def registered() -> dict:
    return {"arms": list(D.ARMS), "cases": len(suite()), "repeats": D.REPEATS, "cap": 30,
            "concurrency": 8, "copy": "code-fixed", "suite_sha256": D.SUITE_SHA,
            "models_requested": {"agent": rm.MODEL, "customer": rm.MODEL},
            "temperatures": {"agent": 0, "customer": 0.7},
            "models_resolved": {"agent": rm.MODEL, "customer": rm.MODEL,
                                "customer_temperature": 0.7}}


def suite() -> list[dict]:
    return json.loads(D.SUITE.read_text())


def ties(cases: list[dict] | None = None) -> dict[str, set]:
    """(scenario, criterion) pairs tied to each rule by design.tied (R5 strict)."""
    cases = suite() if cases is None else cases
    return {r: {(i, c["id"]) for i, case in enumerate(cases)
                for c in case["success_criteria"] if D.tied(r, c, case)} for r in D.RULES}


def check_registered(run: Path, m: dict) -> None:
    got = {**m, "arms": list(m.get("arms") or {})}
    bad = [k for k, v in registered().items() if got.get(k) != v]
    cmd = m.get("command") or []
    if bad or m.get("run") != run.name or m["summary"].get("copy_unchanged") is not True \
            or (m.get("copy_checksum"), m.get("copy_tree_hash")) != (rules.COPY_CHECKSUM,
                                                                     rules.COPY_TREE) \
            or not str(cmd[1:2]).endswith(f"{rf.F}run_followup.py']") or TEST_ONLY & set(cmd):
        raise Refused(f"not the registered run: {bad or 'run/copy/command'}")
    if not (sc.git_ok("merge-base", "--is-ancestor", str(m.get("project_commit")), "HEAD")
            and sc.git_ok("diff", "--quiet", str(m["project_commit"]), "HEAD", "--",
                          *rf.PINNED)):
        raise Refused("project commit missing or pinned files changed since the run")


def check_ids(ids: list[str], n_cases: int, full: bool) -> None:
    """The run's case ids: suite_followup.json's first n_cases, in order; all of them when
    `full` (a registered run). A shortened id list would shrink the completeness check."""
    want = [f"{i}:{c.get('source_task_id', '')}"[:120] for i, c in enumerate(suite())]
    if len(ids) != n_cases or ids != want[:n_cases] or (full and ids != want):
        raise Refused("case ids are not suite_followup.json's, in order and complete")


def load(run: Path, is_registered: bool = True) -> tuple[dict, list[dict], int]:
    m = json.loads((run / "manifest.json").read_text())
    s = m.get("summary") or {}
    if s.get("problems") or not s.get("status"):
        raise Refused(f"run has problems or no summary: {s.get('problems')}")
    seal = rm.seal_path(run)
    if not seal.exists() or seal.read_text() != rm.seal(run):
        raise Refused("seal missing or the run folder changed since it was sealed")
    if m.get("suite_sha256") != D.sha_file(D.SUITE):
        raise Refused("run's scenarios are not suite_followup.json")
    if is_registered:
        if ra.dirty(rf.PINNED):
            raise Refused("pinned files have uncommitted changes")
        sc.committed(seal)
        check_registered(run, m)
    prompts = {k: set(v) for k, v in json.loads((run / "prompts.json").read_text()).items()}
    stopped = set(json.loads((run / "stopped.json").read_text()))
    want_sha = {a: v["policy_sha256"] for a, v in m["arms"].items()}
    again = ra.check(run, list(m["arms"]), m["repeats"], m["case_ids"], prompts, want_sha,
                     stopped)
    if again["problems"] or any(again[k] != s.get(k) for k in again if k != "problems"):
        raise Refused(f"recomputed run check disagrees: {again['problems'][:3]}")
    meta = json.loads((D.MAIN / "variants.json").read_text())
    for arm, v in m["arms"].items():
        if v["policy_sha256"] != meta[arm]["sha256"]:
            raise Refused(f"{arm}: run policy hash is not the main run's")
    cases, ids = suite(), m["case_ids"]
    check_ids(ids, m["cases"], is_registered)
    arms, status = list(m["arms"]), s["status"]
    valid = [(r, c) for r in range(m["repeats"]) for c in ids
             if all(status.get(conv_key(a, r, c)) == "clean" for a in arms)]
    traces = {}
    for f in (run / "traces").glob("*.json"):
        d = json.loads(f.read_text())
        if (d["arm"], d["repeat"], d["case"]) in traces:
            raise Refused(f"duplicate trace {f.name}")
        traces[(d["arm"], d["repeat"], d["case"])] = d
    rows = []
    for r, c in valid:
        i = ids.index(c)
        want = [k["id"] for k in cases[i]["success_criteria"]]
        for a in arms:
            d = traces.get((a, r, c))
            if d is None:
                raise Refused(f"no trace for clean conversation {a}|{r}|{c}")
            crits = d["criteria"]
            if [k["id"] for k in crits] != want or not all(isinstance(k["ok"], bool)
                                                         for k in crits):
                raise Refused(f"{a}|{r}|{c}: criteria differ from the suite")
            ok = [k["ok"] for k in crits]
            if (d["solved"], d["actions_done"], d["actions_required"]) != \
                    (all(ok), sum(ok), len(ok)):
                raise Refused(f"{a}|{r}|{c}: solved/actions disagree with criteria")
            rows += [{"arm": a, "rep": r, "scen": i, "cid": k["id"], "ok": k["ok"]}
                     for k in crits]
    return m, rows, len(valid)


def score(rows: list[dict], tied: dict[str, set]) -> dict:
    every = {(x["scen"], x["cid"]) for x in rows}
    return {r: {"tied": sa.compare(rows, r, tied[r]), "untied": sa.compare(rows, r, every - tied[r])}
            for r in D.RULES if any(x["arm"] == r for x in rows)}


def pooled(main_rows: list[dict], fu_rows: list[dict], fu_tied: dict[str, set]) -> dict:
    """Secondary: both runs per rule, one unit per functionally distinct scenario."""
    main_ties = ties(rules.suite())
    u0 = [D.functional_key(c) for c in rules.suite()]
    u1 = [D.functional_key(c) for c in suite()]
    if set(u0) & set(u1):
        raise Refused("a follow-up scenario is functionally a main-run scenario")
    rows = [dict(x, scen=u0[x["scen"]]) for x in main_rows] + \
        [dict(x, scen=u1[x["scen"]]) for x in fu_rows]
    tied = {r: {(u0[i], c) for i, c in main_ties[r]} | {(u1[i], c) for i, c in fu_tied[r]}
            for r in D.RULES}
    return score(rows, tied)


def main(argv=None) -> int:
    run = Path((argv or sys.argv[1:])[0]).resolve()
    m, rows, n_valid = load(run)
    tied = ties()
    out = score(rows, tied)
    _, main_rows, main_valid = sa.load(MAIN_RUN)
    pool = pooled(main_rows, rows, tied)
    res = {"run": m["run"], "valid_blocks": n_valid, "blocks": m["summary"]["blocks"],
           "cost_usd": m["summary"]["cost_usd"], "rules": out,
           "pooled_secondary": {"main_valid_blocks": main_valid, "rules": pool},
           "ties": {r: sorted(map(list, t)) for r, t in tied.items()},
           "scorer_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=D.HERE,
                                           capture_output=True, text=True).stdout.strip()}
    (run / "score.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"valid blocks {n_valid} of {m['summary']['blocks']}, cost ${m['summary']['cost_usd']}")
    print("PRIMARY (follow-up alone)\n" + sa.table(out))
    print("SECONDARY (main + follow-up pooled, functional duplicates merged)\n" + sa.table(pool))
    return 0


if __name__ == "__main__":
    sys.exit(main())
