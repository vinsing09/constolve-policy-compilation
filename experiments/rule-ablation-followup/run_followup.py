#!/usr/bin/env python3
"""Run control, R3 and R5 on the follow-up scenarios with bench's own build_agent and one().

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/rule-ablation-followup/run_followup.py --run main

Same as ../rule-ablation/run_ablation.py (its Budget, run_all, check and policy files are
imported) except: the scenarios are suite_followup.json, built into journeys as
bench.our_cases does (bench.py:98-111); 3 arms x 14 scenarios x 5 repeats (amendment 3); the follow-up's
own ledger and $45 cap. Writes results/<run>/ and seals/<run>.sha256.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import design as D  # noqa: E402
import ledger as lg  # noqa: E402

HERE, PROJECT = D.HERE, D.PROJECT
sys.path[:0] = [str(PROJECT / "scripts"), str(D.MAIN)]
import run_manifest as rm  # noqa: E402
from runlog import CONV, Recorder, tree_hash  # noqa: E402
import run_ablation as ra  # noqa: E402
from rules import COPY_CHECKSUM, COPY_TREE, sha  # noqa: E402

LEDGER = D.LEDGER
DOMAIN, CAP_TURNS = ra.DOMAIN, ra.CAP_TURNS
F = "experiments/rule-ablation-followup/"
M = "experiments/rule-ablation/"
PINNED = [F + x for x in ("design.py", "ledger.py", "run_followup.py", "score_followup.py",
                          "prereg.md", "suite_followup.json", "suite_followup_16.json", "certificate.json",
                          "selection.json")] + \
    [M + x for x in ("run_ablation.py", "score_ablation.py", "rules.py", "variants.json",
                     "ties.json", "policies")] + \
    ["scripts/run_manifest.py", "scripts/runlog.py", "scripts/score.py",
     "scripts/checksum-fixed.txt", "scripts/tree-fixed.txt",
     "experiments/rule-coverage/coverage.py"]


def suite_cases(n: int | None) -> list[dict]:
    """The narrated, certified follow-up scenarios, refused unless they are the registered file."""
    if D.SUITE_SHA is None or D.sha_file(D.SUITE) != D.SUITE_SHA:
        raise SystemExit("suite_followup.json is not the registered file (design.SUITE_SHA)")
    cert = json.loads(D.CERTIFICATE.read_text())
    suite = json.loads(D.SUITE.read_text())
    walk = {x["functional_key"]: x["walk_index"] for x in json.loads(D.SELECTION.read_text())["cases"]}
    ok = {r["walk_index"] for r in cert["cases"] if r["certified"]}
    if cert["selection_sha256"] != D.sha_file(D.SELECTION) or \
            not all(walk.get(D.functional_key(c)) in ok for c in suite):
        raise SystemExit("a scenario lacks its replay certificate")
    return suite if n is None else suite[:n]


def journeys(suite: list[dict], copy: Path) -> list[dict]:
    from services.suite_bench import journey_of
    if Path(sys.modules["services.suite_bench"].__file__).resolve() != \
            copy / "services" / "suite_bench.py":
        raise SystemExit("journey_of is not the copy's")
    out = []
    for i, c in enumerate(suite):
        j = journey_of(c, None, i)
        j["kind"] = "ours"
        j["case_kind"] = c.get("kind") or "repair"
        out.append(j)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    test = os.environ.get("RULE_ABLATION_TEST") == "1"     # fake-model tests only
    if test:
        ap.add_argument("--copy", default=str(PROJECT / "code-fixed"))
        ap.add_argument("--results", default=str(HERE / "results"))
        ap.add_argument("--cases", type=int)
        ap.add_argument("--repeats", type=int, default=D.REPEATS)
        ap.add_argument("--arms", default=",".join(D.ARMS))
    a = ap.parse_args(argv)
    copy = Path(a.copy).resolve() if test else PROJECT / "code-fixed"
    results = Path(a.results).resolve() if test else HERE / "results"
    arms = [x for x in a.arms.split(",") if x] if test else list(D.ARMS)
    repeats = a.repeats if test else D.REPEATS
    run_dir = results / a.run
    if copy.name != "code-fixed":
        raise SystemExit("this experiment runs on code-fixed only")
    if not test and ra.dirty(PINNED + [F + "spend-ledger.jsonl"]):
        raise SystemExit("pinned files have uncommitted changes")
    if not arms or set(arms) - set(D.ARMS) or len(set(arms)) != len(arms):
        raise SystemExit(f"arms must be distinct names from {D.ARMS}")
    suite = suite_cases(a.cases if test else None)
    texts = ra.policies(arms)
    b = rm.bench_args(["--suite", "ours", "--domain", DOMAIN, "--cases", str(len(suite)),
                       "--repeats", str(repeats), "--max-turns", str(CAP_TURNS)])
    held_ledger = lg.lock(LEDGER)
    if held_ledger is None:
        raise SystemExit("another paid process holds the follow-up ledger lock")
    try:
        prior = lg.prior_spend(LEDGER, None if test else D.LEDGER_PREFIX)
    except lg.BudgetStop as exc:
        raise SystemExit(f"REFUSED: {exc}")
    estimate = len(arms) * repeats * sum(D.EST_BY_KIND[c["kind"]] for c in suite)
    hard = D.CAP_USD - prior
    held = rm.lock(copy)
    bad = ["another run holds the lock on this copy"] if held is None else \
        rm.preflight(copy, run_dir, b, os.environ)
    if estimate > hard:
        bad.append(f"estimate ${estimate:.2f} + ledger ${prior:.2f} exceeds the ${D.CAP_USD} cap")
    if (rm.setup_checksum(copy), tree_hash(copy)) != (COPY_CHECKSUM, COPY_TREE):
        bad.append("copy is not the registered code-fixed (rules.COPY_CHECKSUM, COPY_TREE)")
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 2
    lg.debit(LEDGER, a.run, None, 0.0, lg.FREEZE)
    run_dir.mkdir(parents=True)
    (run_dir / "traces").mkdir()
    tree_before = tree_hash(copy)
    want_sha = {arm: sha(t) for arm, t in texts.items()}
    manifest = {
        "run": a.run, "copy": copy.name, "copy_checksum": rm.setup_checksum(copy),
        "copy_tree_hash": tree_before,
        "project_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT,
                                         capture_output=True, text=True).stdout.strip(),
        "command": [sys.executable, __file__, *(argv if argv is not None else sys.argv[1:])],
        "arms": {arm: {"policy_sha256": want_sha[arm]} for arm in arms},
        "suite_sha256": D.sha_file(D.SUITE),
        "models_requested": {"agent": rm.MODEL, "customer": rm.MODEL},
        "temperatures": {"agent": 0, "customer": 0.7}, "cap": CAP_TURNS,
        "cases": len(suite), "repeats": repeats, "concurrency": ra.CONCURRENCY,
        "order": "repeat, then scenario, then arm", "ledger_before_usd": round(prior, 4),
        "estimate_usd": round(estimate, 2), "hard_stop_usd": round(hard, 2), "start": rm.now()}
    mpath = run_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=1))
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() not in
                   (HERE, D.MAIN, PROJECT / "scripts")]
    os.chdir(copy)
    log = (run_dir / "stdout.log").open("w")
    sys.stdout, sys.stderr = rm.Tee(sys.__stdout__, log), rm.Tee(sys.__stderr__, log)
    rows, case_ids, prompts = [], [], collections.defaultdict(set)
    budget = ra.Budget(hard, a.run)
    try:
        spec = importlib.util.spec_from_file_location("bench", copy / "scripts" / "bench.py")
        bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench)
        bench.TRACES = run_dir / "traces"
        import litellm
        import services.user_sim as us
        if Path(us.__file__).resolve() != copy / "services" / "user_sim.py":
            raise SystemExit(f"loaded {us.__file__}, not this copy's")
        sim = us.UserSim()
        manifest["models_resolved"] = {"agent": os.environ.get("DEFAULT_MODEL"),
                                       "customer": sim.model,
                                       "customer_temperature": sim.temperature}
        if manifest["models_resolved"] != {"agent": rm.MODEL, "customer": rm.MODEL,
                                           "customer_temperature": 0.7}:
            raise SystemExit(f"models resolved wrong: {manifest['models_resolved']}")
        cases = journeys(suite, copy)
        case_ids = [c["id"] for c in cases]
        manifest["case_ids"] = case_ids
        mpath.write_text(json.dumps(manifest, indent=1))
        agents = {arm: bench.build_agent(DOMAIN, None if arm == "control" else texts[arm])
                  for arm in arms}
        if any(sha(ag.system_prompt) != want_sha[arm] for arm, ag in agents.items()):
            raise SystemExit("an agent's system prompt is not its policy file")
        Recorder(run_dir / "calls.jsonl", run_dir / "customer_errors.jsonl").install(
            bench, us, litellm)

        def note_prompt(kw):
            if kw.get("tools"):
                prompts[CONV.get()].add(sha(str(kw["messages"][0]["content"])))

        lg.install_guard(litellm, budget, LEDGER, {rm.MODEL}, CONV.get, note_prompt)
        rows = asyncio.run(ra.run_all(bench, agents, cases, repeats, budget))
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        log.close()
        (run_dir / "rows.json").write_text(json.dumps(rows, indent=1, default=str))
        (run_dir / "prompts.json").write_text(json.dumps(
            {k: sorted(v) for k, v in sorted(prompts.items(), key=str)}, indent=1))
        (run_dir / "stopped.json").write_text(json.dumps(sorted(budget.stopped, key=str)))
        s = ra.check(run_dir, arms, repeats, case_ids, prompts, want_sha, budget.stopped)
        s["copy_unchanged"] = tree_hash(copy) == tree_before
        if not s["copy_unchanged"]:
            s["problems"].append("code copy changed during the run")
        if len(case_ids) != len(suite):
            s["problems"].append(f"cases: asked {len(suite)}, ran {len(case_ids)}")
        manifest.update(end=rm.now(), guard_spent_usd=round(budget.spent, 4), summary=s)
        mpath.write_text(json.dumps(manifest, indent=1))
        seal = rm.seal_path(run_dir)
        seal.parent.mkdir(exist_ok=True)
        seal.write_text(rm.seal(run_dir))
        if held is not None:
            held.close()
        held_ledger.close()
    print(json.dumps({k: v for k, v in s.items() if k != "status"}, indent=1))
    return 0 if not s["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
