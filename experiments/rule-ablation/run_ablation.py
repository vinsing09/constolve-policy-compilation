#!/usr/bin/env python3
"""Run the control and rule-removed arms on code-fixed with bench's own build_agent and one().

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/rule-ablation/run_ablation.py \
        --run main --cases 24 --repeats 3

Same agent, customer, models, temperatures, cases, cap (30) and concurrency (8) as
full-fixed-2; only the agent's system prompt differs between arms (trace.md). Writes
results/<run>/ (manifest.json, calls.jsonl, customer_errors.jsonl, rows.json, traces/,
stdout.log) and seals/<run>.sha256. Spend over all runs in results/ is capped at CAP_USD.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import fcntl
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import run_manifest as rm  # noqa: E402
from runlog import CONV, RETURNED_OK, Recorder, conv_key, read_jsonl, tree_hash  # noqa: E402
from rules import ARMS, CONTROL_SHA, COPY_CHECKSUM, COPY_TREE, sha  # noqa: E402

CAP_USD = 60.0
LEDGER = HERE / "spend-ledger.jsonl"   # every debit of every paid run; not settable from the CLI
MAX_OUT = 32768             # gpt-4.1's output-token limit: the most one call can return
WORST_CALL = 0.36           # gate reserve per call slot: ~45k chars in (policy, tools, history) + MAX_OUT
EST_PER_CONV = 0.1147  # full-fixed-2 with_policy: $13.77 over 120 conversations
NEXT_CONV = 0.20       # full-fixed-2's dearest with_policy conversation cost $0.2001
IN_FLIGHT = 0.10       # what an already-started conversation may still spend
DOMAIN, CAP_TURNS, CONCURRENCY = "telecom", 30, 8
PINNED = ["experiments/rule-ablation/run_ablation.py", "experiments/rule-ablation/rules.py",
          "experiments/rule-ablation/variants.json", "experiments/rule-ablation/ties.json",
          "experiments/rule-ablation/policies", "experiments/rule-ablation/score_ablation.py",
          "experiments/rule-ablation/prereg.md", "scripts/run_manifest.py", "scripts/runlog.py",
          "scripts/score.py", "scripts/checksum-fixed.txt", "scripts/tree-fixed.txt"]


class BudgetStop(RuntimeError):
    pass


class Budget:
    """spent + reserved never passes hard: each call reserves its worst case first."""

    def __init__(self, hard: float, run: str = ""):
        self.hard, self.spent, self.reserved, self.inflight = hard, 0.0, 0.0, 0
        self.run = run
        self.stopped = set()

    def may_start(self) -> bool:
        return (self.spent + self.reserved + self.inflight * IN_FLIGHT + NEXT_CONV
                + CONCURRENCY * WORST_CALL) <= self.hard


def worst_cost(kw: dict, price_in: float, price_out: float) -> float:
    """Upper bound on one call's cost: every character a token, maximum output."""
    chars = len(json.dumps(kw.get("messages"), default=str)) + \
        len(json.dumps(kw.get("tools"), default=str))
    return chars * price_in + MAX_OUT * price_out


def prior_spend() -> float:
    return sum(float(x["usd"]) for x in read_jsonl(LEDGER))


def ledger_lock():
    """One paid run at a time over the whole experiment, held from the spend read to the end."""
    f = LEDGER.with_suffix(".lock").open("a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    return f


def dirty(paths: list[str]) -> bool:
    return bool(subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=PROJECT,
                               capture_output=True, text=True).stdout.strip())


def debit(run: str, conv, usd: float, kind: str) -> None:
    with LEDGER.open("a") as f:
        f.write(json.dumps({"run": run, "conv": conv, "usd": usd, "kind": kind}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def policies(arms: list[str]) -> dict[str, str]:
    meta = json.loads((HERE / "variants.json").read_text())
    out = {}
    for arm in arms:
        text = (HERE / meta[arm]["file"]).read_text()
        if sha(text) != meta[arm]["sha256"]:
            raise SystemExit(f"{arm}: policy file does not match variants.json")
        out[arm] = text
    if "control" in out and sha(out["control"]) != CONTROL_SHA:
        raise SystemExit("control policy is not the full-fixed-2 policy")
    return out


def install_guard(litellm, budget: Budget, prompts: dict) -> None:
    """Outside the recorder: reserve each call's worst cost or refuse it, add up actual
    cost (the worst case if unknown or failed), note each agent call's prompt."""
    inner = litellm.acompletion
    price = litellm.model_cost[rm.MODEL]
    p_in, p_out = price["input_cost_per_token"], price["output_cost_per_token"]

    async def guarded(*a, **kw):
        if kw.get("tools"):
            prompts[CONV.get()].add(sha(str(kw["messages"][0]["content"])))
        worst = worst_cost(kw, p_in, p_out)
        if budget.spent + budget.reserved + worst > budget.hard:
            budget.stopped.add(CONV.get())
            raise BudgetStop(f"spent ${budget.spent:.2f} + reserved ${budget.reserved:.2f}")
        budget.reserved += worst
        debit(budget.run, CONV.get(), worst, "reserve_worst_case")   # survives a crash
        cost, kind = worst, "failed_call_worst_case"
        try:
            resp = await inner(*a, **kw)
            try:
                cost, kind = float(litellm.completion_cost(completion_response=resp)), "call"
            except Exception:  # noqa: BLE001
                kind = "call_cost_unknown_worst_case"
            return resp
        finally:
            budget.reserved -= worst
            budget.spent += cost
            debit(budget.run, CONV.get(), cost - worst, f"settle_{kind}")

    litellm.acompletion = guarded


async def run_all(bench, agents: dict, cases: list, repeats: int, budget: Budget) -> list:
    sem = asyncio.Semaphore(CONCURRENCY)

    async def job(case, arm, rep):
        async with sem:
            if not budget.may_start():
                return {"id": case["id"], "label": arm, "rep": rep, "skipped": True}
            budget.inflight += 1
            try:
                return await bench.one(case, agents[arm], DOMAIN, CAP_TURNS, arm, rep)
            finally:
                budget.inflight -= 1

    return await asyncio.gather(*[job(c, arm, r) for r in range(repeats) for c in cases
                                  for arm in agents])


def check(run_dir: Path, arms: list, repeats: int, case_ids: list, prompts: dict,
          want_sha: dict, stopped: set) -> dict:
    """Classify every conversation and list anything that would make the run not count."""
    rows = json.loads((run_dir / "rows.json").read_text())
    calls = read_jsonl(run_dir / "calls.jsonl")
    cust = {e["conv"] for e in read_jsonl(run_dir / "customer_errors.jsonl")}
    failed = {c["conv"] for c in calls if not c["ok"]}
    traces, p = {}, []
    for f in sorted((run_dir / "traces").glob("*.json")):
        d = json.loads(f.read_text())
        k = conv_key(d["arm"], d["repeat"], d["case"])
        if k in traces:
            p.append(f"duplicate trace for {k}: {traces[k]}, {f.name}")
        traces[k] = f.name
    expected = {conv_key(a, r, c) for a in arms for r in range(repeats) for c in case_ids}
    status = {}
    for r in rows:
        k = conv_key(r["label"], r["rep"], r["id"])
        status[k] = ("skipped" if r.get("skipped") else "budget_stopped" if k in stopped
                     else "error" if r.get("error") or
                     k in failed or k in cust or k not in traces else "clean")
        if status[k] == "clean" and prompts.get(k) != {want_sha[r["label"]]}:
            status[k] = "wrong_prompt"
            p.append(f"{k}: agent prompts {sorted(prompts.get(k, []))}")
    if set(status) != expected or len(rows) != len(expected):
        p.append("result rows are not exactly the expected conversations")
    if set(traces) - {k for k, s in status.items() if s not in ("skipped", "budget_stopped")}:
        p.append("trace files for conversations that did not run")
    ok = [c for c in calls if c["ok"]]
    if {c["conv"] for c in calls} - set(status):
        p.append("model calls from unknown conversations")
    if {str(c["requested_model"]) for c in calls} - {rm.MODEL}:
        p.append("a call requested another model")
    if not all(RETURNED_OK.match(str(c.get("returned_model"))) for c in ok):
        p.append("a call returned another model")
    temps = {(c["caller"], c["temperature"]) for c in calls}
    if temps - {("agent", 0), ("customer", 0.7)}:
        p.append(f"callers/temperatures {sorted(map(str, temps))}")
    if sum(not rm.usable(c) for c in ok):
        p.append("calls missing usage or cost")
    count = collections.Counter(status.values())
    blocks = {(r, c) for r in range(repeats) for c in case_ids}
    valid = {(r, c) for r, c in blocks
             if all(status.get(conv_key(a, r, c)) == "clean" for a in arms)}
    bad_blocks = {(r, c) for r, c in blocks - valid
                  if any(status.get(conv_key(a, r, c)) not in ("clean", "skipped",
                                                                "budget_stopped") for a in arms)}
    if not valid:
        p.append("no valid blocks")
    if len(bad_blocks) > 0.1 * len(blocks):
        p.append(f"{len(bad_blocks)} of {len(blocks)} blocks lost to failures (limit 10%)")
    return {"conversations": dict(count), "blocks": len(blocks), "valid_blocks": len(valid),
            "blocks_lost_to_failures": len(bad_blocks), "model_calls": len(calls),
            "failed_calls": len(calls) - len(ok),
            "sdk_retries": sum(c["sdk_retries"] for c in calls),
            "cost_usd": round(sum(c["cost_usd"] for c in ok if rm.usable(c)), 4),
            "cost_by_arm": {a: round(sum(c["cost_usd"] for c in ok if rm.usable(c) and
                                         c["conv"].split("|")[0] == a), 4) for a in arms},
            "status": status, "problems": p}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--cases", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--arms", default=",".join(ARMS))
    test = os.environ.get("RULE_ABLATION_TEST") == "1"     # fake-model tests only
    if test:
        ap.add_argument("--copy", default=str(PROJECT / "code-fixed"))
        ap.add_argument("--results", default=str(HERE / "results"))
    a = ap.parse_args(argv)
    copy = Path(a.copy).resolve() if test else PROJECT / "code-fixed"
    results = Path(a.results).resolve() if test else HERE / "results"
    run_dir, arms = results / a.run, [x for x in a.arms.split(",") if x]
    if copy.name != "code-fixed":
        raise SystemExit("this experiment runs on code-fixed only")
    if not test and dirty(PINNED):
        raise SystemExit("pinned experiment files have uncommitted changes")
    if not arms or set(arms) - set(ARMS) or len(set(arms)) != len(arms):
        raise SystemExit(f"arms must be distinct names from {ARMS}")
    texts = policies(arms)
    b = rm.bench_args(["--suite", "ours", "--domain", DOMAIN, "--cases", str(a.cases),
                       "--repeats", str(a.repeats), "--max-turns", str(CAP_TURNS)])
    ledger = ledger_lock()
    if ledger is None:
        raise SystemExit("another paid run of this experiment holds the ledger lock")
    prior = prior_spend()
    estimate = len(arms) * a.cases * a.repeats * EST_PER_CONV
    hard = CAP_USD - prior
    held = rm.lock(copy)
    bad = ["another run holds the lock on this copy"] if held is None else \
        rm.preflight(copy, run_dir, b, os.environ)
    if estimate > hard:
        bad.append(f"estimate ${estimate:.2f} + ledger ${prior:.2f} exceeds the ${CAP_USD} cap")
    if (rm.setup_checksum(copy), tree_hash(copy)) != (COPY_CHECKSUM, COPY_TREE):
        bad.append("copy is not the registered code-fixed (rules.COPY_CHECKSUM, COPY_TREE)")
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 2
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
        "models_requested": {"agent": rm.MODEL, "customer": rm.MODEL},
        "temperatures": {"agent": 0, "customer": 0.7}, "cap": CAP_TURNS,
        "cases": a.cases, "repeats": a.repeats, "concurrency": CONCURRENCY,
        "order": "repeat, then scenario, then arm", "ledger_before_usd": round(prior, 4),
        "estimate_usd": round(estimate, 2), "hard_stop_usd": round(hard, 2), "start": rm.now()}
    mpath = run_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=1))
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() not in (HERE, PROJECT / "scripts")]
    os.chdir(copy)
    log = (run_dir / "stdout.log").open("w")
    sys.stdout, sys.stderr = rm.Tee(sys.__stdout__, log), rm.Tee(sys.__stderr__, log)
    rows, case_ids, prompts = [], [], collections.defaultdict(set)
    budget = Budget(hard, a.run)
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
        cases = bench.our_cases(DOMAIN, a.cases)
        case_ids = [c["id"] for c in cases]
        manifest["case_ids"] = case_ids
        mpath.write_text(json.dumps(manifest, indent=1))
        if len(case_ids) != a.cases:
            raise SystemExit(f"asked for {a.cases} cases, the suite gives {len(case_ids)}")
        agents = {arm: bench.build_agent(DOMAIN, None if arm == "control" else texts[arm])
                  for arm in arms}
        if any(sha(ag.system_prompt) != want_sha[arm] for arm, ag in agents.items()):
            raise SystemExit("an agent's system prompt is not its policy file")
        Recorder(run_dir / "calls.jsonl", run_dir / "customer_errors.jsonl").install(
            bench, us, litellm)
        install_guard(litellm, budget, prompts)
        rows = asyncio.run(run_all(bench, agents, cases, a.repeats, budget))
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        log.close()
        (run_dir / "rows.json").write_text(json.dumps(rows, indent=1, default=str))
        (run_dir / "prompts.json").write_text(json.dumps(
            {k: sorted(v) for k, v in sorted(prompts.items(), key=str)}, indent=1))
        (run_dir / "stopped.json").write_text(json.dumps(sorted(budget.stopped, key=str)))
        s = check(run_dir, arms, a.repeats, case_ids, prompts, want_sha, budget.stopped)
        s["copy_unchanged"] = tree_hash(copy) == tree_before
        if not s["copy_unchanged"]:
            s["problems"].append("code copy changed during the run")
        if len(case_ids) != a.cases:
            s["problems"].append(f"cases: asked {a.cases}, ran {len(case_ids)}")
        manifest.update(end=rm.now(), guard_spent_usd=round(budget.spent, 4), summary=s)
        mpath.write_text(json.dumps(manifest, indent=1))
        seal = rm.seal_path(run_dir)
        seal.parent.mkdir(exist_ok=True)
        seal.write_text(rm.seal(run_dir))
        held.close()
        ledger.close()
    print(json.dumps({k: v for k, v in s.items() if k != "status"}, indent=1))
    return 0 if not s["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
