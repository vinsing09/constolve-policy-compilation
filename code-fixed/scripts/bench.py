#!/usr/bin/env python3
"""ONE benchmark script. Both suites, one conversation cap, one grader.

WHY THIS EXISTS

tau3's tasks and our tasks were being run by two different scripts with two different
hardcoded conversation caps (25 and 20), and their results were compared to each other
and to tau3's published numbers, which come from a cap of 100. None of the three were
comparable. This runs everything the same way.

WHAT A TEST IS

    a starting condition   the simulated phone/account, broken in specific ways
    what the customer says  their opening line and what they will reveal if asked
    a specific resolution   the exact tool calls that must happen, and by whom

WHAT IS MEASURED, two numbers, both deterministic

    solved fully     every required action happened
    solved partially what fraction of the required actions happened

THE GRADER HAS NO MODEL IN IT. It checks whether specific calls appear in the
transcript. The simulated customer IS a model and varies; the grader does not.

    python3 scripts/bench.py --suite tau3 --repeats 5
    python3 scripts/bench.py --suite ours --repeats 5
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import importlib
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party/tau3-bench/src"))
for _l in (REPO / ".env").read_text().splitlines():
    if "=" in _l and not _l.strip().startswith("#"):
        _k, _, _v = _l.partition("=")
        os.environ.setdefault(_k.strip(), _v.strip())

from models import Agent                                       # noqa: E402
from runner.journey_runner import run_journey                  # noqa: E402
from runner.verdict import graded_score                        # noqa: E402
from services.environments.compiled_env import CompiledEnv     # noqa: E402
from services.environments.tau3_bench_env import Tau3BenchEnv  # noqa: E402
from services.user_sim import UserSim                          # noqa: E402

STUB = ("You are a customer support agent. Help the user with their request "
        "using the tools available to you.")
TRACES = REPO / "logs" / "bench"


def build_agent(domain: str, system_prompt: str | None) -> Agent:
    envmod = importlib.import_module(f"tau2.domains.{domain}.environment")
    env = envmod.get_environment()
    schemas = [(s.get("function") if isinstance(s.get("function"), dict) else s)
               for s in (t.openai_schema for t in env.get_tools())]
    return Agent(id=f"{domain}-under-test", name=domain,
                 system_prompt=(env.get_policy() or "") if system_prompt is None
                 else system_prompt,
                 tool_schemas=schemas, created_at="2026-08-19", kind="multi_turn",
                 tool_execution_mode="serial")


def tau3_cases(domain: str, limit: int) -> list[dict]:
    """tau3's own tasks, graded by OUR criteria: their required actions, by their actor."""
    envmod = importlib.import_module(f"tau2.domains.{domain}.environment")
    tasks = [t.model_dump() for t in envmod.get_tasks()]
    by_n = collections.defaultdict(list)
    for t in tasks:
        by_n[len(t["evaluation_criteria"]["actions"] or [])].append(t)
    picked, keys, i = [], sorted(by_n), 0
    while len(picked) < limit and any(by_n[k] for k in keys):
        k = keys[i % len(keys)]
        if by_n[k]:
            picked.append(by_n[k].pop(0))
        i += 1
    out = []
    for t in picked:
        crits = [{"id": f"a{j}_{a['name']}", "type": "behavior_check",
                  "rule": "tool_called_with",
                  "param": {"tool": a["name"], "arguments": {}, "compare_args": [],
                            "requestor": a.get("requestor") or "assistant"},
                  "description": a["name"]}
                 for j, a in enumerate(t["evaluation_criteria"]["actions"] or [])]
        out.append({"id": t["id"][:110], "kind": "tau3", "task_id": t["id"],
                    "success_criteria": crits, "goal": "", "world_seed": t["id"]})
    return out


def our_cases(domain: str, limit: int) -> list[dict]:
    """Our generated cases, in the shape run_journey expects."""
    suite = json.load(open(REPO / "generated" / f"graph_suite_{domain}.json"))[:limit]
    # ONE BUILDER, so this script and phase15 hand the simulator the same thing. They
    # each had their own copy and each dropped a different field; the two numbers they
    # produced for the identical suite were therefore not comparable.
    from services.suite_bench import journey_of
    out = []
    for i, c in enumerate(suite):
        j = journey_of(c, None, i)
        j["kind"] = "ours"          # routes to CompiledEnv in one(); not the case kind
        j["case_kind"] = c.get("kind") or "repair"
        out.append(j)
    return out


async def one(case: dict, agent: Agent, domain: str, cap: int, label: str,
              rep: int) -> dict:
    if case["kind"] == "tau3":
        env = Tau3BenchEnv(config={"domain": domain, "task_id": case["task_id"]})
        sim = None                       # tau3's env drives its own customer
    else:
        env = CompiledEnv(domain)
        env.set_world((case.get("world_setup") or {}).get("entities") or {})
        sim = UserSim()
    try:
        jr = await run_journey(agent, env, case, user_sim=sim, max_turns=cap)
    except Exception as exc:             # noqa: BLE001
        return {"id": case["id"], "label": label, "rep": rep, "error":
                f"{type(exc).__name__}: {exc}", "solved": False, "k": 0, "n": 0}
    k, n = graded_score(jr.verdict)
    calls = [tc.get("tool_name") for t in jr.history
             for tc in (t.get("tool_calls") or [])]
    TRACES.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() else "_" for ch in case["id"])[:70]
    (TRACES / f"{case['kind']}_{label}_{rep}_{safe}.json").write_text(json.dumps(
        {"case": case["id"], "arm": label, "repeat": rep, "solved": jr.verdict.passed,
         "actions_done": k, "actions_required": n, "turns": len(jr.history),
         "calls": calls,
         "criteria": [{"id": r.criterion_id, "ok": r.passed, "why": r.reason}
                      for r in (jr.verdict.results or [])],
         "history": jr.history}, indent=1, default=str))
    return {"id": case["id"], "label": label, "rep": rep, "error": jr.error,
            "solved": bool(jr.verdict.passed), "k": k, "n": n,
            "turns": len(jr.history), "hit_cap": len(jr.history) >= cap * 2 - 1}


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=("tau3", "ours"), required=True)
    ap.add_argument("--domain", default="telecom")
    ap.add_argument("--cases", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-turns", type=int, default=100)   # tau3 uses 100
    ap.add_argument("--concurrency", type=int, default=8)
    a = ap.parse_args()

    cases = (tau3_cases if a.suite == "tau3" else our_cases)(a.domain, a.cases)
    print("=" * 78)
    print(f"BENCHMARK — {a.suite} suite, {a.domain}")
    print("=" * 78)
    print(f"  cases {len(cases)} | repeats {a.repeats} | conversation cap {a.max_turns}")
    print(f"  required actions per case: "
          f"{dict(sorted(collections.Counter(len(c['success_criteria']) for c in cases).items()))}")
    print(f"  traces -> {TRACES.relative_to(REPO)}/")

    sem = asyncio.Semaphore(a.concurrency)

    async def guarded(f):
        async with sem:
            return await f()

    jobs = []
    for label, prompt in (("with_policy", None), ("stub", STUB)):
        agent = build_agent(a.domain, prompt)
        for rep in range(a.repeats):
            for c in cases:
                jobs.append(guarded(lambda c=c, ag=agent, l=label, r=rep:
                                    one(c, ag, a.domain, a.max_turns, l, r)))
    res = await asyncio.gather(*jobs)

    print(f"\n{'-' * 78}\nRESULT\n{'-' * 78}")
    out = {}
    for label in ("with_policy", "stub"):
        rows = [r for r in res if r["label"] == label and not r.get("error")]
        per = collections.defaultdict(list)
        for r in rows:
            per[r["rep"]].append(r)
        full = [100.0 * sum(1 for x in v if x["solved"]) / max(1, len(v))
                for v in per.values()]
        part = [100.0 * statistics.mean([x["k"] / x["n"] for x in v if x["n"]])
                for v in per.values() if any(x["n"] for x in v)]
        caps = sum(1 for r in rows if r.get("hit_cap"))
        out[label] = {"fully": full, "partly": part,
                      "fully_mean": statistics.mean(full) if full else 0,
                      "partly_mean": statistics.mean(part) if part else 0,
                      "runs": len(rows), "errors": len(rows) - len(rows),
                      "hit_cap": caps}
        print(f"  {label:12s} solved fully  per run {[round(x,1) for x in full]}"
              f"  mean {out[label]['fully_mean']:.1f}%")
        print(f"  {'':12s} solved partly per run {[round(x,1) for x in part]}"
              f"  mean {out[label]['partly_mean']:.1f}%")
        print(f"  {'':12s} runs {len(rows)} | hit the conversation cap {caps}")

    gf = out["with_policy"]["fully_mean"] - out["stub"]["fully_mean"]
    gp = out["with_policy"]["partly_mean"] - out["stub"]["partly_mean"]
    print(f"\n  DIFFERENCE the rulebook makes:")
    print(f"    solved fully   {gf:+.1f} points")
    print(f"    solved partly  {gp:+.1f} points")
    errs = [r for r in res if r.get("error")]
    print(f"\n  errors: {len(errs)}")
    for e in errs[:3]:
        print(f"    {str(e['error'])[:100]}")

    rep = REPO / "reports" / f"bench_{a.suite}_{a.domain}.json"
    rep.write_text(json.dumps({"suite": a.suite, "cases": len(cases),
                               "repeats": a.repeats, "cap": a.max_turns,
                               "arms": out, "gap_fully": round(gf, 1),
                               "gap_partly": round(gp, 1), "runs": res},
                              indent=1, default=str))
    print(f"\nwritten: {rep.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
