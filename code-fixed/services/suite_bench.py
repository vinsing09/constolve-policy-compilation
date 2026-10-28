"""Shared harness for running a compiled suite against an agent.

Extracted so the variance decomposition and the policy-dependence test are the SAME
measurement under different conditions. Two scripts with their own copies of "build the
agent, build the journey, pick the cases" drift, and when they drift the comparison
between their results stops meaning anything.
"""
from __future__ import annotations

import asyncio
import collections
import importlib
import json
from pathlib import Path

from models import Agent
from runner.journey_runner import run_journey
from runner.verdict import graded_score
from services.environments.compiled_env import CompiledEnv

REPO = Path(__file__).resolve().parents[1]

ERRORS: list[str] = []


def agent_under_test(domain: str, system_prompt: str | None = None) -> Agent:
    """The customer's agent: their policy, their tools.

    `system_prompt` overrides the policy, which is the whole mechanism of the
    policy-dependence test: same tools, same suite, policy removed.
    """
    envmod = importlib.import_module(f"tau2.domains.{domain}.environment")
    env = envmod.get_environment()
    schemas = [(s.get("function") if isinstance(s.get("function"), dict) else s)
               for s in (t.openai_schema for t in env.get_tools())]
    return Agent(id=f"{domain}-under-test", name=f"{domain} agent",
                 system_prompt=(env.get_policy() or "") if system_prompt is None
                 else system_prompt,
                 tool_schemas=schemas, created_at="2026-08-18", kind="multi_turn",
                 tool_execution_mode="serial")


def journey_of(case: dict, persona: dict | None, idx: int) -> dict:
    j = {
        "id": f"{idx}:{case.get('source_task_id', '')}"[:120],
        "goal": case.get("goal") or "",
        "known_facts": case.get("known_facts") or {},
        "wont_volunteer": case.get("wont_volunteer") or [],
        "will_reveal_if_asked": case.get("will_reveal_if_asked") or [],
        "success_criteria": case.get("success_criteria") or [],
        "world_setup": case.get("world_setup") or {},
        "world_seed": None,
        # n_atoms lives under known_facts in a narrated case; reading it off the top
        # level returned 0 for every case and silently flattened the depth axis.
        "n_atoms": (case.get("known_facts") or {}).get("n_atoms")
                   or case.get("n_atoms") or len(case.get("breaks") or []),
        # EVERY FIELD THE SIMULATOR READS, IN ONE PLACE. `task_instructions` carries the
        # rule that makes the customer run a check instead of reciting an answer, and it
        # was dropped here while scripts/bench.py passed it — so the two scripts measured
        # the same suite with different customers and their numbers were never
        # comparable (+11.7pp with persona and no grounding, +7.5pp with grounding and no
        # persona). The remaining fields are read by the case and prefix gates.
        "task_instructions": case.get("task_instructions") or "",
        "kind": case.get("kind") or "repair",
        "caller": case.get("caller") or {},
        "symptoms": case.get("symptoms") or {},
        "reference_solution": case.get("reference_solution") or [],
    }
    # ACCEPTANCE TRAVELS WITH THE JOURNEY. A case can carry the customer's decision
    # rules and have them silently dropped here, which is the same class of defect as
    # the persona axis being disconnected: attached, reported, and changing nothing.
    if case.get("acceptance"):
        j["acceptance"] = list(case["acceptance"])
    if persona:
        j["persona_id"] = persona["id"]
        j["persona_fragment"] = persona["system_prompt_fragment"]
    return j


def stratified(cases: list[dict], want: int) -> list[dict]:
    """Even coverage across atom counts, never a prefix.

    A narrated suite is written in ascending depth order, so `[:54]` of 589 covers the
    shallow end and silently drops the deepest cases — raising the score by excluding
    the hardest work and reporting nothing about it.
    """
    bins = collections.defaultdict(list)
    for c in cases:
        bins[(c.get("known_facts") or {}).get("n_atoms")].append(c)
    keys = sorted(bins, key=lambda k: (k is None, k))
    out, i = [], 0
    while len(out) < want and any(bins[k] for k in keys):
        k = keys[i % len(keys)]
        if bins[k]:
            out.append(bins[k].pop(0))
        i += 1
    return out


def graded_count(case: dict) -> int:
    """How many actions must ALL be right. This, not fault count, is the difficulty."""
    return len([c for c in (case.get("success_criteria") or [])
                if c.get("rule") == "tool_called_with"])


def representative(cases: list[dict], want: int) -> list[dict]:
    """A sample that MIRRORS the suite's own difficulty distribution.

    `stratified` spreads evenly across ATOM counts, and atom count is not criteria
    count: a 3-fault case can grade 10 actions. So drawing evenly by atoms out of a
    suite deliberately shaped to tau3's action distribution reintroduced exactly the
    ungradeable deep cases the shaping removed — the measured 45 had a median of 6
    graded actions against the suite's 3, and the generation fix never reached the
    experiment. The pass gap fell from +3.0pp to +1.5pp and it read as a regression.

    Even coverage is right when the question is "how does depth behave". It is wrong
    when the question is "what does this suite score", because then the sample must
    look like the suite.
    """
    bins: dict = collections.defaultdict(list)
    for c in cases:
        bins[graded_count(c)].append(c)
    total = len(cases) or 1
    out = []
    for n in sorted(bins):
        share = max(1, round(want * len(bins[n]) / total))
        out.extend(bins[n][:share])
    return out[:want]


def load_suite(domain: str, cases: int) -> list[dict]:
    path = REPO / "generated" / f"suite_{domain}.json"
    return stratified(json.load(open(path)), cases)


async def gather_bounded(factories, limit: int):
    """Independent journeys, run concurrently and bounded.

    Each builds its own CompiledEnv from its own world, so concurrency changes the wall
    clock and nothing about what is measured.
    """
    sem = asyncio.Semaphore(limit)

    async def run(f):
        async with sem:
            return await f()
    return await asyncio.gather(*[run(f) for f in factories])


TRACE_DIR = None          # set by a caller that wants the conversations kept


async def one_run(agent, domain: str, journey: dict, sim, max_turns: int = 30,
                  trace_tag: str = "") -> dict:
    """-> {"verdict": pass|fail|error, "k": satisfied, "n": graded}

    Both scales, always. `verdict` is the gate everyone quotes and it saturates to zero
    on any case grading more than a couple of actions; k/n is how much of the repair
    actually happened, and it keeps moving where the gate is pinned.

    Errors are RECORDED, never merely counted: a run that fails without saying why gets
    read as a result.
    """
    env = CompiledEnv(domain)
    env.set_world((journey.get("world_setup") or {}).get("entities") or {})
    try:
        jr = await run_journey(agent, env, journey, user_sim=sim, max_turns=max_turns)
    except Exception as exc:  # noqa: BLE001
        ERRORS.append(f"raised {type(exc).__name__}: {exc}")
        return {"verdict": "error", "k": 0, "n": 0}
    if jr.error:
        ERRORS.append(str(jr.error))
        return {"verdict": "error", "k": 0, "n": 0}
    k, n = graded_score(jr.verdict)
    # KEEP THE CONVERSATION when asked. Every graph-suite run so far returned a turn
    # COUNT and discarded the history, so a pass rate could be reported and the reason
    # for it could not be read. 63% of our graded actions are user-side and fail at the
    # same rate in both arms; no outcome number can say whether the agent never
    # instructed the user, or instructed and the simulator ignored it, and those need
    # opposite fixes.
    if TRACE_DIR:
        import json as _j
        from pathlib import Path as _P
        d = _P(TRACE_DIR); d.mkdir(parents=True, exist_ok=True)
        # `id`, not `source_task_id` — journey_of() renames it, so every case wrote to
        # the SAME filename and only the last one of each arm survived. The first
        # "before/after" comparison drawn from these traces was two single cases.
        name = f"{trace_tag or 'run'}_{str(journey.get('id') or journey.get('source_task_id'))[:60]}"
        name = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)[:120]
        (d / f"{name}.json").write_text(_j.dumps({
            "journey": journey.get("source_task_id"),
            "tag": trace_tag,
            "verdict": "pass" if jr.verdict.passed else "fail", "k": k, "n": n,
            "turns": len(jr.history),
            "criteria": [{"id": r.criterion_id, "ok": r.passed, "why": r.reason}
                         for r in (jr.verdict.results or [])],
            "history": jr.history,
        }, indent=1, default=str))
    return {"verdict": "pass" if jr.verdict.passed else "fail", "k": k, "n": n,
            # WHICH criterion, and WHY. A pass rate cannot tell you whether a case
            # failed because the agent skipped a step, called it with wrong arguments,
            # or did something the policy forbids — and those need different fixes.
            "crits": [{"id": r.criterion_id, "ok": r.passed, "why": (r.reason or "")[:90]}
                      for r in (jr.verdict.results or [])],
            "turns": len(jr.history)}


def repair_rate(runs: list[dict]) -> float:
    """Mean fraction of required actions achieved, over runs that graded anything."""
    graded = [r for r in runs if r.get("n")]
    return 100.0 * sum(r["k"] / r["n"] for r in graded) / max(1, len(graded))
