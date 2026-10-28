"""Multi-turn journey runner.

Drives a conversation between a simulated user (env.user_reply), an LLM agent
(litellm w/ tool-calling), and an Environment (env.step routes tool calls).
Isolation is strict: env.reset() is called once per journey, before any other
calls. When the loop ends — user sim says done, agent stops calling tools,
max_turns hit, or a typed EnvError bubbles up — we call verdict.evaluate()
and return a JourneyResult.

Parallelism: `run_journeys` runs multiple journeys with an asyncio.Semaphore
capped at JOURNEY_MAX_PARALLEL (env var, default 5). Each journey gets its
own Environment instance built by `env_factory` — never share env instances.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Awaitable, Callable

import litellm
from pydantic import BaseModel

from models import Agent, Verdict
from services.environments.base import Environment, ToolCall
from services.llm_invariants import assert_tool_calls_resolved
from services.user_sim import UserSim
from runner.verdict import evaluate


class JourneyResult(BaseModel):
    journey_id: str
    agent_id: str
    passed: bool
    verdict: Verdict
    history: list[dict]
    final_state: dict
    turn_count: int
    duration_ms: int
    error: str | None = None


async def run_journey(
    agent: Agent,
    env: Environment,
    journey: dict,
    user_sim: UserSim | None = None,
    max_turns: int = 10,
) -> JourneyResult:
    """Run one journey end-to-end and return the JourneyResult.

    `journey` is a dict-shaped TestGridCase (goal, known_facts, wont_volunteer,
    will_reveal_if_asked, persona_id, world_seed, success_criteria, ...).
    `env` must already be constructed but not reset — we call env.reset first
    so the caller can't accidentally share state across journeys.
    """
    journey_id = str(journey.get("id", ""))
    # A user simulator, when supplied, is SEATED IN THE ENVIRONMENT rather than
    # driven from here, because user-side actions can only be applied by the world.
    # This line previously read `sim = user_sim or UserSim()` and `sim` was never
    # used again: every caller passing a simulator silently got the environment's
    # own instead. Envs that cannot seat one now raise.
    if user_sim is not None:
        env.set_user_sim(user_sim)
    t0 = time.monotonic()
    history: list[dict] = []
    error: str | None = None

    try:
        await env.reset(seed_ref=journey.get("world_seed"))

        # Single growing OpenAI messages list — MUST persist across turns so
        # the LLM sees prior tool_calls + role=tool results. Matches native
        # tau-bench's ToolCallingAgent.solve pattern. Rebuilding per-turn
        # strips tool context and induces hallucinated args on any follow-up
        # turn (see reports/tau_bench_parity/harness_3way_20260424.md trace).
        messages: list[dict] = [{"role": "system", "content": agent.system_prompt}]

        # Prime the conversation with the simulated user's opening turn.
        opening = await env.user_reply(history=[], task=journey)
        # What the user DID goes in the transcript beside what they said, because
        # grading reads tool calls from the transcript and 94 of tau3 telecom's 114
        # tasks require a user-side action. Left out, those criteria are unpassable
        # however well the agent performs.
        opening_acts = env.pop_user_actions()
        if opening or opening_acts:
            turn = {"role": "user", "content": opening}
            if opening_acts:
                turn["tool_calls"] = opening_acts
            history.append(turn)
        if opening:
            messages.append({"role": "user", "content": opening})

        turn = 0
        while turn < max_turns:
            turn += 1
            agent_turn = await _run_agent_turn(agent, env, messages)
            history.append(agent_turn)

            if env.done():
                break

            # After an agent turn (tool calls are drained inside _run_agent_turn),
            # hand the floor to the user sim for the next message.
            reply = await env.user_reply(history=history, task=journey)
            acts = env.pop_user_actions()
            if acts:
                # Recorded even on the turn the user hangs up: they still did it.
                history.append({"role": "user", "content": reply,
                                "tool_calls": acts})
            if env.done():
                history.append({"role": "_user_done", "content": True})
                break
            if not acts:
                history.append({"role": "user", "content": reply})
            messages.append({"role": "user", "content": reply})

        criteria = journey.get("success_criteria") or []
        policy_doc = journey.get("policy_doc") or ""
        final_state = await env.get_state()
        verdict = await evaluate(
            criteria=criteria,
            final_state=final_state,
            history=history,
            policy_doc=policy_doc,
            journey_id=journey_id,
        )
    except Exception as exc:  # env failures, LLM errors — surface as failed verdict
        error = f"{type(exc).__name__}: {exc}"
        final_state = {}
        try:
            final_state = await env.get_state()
        except Exception:
            pass
        verdict = Verdict(
            journey_id=journey_id,
            passed=False,
            results=[],
            history_len=len(history),
            final_state=final_state,
        )

    duration_ms = int((time.monotonic() - t0) * 1000)
    return JourneyResult(
        journey_id=journey_id,
        agent_id=agent.id,
        passed=verdict.passed and error is None,
        verdict=verdict,
        history=history,
        final_state=verdict.final_state or {},
        turn_count=sum(1 for h in history if h.get("role") in ("user", "agent")),
        duration_ms=duration_ms,
        error=error,
    )


def _llm_timeout_s() -> float | None:
    """Per-call LLM timeout from JOURNEY_LLM_TIMEOUT_S (seconds); <=0 disables.

    Default 240s. Added after the 2026-06-05 banking eval: one journey's
    unbounded network call hung for 2h48m and consumed the driver's budget.
    """
    raw = os.getenv("JOURNEY_LLM_TIMEOUT_S", "240")
    try:
        val = float(raw)
    except ValueError:
        return 240.0
    return val if val > 0 else None


async def _run_agent_turn(
    agent: Agent,
    env: Environment,
    messages: list[dict],
) -> dict:
    """One agent turn: call the LLM, execute any tool calls via env.step,
    mutate `messages` in place (growing list owned by run_journey), return
    a single history entry with role='agent'.

    Tool-call loop is bounded — we stop after a max-tool-per-turn cap to
    prevent runaway loops inside one turn.
    """
    tool_schemas = _normalize_tool_schemas(agent.tool_schemas or [])

    tool_calls_made: list[dict] = []
    tool_observations: list[dict] = []
    final_content = ""

    max_tool_iters = int(os.getenv("JOURNEY_MAX_TOOL_ITERS_PER_TURN", "8"))
    for _ in range(max_tool_iters):
        # Fail-fast on the OpenAI 400 invariant we've now hit three times
        # (commits 2c7310c, 010bf31, 4ed8347) — any unmatched tool_calls
        # surfaces here with the bad index, not as opaque litellm 400.
        assert_tool_calls_resolved(messages, context="_run_agent_turn")
        # Honor JOURNEY_LLM_SEED so cross-adapter parity runs (τ³ native vs
        # our journey_runner) can be made byte-identical. Without a seed,
        # gpt-4.1 explores different conversation paths from the same task,
        # which surfaces as 5-10pp ours-vs-native gap on retail.
        _seed_env = os.getenv("JOURNEY_LLM_SEED")
        _llm_kwargs = {
            "model": os.getenv("DEFAULT_MODEL", "claude-sonnet-4-6"),
            "temperature": 0,
            "messages": messages,
            "tools": tool_schemas or None,
            "tool_choice": "auto" if tool_schemas else None,
        }
        if _seed_env is not None and _seed_env.strip():
            try:
                _llm_kwargs["seed"] = int(_seed_env)
            except ValueError:
                pass
        # Bound the call; retry a timeout once, then let run_journey's
        # except record the journey as failed instead of hanging the eval.
        _timeout_s = _llm_timeout_s()
        if _timeout_s:
            _llm_kwargs["timeout"] = _timeout_s
            try:
                response = await litellm.acompletion(**_llm_kwargs)
            except litellm.Timeout:
                response = await litellm.acompletion(**_llm_kwargs)
        else:
            response = await litellm.acompletion(**_llm_kwargs)
        msg = response.choices[0].message
        if msg.tool_calls:
            calls_this_turn = msg.tool_calls
            if getattr(agent, "tool_execution_mode", "parallel") == "serial":
                calls_this_turn = calls_this_turn[:1]
            # The assistant msg we append MUST carry only the tool_calls we
            # will execute — OpenAI's API 400s if a tool_calls array isn't
            # matched 1:1 by subsequent role=tool messages.
            msg_dict = msg.model_dump(exclude_none=False)
            msg_dict["tool_calls"] = [tc.model_dump() for tc in calls_this_turn]
            messages.append(msg_dict)
            for tc in calls_this_turn:
                tool_name = tc.function.name
                try:
                    params = json.loads(tc.function.arguments or "{}")
                except (json.JSONDecodeError, TypeError):
                    params = {}
                obs = await env.step(ToolCall(tool_name=tool_name, params=params))
                # `arguments`, because that is what the verdict reads. This dict was
                # written as `params` and read as `arguments`, so every criterion that
                # matched on argument VALUES was unpassable: 1,907 of 4,114 on telecom,
                # 46%. It is also where policy knowledge shows — which network mode,
                # how much data — so the suite could not grade the part that most
                # needs the policy. `params` is kept as an alias for older readers.
                #
                # A FAILED call is not an action performed. tau3 verifies its graded
                # actions against database state; ours read the transcript, so a call
                # that raised still satisfied a name-only criterion. The escalation
                # tool raises on every world we generate and its criterion passed
                # anyway.
                tool_calls_made.append({"tool_name": tool_name, "arguments": params,
                                        "params": params,
                                        "failed": obs.error is not None})
                tool_observations.append(obs.model_dump())
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(obs.response if obs.error is None else {"error": obs.error}),
                })
            continue
        final_content = msg.content or ""
        # Persist the final assistant text so the next turn's LLM call sees it.
        messages.append({"role": "assistant", "content": final_content})
        break

    return {
        "role": "agent",
        "content": final_content,
        "tool_calls": tool_calls_made,
        "tool_observations": tool_observations,
    }


def _normalize_tool_schemas(schemas: list[dict]) -> list[dict]:
    out = []
    for schema in schemas:
        s = json.loads(json.dumps(schema))
        if "function" not in s:
            s = {
                "type": "function",
                "function": {
                    "name": s.get("name", ""),
                    "description": s.get("description", ""),
                    "parameters": s.get("parameters", {"type": "object", "properties": {}}),
                },
            }
        out.append(s)
    return out


async def run_journeys(
    agent: Agent,
    journeys: list[dict],
    env_factory: Callable[[], Awaitable[Environment]] | Callable[[], Environment],
    max_turns: int = 10,
    max_parallel: int | None = None,
) -> list[JourneyResult]:
    """Run many journeys in parallel with a concurrency cap.

    `env_factory` is a callable (sync or async) that builds a NEW Environment
    per journey — passing a single shared env would violate per_journey
    isolation. Defaults to JOURNEY_MAX_PARALLEL env var, or 5.
    """
    cap = max_parallel if max_parallel is not None else int(os.getenv("JOURNEY_MAX_PARALLEL", "5"))
    sem = asyncio.Semaphore(max(1, cap))

    async def _one(journey: dict) -> JourneyResult:
        async with sem:
            env_or_awaitable: Any = env_factory()
            env = await env_or_awaitable if asyncio.iscoroutine(env_or_awaitable) else env_or_awaitable
            return await run_journey(agent, env, journey, max_turns=max_turns)

    return await asyncio.gather(*(_one(j) for j in journeys))
