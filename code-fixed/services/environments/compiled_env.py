"""Run a journey against the COMPILED environment: real Python, zero variance.

WHY THIS EXISTS

Every other environment in this package either invents its answers or is tied to someone
else's benchmark:

    StubEnv        answers from a static per-tool dict, so every journey gets the same
                   reply regardless of what happened
    SimulatedEnv   answers each tool call with a MODEL, and says so in its own docstring:
                   "It is a simulator, not a database ... consistency is not guaranteed"
    Tau3BenchEnv   only accepts tau3 task ids

So a suite generated against our compiled world could not actually be RUN against it. The
determinism result — 120/120 identical on telecom, in-process and cross-process — applied
to replaying known actions, not to a live agent acting. This closes that gap, and it is
the precondition for arm A of the variance decomposition: with the environment provably
still and the user replayed, whatever moves is the agent.

WHAT IT DOES NOT DO

It does not simulate the user. `user_reply` raises, because the user is a separate actor
driven by `services/user_sim.py` or `services/user_replay.py`. Conflating the two is how
"environment" and "simulation" get muddled, and the whole variance decomposition depends
on keeping them apart.
"""
from __future__ import annotations

import importlib
import json
import time
from typing import Any

from services.environments.base import (
    Environment, EnvError, ToolCall, ToolObservation, Turn, register_env,
)


@register_env("compiled")
class CompiledEnv(Environment):
    """A generated `env_<domain>.py` as a journey environment.

    The world comes from the journey's own `world_setup.entities`, which
    `task_adapter` built by EXECUTING the faults. So the state a journey starts in is the
    state its replay proof was computed against, and the two cannot drift.
    """

    type_name = "compiled"

    def __init__(self, domain: str, world: dict | None = None):
        self.domain = domain
        self.mod = importlib.import_module(f"generated.env_{domain}")
        self._initial = world
        self.state: dict = {}
        self._done = False
        self.calls: list[str] = []
        self._user_sim = None
        self._user_actions: list[dict] = []

    def _fresh(self, world: dict | None) -> dict:
        src = world if world is not None else getattr(self.mod, "SEED", {})
        st = json.loads(json.dumps(src, default=str))
        self.mod.derive(st)
        return st

    async def reset(self, seed_ref: str | None = None) -> None:
        """Per-journey isolation. The world is rebuilt from source every time.

        Deep-copied rather than shared, because cross-journey contamination is the bug
        tau-bench parity surfaced and the base class exists to make it impossible.
        """
        self.state = self._fresh(self._initial)
        self._done = False
        self.calls = []
        self._user_actions = []

    def set_world(self, world: dict) -> None:
        """Point at a journey's own starting world before reset."""
        self._initial = world

    async def step(self, tool_call: ToolCall) -> ToolObservation:
        t0 = time.monotonic()
        name = tool_call.tool_name
        if name not in getattr(self.mod, "TOOLS", {}):
            # An unknown tool is the agent's error, not an env failure, and it must reach
            # the agent as a refusal it can recover from rather than killing the journey.
            return ToolObservation(
                tool_name=name, response={},
                latency_ms=int((time.monotonic() - t0) * 1000),
                error=f"no such tool: {name}")
        try:
            result = self.mod.apply(self.state, name, **(tool_call.params or {}))
        except TypeError as exc:
            return ToolObservation(tool_name=name, response={},
                                   latency_ms=int((time.monotonic() - t0) * 1000),
                                   error=f"bad arguments: {exc}")
        except Exception as exc:  # noqa: BLE001
            # A precondition refusal is a legitimate answer from a real system. The agent
            # should see it and adapt, exactly as it would against production.
            return ToolObservation(tool_name=name, response={},
                                   latency_ms=int((time.monotonic() - t0) * 1000),
                                   error=str(exc))
        self.calls.append(name)
        payload = result if isinstance(result, dict) else {"result": result}
        return ToolObservation(tool_name=name, response=payload,
                               latency_ms=int((time.monotonic() - t0) * 1000))

    def set_user_sim(self, sim) -> None:
        self._user_sim = sim

    async def user_reply(self, history: list[Turn], task: Any) -> str:
        """Let the seated user speak, and apply what they DID to the world.

        The environment still does not invent a user — with no simulator seated this
        raises rather than quietly manufacturing one, because a world that answers for
        the person in it is exactly the conflation the variance decomposition exists to
        separate. But the actions a user takes can only be applied by the world, so they
        are applied HERE, the same division SimulatedEnv uses.

        The difference is that SimulatedEnv asks a MODEL what a user action did to the
        world, so its result varies run to run. Here the action goes through the same
        compiled `apply` the agent's tool calls do, so a user-side action is as
        deterministic as an agent-side one. That is what lets arm A hold everything but
        the agent still.
        """
        if self._user_sim is None:
            raise EnvError(
                "not_a_user",
                "CompiledEnv is the world, not the person in it. Seat a simulator with "
                "set_user_sim(), from services/user_sim.py or services/user_replay.py.")
        # THE OBSERVATION LOOP. tau3's user calls a tool, READS THE RESULT, and decides
        # whether to call another, all inside one turn (tau3_bench_env.py:255, `while
        # user_msg.tool_calls`). Ours emitted its message and its actions in one JSON and
        # never saw what they returned, so it acted once and spoke about a world it had
        # not yet observed.
        #
        # Measured before this: 1 call per acting user turn against tau3's 1-6, 4.0
        # actions per conversation against 8.8. And the grounding rule we copied from
        # tau3 — "if a tool call does not return updated status information, you may need
        # to perform another tool call" — was literally unexecutable, because there was
        # no second chance within the turn.
        #
        # Intermediate messages are DISCARDED, as tau3 does: "your messages when
        # performing tool calls will not be displayed to the agent". Only the message
        # from the round that stops acting reaches the agent.
        observations: list[dict] = []
        rounds: list[dict] = []      # what THIS turn has already said and done
        reply = None
        retried = False
        for _ in range(self.MAX_USER_TOOL_ROUNDS):
            reply = await self._user_sim.reply(history=history, task=task,
                                               observations=observations,
                                               rounds=rounds)
            # A CLAIM WITHOUT THE ACTION IS REJECTED ONCE. Read from a transcript: "I
            # just turned off Airplane Mode" with no action emitted, so the world never
            # changed and the agent spent eleven further turns on a fault it had already
            # diagnosed. `toggle_airplane_mode` was missed in 34 of 35 runs; every
            # state-CHANGING user tool showed the same pattern while every read-only
            # check was performed properly.
            from services import action_claim, state_claim
            bogus = action_claim.unsupported_claims(self.mod, reply.message,
                                                    reply.actions or [])
            if bogus and not retried:
                retried = True
                observations = list(observations) + [
                    {"tool": "?", "error": action_claim.correction_note(bogus)}]
                self.calls.append(f"user:claimed-without-acting:{','.join(bogus)}")
                continue

            # A READING THEY NEVER TOOK IS REJECTED THE SAME WAY. Different sentence,
            # different failure: not "I turned it off" but "Airplane mode is off",
            # asserted with no check, on a world where it is ON. The agent replied
            # "thank you for confirming those settings" and never raised it again.
            # 81 of 126 device-state claims across 240 traces were made without ever
            # calling a tool that reports the field.
            observed = {c.split(":")[-1] for c in self.calls if c.startswith("user:")}
            observed |= {a.get("tool") for rd in rounds for a in (rd.get("actions") or [])
                         if isinstance(a, dict)}
            observed |= {a.get("tool") for a in (reply.actions or []) if isinstance(a, dict)}
            unseen = state_claim.unobserved_claims(self.mod, reply.message, observed)
            if unseen and not retried:
                retried = True
                observations = list(observations) + [
                    {"tool": "?", "error": state_claim.correction_note(unseen, self.mod)}]
                self.calls.append(f"user:claimed-unchecked-state:{','.join(unseen)}")
                continue
            if not (reply.actions or []):
                break
            # AN ACTION ALREADY TAKEN THIS TURN IS NOT TAKEN AGAIN. Belt as well as
            # braces: the simulator can now see what it did, and the world refuses a
            # repeat either way. Measured before this: reboot_device four times in one
            # turn, and toggles called an even number of times so the world ended
            # exactly where it started.
            done_already = {(r.get("tool"), json.dumps(r.get("params") or {}, sort_keys=True))
                            for rd in rounds for r in (rd.get("actions") or [])
                            if isinstance(r, dict)}
            fresh, results = [], []
            for act in reply.actions:
                key = (act.get("tool") if isinstance(act, dict) else None,
                       json.dumps((act.get("params") if isinstance(act, dict) else {}) or {},
                                  sort_keys=True))
                if key in done_already:
                    self.calls.append(f"user:repeat-suppressed:{key[0]}")
                    continue
                fresh.append(act)
                results.append(self._apply_user_action(act))
            observations.extend(results)
            rounds.append({"message": reply.message, "actions": list(reply.actions or []),
                           "results": results})
            if reply.done:
                break
            if not fresh:
                break          # nothing new happened; asking again would only repeat
        if reply is None:
            return ""
        if reply.done:
            self._done = True
        return reply.message

    # tau3 bounds its user tool rounds the same way and refuses to loop further; a
    # simulator that only ever calls tools never yields the floor back to the agent.
    MAX_USER_TOOL_ROUNDS = 4

    def _apply_user_action(self, act) -> dict:
        """One user-side tool against the compiled world. -> what the customer SAW.

        The return value used to be discarded. It is the observation the customer needs
        in order to say what happened rather than assert it, which is the whole point of
        the grounding rule.

        Failures are recorded, not raised: a user who tries something and gets nowhere
        is a real turn in a troubleshooting call, and aborting the journey would score
        the agent for the user's mistake.
        """
        get = act.get if isinstance(act, dict) else (lambda k, d=None: getattr(act, k, d))
        name = get("tool")
        if name not in getattr(self.mod, "TOOLS", {}):
            self.calls.append(f"user:{name}:unknown")
            return {"tool": name, "error": "no such tool on your device"}
        try:
            result = self.mod.apply(self.state, name, **(get("params") or {}))
            self.calls.append(f"user:{name}")
            # Recorded in the shape the verdict reads, with the actor kept so a trace
            # can still tell who did what.
            self._user_actions.append({"tool_name": name,
                                       "arguments": get("params") or {},
                                       "actor": "user"})
            return {"tool": name, "result": result}
        except Exception as exc:  # noqa: BLE001
            self.calls.append(f"user:{name}:{type(exc).__name__}")
            # A failure is an observation too — a customer who tries something and is
            # told it cannot be done has learned something and should say so.
            return {"tool": name, "error": f"{type(exc).__name__}: {exc}"}

    def pop_user_actions(self) -> list[dict]:
        out, self._user_actions = self._user_actions, []
        return out

    async def get_state(self) -> dict:
        return json.loads(json.dumps(self.state, default=str))

    async def predicates(self) -> dict:
        """What the world currently satisfies. Used for grading beyond tool calls."""
        return self.mod.evaluate(self.state) if hasattr(self.mod, "evaluate") else {}

    def done(self) -> bool:
        return self._done
