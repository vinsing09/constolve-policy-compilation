"""Stub environment — wraps today's static tool_stubs pattern.

config schema:
  {
    "tool_stubs": { "<tool_name>": {"response": {...}, "latency_ms": int,
                                    "simulate_failure": bool} },
    "user_sim": {...}  # optional — passed through to UserSim
  }

Journey model: step() looks up the tool by name, returns the canned response.
user_reply() delegates to a UserSim instance (lazily constructed) so we get
persona-aware simulated users without each env re-implementing the LLM loop.
done() flips when either:
  - history contains a "_user_done" marker,
  - or the runner hits its max-turn cap (journey_runner enforces; env trusts).
"""
from __future__ import annotations

from typing import Any

from services.environments.base import (
    Environment,
    ToolCall,
    ToolObservation,
    Turn,
    register_env,
    EnvError,
)


@register_env("stub")
class StubEnv(Environment):
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.tool_stubs: dict[str, dict] = self.config.get("tool_stubs", {}) or {}
        self._state: dict = {}
        self._done = False
        self._user_sim = None  # lazy

    async def reset(self, seed_ref: str | None = None) -> None:
        self._state = {"seed_ref": seed_ref, "tool_calls": []}
        self._done = False

    async def step(self, tool_call: ToolCall) -> ToolObservation:
        stub = self.tool_stubs.get(tool_call.tool_name)
        if stub is None:
            return ToolObservation(
                tool_name=tool_call.tool_name,
                response={},
                error=f"no_stub_for_{tool_call.tool_name}",
            )
        if stub.get("simulate_failure"):
            raise EnvError("tool_failure", f"simulated failure in {tool_call.tool_name}")
        self._state.setdefault("tool_calls", []).append(tool_call.tool_name)
        return ToolObservation(
            tool_name=tool_call.tool_name,
            response=stub.get("response", {}),
            latency_ms=int(stub.get("latency_ms") or 0),
        )

    def set_user_sim(self, sim) -> None:
        self._user_sim = sim

    async def user_reply(self, history: list[Turn], task: Any) -> str:
        if self._user_sim is None:
            from services.user_sim import UserSim  # lazy to avoid import cycle
            self._user_sim = UserSim()
        reply = await self._user_sim.reply(history=history, task=task)
        if reply.done:
            self._done = True
        return reply.message

    async def get_state(self) -> dict:
        return dict(self._state)

    def done(self) -> bool:
        return self._done
