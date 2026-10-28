"""No-op environment for stateless agents.

Used when an agent's `kind == stateless` is selected but the journey_runner
path is still the entry point (e.g. for shape-uniformity in testing). All
methods are no-ops; state is empty; done() always returns True so the
runner terminates immediately.
"""
from __future__ import annotations

from typing import Any

from services.environments.base import (
    Environment,
    ToolCall,
    ToolObservation,
    Turn,
    register_env,
)


@register_env("stateless")
class StatelessEnv(Environment):
    def __init__(self, config: dict | None = None):
        self._done = True

    async def reset(self, seed_ref: str | None = None) -> None:
        self._done = True

    async def step(self, tool_call: ToolCall) -> ToolObservation:
        return ToolObservation(
            tool_name=tool_call.tool_name,
            response={"_stateless_noop": True},
            latency_ms=0,
        )

    async def user_reply(self, history: list[Turn], task: Any) -> str:
        return ""

    async def get_state(self) -> dict:
        return {}

    def done(self) -> bool:
        return True
