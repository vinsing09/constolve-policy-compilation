"""Environment ABC + registry for pluggable journey execution.

An Environment is an external system that the agent interacts with through
tool calls + user replies. Implementations:
- StatelessEnv: no-op for agents that don't need an env.
- StubEnv: wraps the current static tool_stubs pattern.
- HttpEnv: plugs into a customer-hosted HTTP env server.
- TauBenchEnv: thin adapter over tau_bench's native envs.

Invariant: every Environment is reset once per journey. Cross-journey state
contamination is the bug τ-bench parity surfaced — make it impossible.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, Callable, Type

from pydantic import BaseModel


class ToolCall(BaseModel):
    tool_name: str
    params: dict


class ToolObservation(BaseModel):
    tool_name: str
    response: dict
    latency_ms: int = 0
    error: str | None = None


class Turn(BaseModel):
    role: Literal["user", "agent", "tool", "system"]
    content: str = ""
    tool_calls: list[ToolCall] = []
    tool_observations: list[ToolObservation] = []


class EnvError(Exception):
    """Typed env-layer failure. journey_runner converts to a failed
    behavior_check criterion; never re-raised past journey boundary."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"[{code}] {message}")


class Environment(ABC):
    """Base class for pluggable envs.

    `isolation_mode` is a class-level invariant, not per-instance state —
    every Environment declares it to be "per_journey" and journey_runner
    enforces env.reset() before each journey.
    """

    isolation_mode: ClassVar[Literal["per_journey"]] = "per_journey"
    type_name: ClassVar[str] = "base"

    @abstractmethod
    async def reset(self, seed_ref: str | None = None) -> None:
        """Reset to a clean per-journey state. Called before every journey."""

    @abstractmethod
    async def step(self, tool_call: ToolCall) -> ToolObservation:
        """Execute one tool call and return the observation."""

    @abstractmethod
    async def user_reply(self, history: list[Turn], task: Any) -> str:
        """Simulated user's next utterance. `task` carries goal/persona/facts."""

    @abstractmethod
    async def get_state(self) -> dict:
        """Return the full world state for world_state_check evaluation."""

    @abstractmethod
    def done(self) -> bool:
        """Has the env signaled the journey is complete?"""

    def pop_user_actions(self) -> list[dict]:
        """Actions the USER performed since the last call, and clear them.

        Grading reads tool calls out of the transcript, and the transcript only ever
        held the AGENT's. τ³ telecom requires a user-side action in 94 of 114 tasks —
        `reset_apn_settings` and `reboot_device` are user-side only — so a criterion
        naming one could never be satisfied no matter what happened in the
        conversation. Measured on our own telecom suite: 42 of 48 criteria, and all 10
        cases, were unpassable by construction, which read as an agent scoring 0%.

        The world applies these actions, so the world is what can report them.
        """
        return []

    def set_user_sim(self, sim: Any) -> None:
        """Seat a user simulator in this environment.

        Loud by default. `run_journey` takes a `user_sim` argument that was assigned
        and never read, so every caller that passed one got the environment's own user
        instead and no error. A silent no-op on the axis a variance decomposition is
        measuring is the worst possible failure, so an environment that cannot seat a
        user says so rather than ignoring it.
        """
        raise EnvError(
            "no_user_slot",
            f"{type(self).__name__} drives its own user; an injected simulator "
            f"would be silently ignored.")


# ── Registry ─────────────────────────────────────────────────────────────────

ENV_REGISTRY: dict[str, Type[Environment]] = {}


def register_env(type_name: str) -> Callable[[Type[Environment]], Type[Environment]]:
    """Decorator to register an Environment subclass under a string type name."""

    def _wrap(cls: Type[Environment]) -> Type[Environment]:
        if not issubclass(cls, Environment):
            raise TypeError(f"{cls.__name__} must subclass Environment")
        cls.type_name = type_name  # type: ignore[misc]
        ENV_REGISTRY[type_name] = cls
        return cls

    return _wrap


def get_env_class(type_name: str) -> Type[Environment]:
    if type_name not in ENV_REGISTRY:
        raise KeyError(f"unknown env type {type_name!r}; registered: {list(ENV_REGISTRY)}")
    return ENV_REGISTRY[type_name]
