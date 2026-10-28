"""Thin adapter wrapping tau_bench's native Env for journey_runner.

Tau-bench ships a synchronous `Env` per domain (airline | retail). Its step()
serves both tool calls and user responses — when the agent calls the special
"respond" action, env.step proxies to the user simulator and returns the next
user utterance.

Our Environment ABC splits those concerns: journey_runner calls `step(tool_call)`
for tools, `user_reply(history, task)` for user turns, `done()` for termination.
This adapter bridges the two models:

  reset(seed_ref)            → env.reset(task_index=int(seed_ref)); cache the
                               initial observation to serve as the opening user
                               turn.
  step(ToolCall)             → env.step(Action(tool_call.tool_name, params)).
                               5xx-style failures → EnvError("tool_failure").
  user_reply(history, task)  → take the last agent content, env.step(respond);
                               flip `self._done` when tau-bench says done=True.
  get_state()                → returns {"data": env.data, "reward": last_reward}
                               so world_state_check criteria can assert against
                               the tau-bench domain DB.
  done()                     → True when tau-bench's done flag has flipped.

Config:
  {
    "env_name":      "airline" | "retail",          # required
    "user_strategy": "llm" | "react" | "verify",    # default "llm"
    "user_model":    "gpt-4o",                      # default "gpt-4o"
    "task_split":    "train" | "test",              # default "test"
    "user_provider": "openai" | ...,                # optional
    "task_index":    int,                            # fallback if seed_ref missing
    "journey_id":    "…",                            # injected by orchestrator
  }

Per-journey isolation is guaranteed by constructing a fresh `Env` in reset() —
tau-bench's reset reloads `data` from `data_load_func`, but we rebuild the
whole Env instance anyway to keep adapter state simple.

Blocking-IO note: tau-bench's Env is sync (hits dict-backed DB; user sim
makes LLM calls internally). We wrap step/reset with `asyncio.to_thread` so
the journey_runner's event loop isn't blocked.
"""
from __future__ import annotations

"""DEPRECATED (2026-04-25): wraps tau-bench v0.1.0 whose own README marks
the tasks as broken. Use `tau3_bench` for new work — see
`services/environments/tau3_bench_env.py`. This module is kept only so
historical runs (eval_runs/journey_runs rows created before 2026-04-25) can
still be re-opened without a foreign-key error."""

import asyncio
import logging
from typing import Any

from services.environments.base import (
    Environment,
    EnvError,
    ToolCall,
    ToolObservation,
    Turn,
    register_env,
)

_DEPRECATION_WARNED = False


@register_env("tau_bench")
class TauBenchEnv(Environment):
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        env_name = cfg.get("env_name")
        if env_name not in ("airline", "retail"):
            raise EnvError(
                "env_config",
                f"tau_bench env_name must be 'airline' or 'retail', got {env_name!r}",
            )
        self._env_name = env_name
        self._user_strategy = cfg.get("user_strategy", "llm")
        self._user_model = cfg.get("user_model", "gpt-4o")
        self._task_split = cfg.get("task_split", "test")
        # tau-bench's load_user raises if user_strategy needs a model + the
        # provider is None. Default to "openai" when the strategy actually
        # needs one so existing environment rows (written before we added the
        # field) don't trip env_unavailable on every reset.
        _provider = cfg.get("user_provider")
        if _provider is None and self._user_strategy in ("llm", "react", "verify", "reflection"):
            _provider = "openai"
        self._user_provider = _provider
        self._default_task_index = cfg.get("task_index")
        self.journey_id = cfg.get("journey_id", "")

        self._env = None
        self._done = False
        self._opening: str = ""
        self._opening_consumed = False
        self._last_reward: float = 0.0

    async def reset(self, seed_ref: str | None = None) -> None:
        global _DEPRECATION_WARNED
        if not _DEPRECATION_WARNED:
            logging.getLogger(__name__).warning(
                "tau_bench env is deprecated; use tau3_bench for new work "
                "(see services/environments/tau3_bench_env.py)"
            )
            _DEPRECATION_WARNED = True
        task_index = self._resolve_task_index(seed_ref)

        def _build_and_reset():
            # tau-bench is vendored at <repo>/third_party/tau-bench and is not
            # installed as a package. Prepend it to sys.path on first use so
            # the server process (which doesn't go through scripts/tau_bench_*)
            # can import it.
            import sys
            from pathlib import Path
            _tau_root = Path(__file__).resolve().parents[2] / "third_party" / "tau-bench"
            if str(_tau_root) not in sys.path:
                sys.path.insert(0, str(_tau_root))

            from tau_bench.envs import get_env

            env = get_env(
                env_name=self._env_name,
                user_strategy=self._user_strategy,
                user_model=self._user_model,
                task_split=self._task_split,
                user_provider=self._user_provider,
                task_index=task_index,
            )
            reset_resp = env.reset(task_index=task_index)
            return env, reset_resp.observation

        try:
            self._env, self._opening = await asyncio.to_thread(_build_and_reset)
        except Exception as exc:
            raise EnvError("env_unavailable", f"tau-bench env construction failed: {exc}")

        self._done = False
        self._opening_consumed = False
        self._last_reward = 0.0

    async def step(self, tool_call: ToolCall) -> ToolObservation:
        if self._env is None:
            raise EnvError("env_unavailable", "TauBenchEnv.step() called before reset()")

        def _do_step():
            from tau_bench.types import Action

            return self._env.step(Action(name=tool_call.tool_name, kwargs=dict(tool_call.params)))

        try:
            resp = await asyncio.to_thread(_do_step)
        except Exception as exc:
            raise EnvError("tool_failure", f"tau-bench step raised: {exc}")

        if resp.done:
            self._done = True
            self._last_reward = float(resp.reward or 0.0)

        return ToolObservation(
            tool_name=tool_call.tool_name,
            response={"observation": resp.observation},
            latency_ms=0,
        )

    async def user_reply(self, history: list[Turn], task: Any) -> str:
        if self._env is None:
            raise EnvError("env_unavailable", "TauBenchEnv.user_reply() called before reset()")

        if not self._opening_consumed:
            self._opening_consumed = True
            return self._opening

        last_agent_content = ""
        for h in reversed(history):
            role = h.get("role") if isinstance(h, dict) else getattr(h, "role", None)
            if role == "agent":
                last_agent_content = (h.get("content") if isinstance(h, dict) else h.content) or ""
                break

        def _do_respond():
            from tau_bench.types import Action, RESPOND_ACTION_NAME

            return self._env.step(
                Action(name=RESPOND_ACTION_NAME, kwargs={"content": last_agent_content})
            )

        try:
            resp = await asyncio.to_thread(_do_respond)
        except Exception as exc:
            raise EnvError("user_sim_error", f"tau-bench respond step raised: {exc}")

        if resp.done:
            self._done = True
            self._last_reward = float(resp.reward or 0.0)
        return resp.observation or ""

    async def get_state(self) -> dict:
        if self._env is None:
            return {"reward": self._last_reward}
        return {
            "data": self._env.data,
            "reward": self._last_reward,
            "task_index": getattr(self._env, "task_index", None),
        }

    def done(self) -> bool:
        return self._done

    def _resolve_task_index(self, seed_ref: str | None) -> int:
        if seed_ref is not None:
            try:
                return int(seed_ref)
            except (ValueError, TypeError):
                pass
        if self._default_task_index is not None:
            return int(self._default_task_index)
        return 0
