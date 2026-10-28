"""HTTP Environment — customer-hosted env speaking our protocol.

Protocol (customer implements):
  POST /reset          Body: {seed_ref}                → 200
  POST /step           Body: {tool_call: {name, args}} → {observation, state_summary}
  POST /user_reply     Body: {history, task}           → {message, done}
  GET  /state                                          → {full_state}

Reliability:
- Per-request timeout (default 15s, override via config.timeout_ms)
- 3 exponential-backoff retries on 5xx / network errors (100 → 400 → 1600ms)
- Circuit breaker: 5 consecutive failures in one journey → EnvError("environment_unavailable")
- Every request tagged with journey_id + turn_idx headers (customer-side debugging)
- Auth: Bearer token from config.token
- All failures surface as typed EnvError, never raw httpx exceptions

config schema:
  {
    "base_url": "https://customer.example.com/env",
    "token": "<opaque>",
    "timeout_ms": 15000,
    "max_retries": 3,
    "max_consecutive_failures": 5,
    "journey_id": "<set per journey>"
  }
"""
from __future__ import annotations

import asyncio
import random
from typing import Any

import httpx

from services.environments.base import (
    EnvError,
    Environment,
    ToolCall,
    ToolObservation,
    Turn,
    register_env,
)


@register_env("http")
class HttpEnv(Environment):
    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.base_url: str = (self.config.get("base_url") or "").rstrip("/")
        if not self.base_url:
            raise EnvError("env_config", "http env requires base_url")
        self.token: str = self.config.get("token", "")
        self.timeout = httpx.Timeout(self.config.get("timeout_ms", 15000) / 1000.0)
        self.max_retries: int = int(self.config.get("max_retries", 3))
        self.max_consec_failures: int = int(self.config.get("max_consecutive_failures", 5))
        self.journey_id: str = self.config.get("journey_id", "")
        self._turn_idx: int = 0
        self._consec_failures: int = 0
        self._done = False
        self._last_state: dict = {}

    # ── Environment contract ──────────────────────────────────────────

    async def reset(self, seed_ref: str | None = None) -> None:
        self._turn_idx = 0
        self._consec_failures = 0
        self._done = False
        self._last_state = {}
        await self._request("POST", "/reset", json={"seed_ref": seed_ref})

    async def step(self, tool_call: ToolCall) -> ToolObservation:
        self._turn_idx += 1
        body = {"tool_call": {"name": tool_call.tool_name, "args": tool_call.params}}
        data = await self._request("POST", "/step", json=body)
        response = data.get("observation") or {}
        if "state_summary" in data:
            self._last_state = {**self._last_state, **(data.get("state_summary") or {})}
        return ToolObservation(
            tool_name=tool_call.tool_name,
            response=response,
            latency_ms=int(data.get("latency_ms", 0)),
        )

    async def user_reply(self, history: list[Turn], task: Any) -> str:
        self._turn_idx += 1
        # Normalize task/history to JSON-safe shapes.
        body = {
            "history": [h.model_dump() if hasattr(h, "model_dump") else h for h in history],
            "task": task.model_dump() if hasattr(task, "model_dump") else task,
        }
        data = await self._request("POST", "/user_reply", json=body)
        if bool(data.get("done", False)):
            self._done = True
        return str(data.get("message", ""))

    async def get_state(self) -> dict:
        data = await self._request("GET", "/state", json=None)
        full = data.get("full_state") or {}
        self._last_state = full
        return full

    def done(self) -> bool:
        return self._done

    # ── Internal HTTP with retry/backoff/circuit-breaker ──────────────

    async def _request(self, method: str, path: str, json: dict | None) -> dict:
        if self._consec_failures >= self.max_consec_failures:
            raise EnvError(
                "environment_unavailable",
                f"circuit open after {self._consec_failures} consecutive failures",
            )

        url = f"{self.base_url}{path}"
        headers = {
            "X-Journey-Id": self.journey_id,
            "X-Turn-Idx": str(self._turn_idx),
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    if method == "GET":
                        resp = await client.get(url, headers=headers)
                    else:
                        resp = await client.request(method, url, json=json, headers=headers)
                if 500 <= resp.status_code < 600:
                    last_exc = EnvError(
                        "env_5xx",
                        f"{method} {path} → {resp.status_code}: {resp.text[:200]}",
                    )
                elif resp.status_code >= 400:
                    # Client errors are not retried — raise immediately.
                    self._consec_failures += 1
                    raise EnvError(
                        "env_client_error",
                        f"{method} {path} → {resp.status_code}: {resp.text[:200]}",
                    )
                else:
                    self._consec_failures = 0
                    try:
                        return resp.json() if resp.content else {}
                    except ValueError:
                        return {}
            except (httpx.RequestError, httpx.TimeoutException) as exc:
                last_exc = EnvError("env_network", f"{method} {path} failed: {exc}")

            if attempt < self.max_retries:
                delay = 0.1 * (4 ** attempt) + random.uniform(0, 0.05)
                await asyncio.sleep(delay)

        self._consec_failures += 1
        if isinstance(last_exc, EnvError):
            raise last_exc
        raise EnvError("env_unknown", f"{method} {path} exhausted retries")
