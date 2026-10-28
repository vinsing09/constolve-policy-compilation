"""Thin adapter wrapping τ³-bench (tau2) for journey_runner.

τ³ splits what tau-bench's step combined: env.get_response (tool routing),
UserSimulator.generate_next_message (user turns), evaluate_simulation
(reward, lazy at done). This adapter keeps a τ³-format messages list so
evaluate_simulation() has a valid SimulationRun to score at done.

Forensic logging: every reset/step/user_reply/done/reward-compute appends a
JSONL event under logs/tau3_bench/ for parity-diff investigation.

Config: {"domain": "airline"|"retail"|"banking_knowledge" (required),
         "user_model": "gpt-4o", "user_llm_args": {...},
         "journey_id": "..." (injected by runner)}
"""
from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from services.environments.base import (
    Environment,
    EnvError,
    ToolCall,
    ToolObservation,
    Turn,
    register_env,
)

_ALLOWED_DOMAINS = {"airline", "retail", "banking_knowledge", "telecom"}
_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDOR_SRC = _REPO_ROOT / "third_party" / "tau3-bench" / "src"
_TRACE_DIR = _REPO_ROOT / "logs" / "tau3_bench"

# Cap on consecutive user-side tool-call rounds within a single user_reply().
# Native tau3 caps the whole simulation at max_steps=100; a single user turn
# fanning into >10 sequential tool rounds is pathological and likely a loop.
_MAX_USER_TOOL_ROUNDS = 10


def _lazy_path_inject() -> None:
    if str(_VENDOR_SRC) not in sys.path:
        sys.path.insert(0, str(_VENDOR_SRC))


def _load_domain(domain: str):
    if domain not in _ALLOWED_DOMAINS:
        raise EnvError("env_config", f"unsupported tau3 domain: {domain!r}")
    _lazy_path_inject()
    mod = __import__(f"tau2.domains.{domain}.environment", fromlist=["get_environment", "get_tasks"])
    if domain == "banking_knowledge":
        return mod.get_environment, mod.get_tasks
    return mod.get_environment, lambda: mod.get_tasks(task_split_name="base")


class _TraceLog:
    """Append-only JSONL forensic log per (journey_id, task_id) pair."""

    def __init__(self, journey_id: str, task_id: str):
        _TRACE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_jid = (journey_id or "anon").replace("/", "_")[:40]
        safe_tid = str(task_id).replace("/", "_")[:40]
        self.path = _TRACE_DIR / f"{ts}_{safe_jid}_{safe_tid}.jsonl"
        self._fh = open(self.path, "a")

    def log(self, event: str, **fields: Any) -> None:
        row = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **fields}
        self._fh.write(json.dumps(row, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


@register_env("tau3_bench")
class Tau3BenchEnv(Environment):
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        domain = cfg.get("domain")
        if domain not in _ALLOWED_DOMAINS:
            raise EnvError(
                "env_config",
                f"tau3 domain must be one of {sorted(_ALLOWED_DOMAINS)}, got {domain!r}",
            )
        self._domain = domain
        self._user_model = cfg.get("user_model", "gpt-4o")
        self._user_llm_args = cfg.get("user_llm_args", {"temperature": 0.0})
        self.journey_id = cfg.get("journey_id", "")

        self._env = None
        self._user = None
        self._user_state = None
        self._task = None
        self._messages: list = []
        self._opening: str = ""
        self._opening_consumed = False
        self._done = False
        self._last_reward = 0.0
        self._reward_computed = False
        self._user_actions: list[dict] = []
        self._trace: _TraceLog | None = None
        self._turn = 0  # increments per step / user_reply for forensic ordering

    def _find_task(self, task_id: str):
        _, get_tasks = _load_domain(self._domain)
        tasks = list(get_tasks())
        for t in tasks:
            if t.id == task_id:
                return t
        # Adapter seeds world_seed=str(idx). That matches task.id on airline
        # (`'0'`, `'1'`, …) and retail, but NOT banking_knowledge, whose ids
        # are opaque strings (`'task_001'`, `'task_012'`, …). Fall back to a
        # positional lookup so the numeric seed still resolves to the same
        # task the ingest + native parity used at that index.
        if task_id.isdigit():
            idx = int(task_id)
            if 0 <= idx < len(tasks):
                return tasks[idx]
        raise EnvError(
            "env_config",
            f"task_id {task_id!r} not found in {self._domain} base split",
        )

    async def reset(self, seed_ref: str | None = None) -> None:
        if not seed_ref:
            raise EnvError("env_config", "Tau3BenchEnv.reset requires seed_ref (task_id)")

        def _build():
            _lazy_path_inject()
            from tau2.data_model.message import AssistantMessage
            from tau2.user.user_simulator import UserSimulator

            get_environment, _ = _load_domain(self._domain)
            env = get_environment()
            task = self._find_task(seed_ref)
            init = task.initial_state
            msg_history = (init.message_history if init is not None else []) or []
            if init is not None:
                env.set_state(init.initialization_data, init.initialization_actions, msg_history)
            user = UserSimulator(
                llm=self._user_model,
                instructions=task.user_scenario,
                tools=env.get_user_tools() if env.user_tools is not None else None,
                llm_args=self._user_llm_args,
            )
            user_state = user.get_init_state(message_history=msg_history)
            greeting = AssistantMessage(role="assistant", content="Hi! How can I help you today?", cost=0.0)
            first_user_msg, user_state = user.generate_next_message(greeting, user_state)
            messages = [greeting, first_user_msg]
            return env, task, user, user_state, first_user_msg.content or "", messages

        try:
            (self._env, self._task, self._user, self._user_state,
             self._opening, self._messages) = await asyncio.to_thread(_build)
        except EnvError:
            raise
        except Exception as exc:
            raise EnvError("env_unavailable", f"tau3 env construction failed: {exc}") from exc

        self._opening_consumed = False
        self._done = False
        self._last_reward = 0.0
        self._reward_computed = False
        self._turn = 0
        if self._trace is not None:
            self._trace.close()
        self._trace = _TraceLog(self.journey_id, seed_ref)
        self._trace.log(
            "reset", domain=self._domain, task_id=seed_ref,
            opening_len=len(self._opening), n_tools=len(self._env.get_tools()),
            n_user_tools=(len(self._env.get_user_tools()) if self._env.user_tools else 0),
        )

    async def step(self, tool_call: ToolCall) -> ToolObservation:
        if self._env is None:
            raise EnvError("env_unavailable", "Tau3BenchEnv.step before reset")

        def _do_step():
            from tau2.data_model.message import AssistantMessage, ToolCall as Tau3ToolCall
            call_id = f"call_{self._turn}_{len(self._messages)}"
            tau3_call = Tau3ToolCall(id=call_id, name=tool_call.tool_name, arguments=dict(tool_call.params), requestor="assistant")
            self._messages.append(AssistantMessage(role="assistant", content=None, tool_calls=[tau3_call], cost=0.0))
            tm = self._env.get_response(tau3_call)
            self._messages.append(tm)
            return tm, call_id

        try:
            tm, call_id = await asyncio.to_thread(_do_step)
        except Exception as exc:
            self._safe_trace("step_error", tool=tool_call.tool_name, error=str(exc))
            raise EnvError("tool_failure", f"tau3 get_response raised: {exc}") from exc

        self._turn += 1
        try:
            parsed = json.loads(tm.content) if tm.content else {}
        except (ValueError, TypeError):
            parsed = {"raw": tm.content}

        self._safe_trace(
            "step", turn=self._turn, tool=tool_call.tool_name,
            call_id=call_id, tau3_error=bool(tm.error),
            params_keys=sorted(tool_call.params.keys()) if tool_call.params else [],
        )
        return ToolObservation(
            tool_name=tool_call.tool_name,
            response={"observation": parsed, "error": bool(tm.error)},
            latency_ms=0,
            error=("tau3_tool_error" if tm.error else None),
        )

    async def user_reply(self, history: list[Turn], task: Any) -> str:
        if self._user is None:
            raise EnvError("env_unavailable", "Tau3BenchEnv.user_reply before reset")
        if not self._opening_consumed:
            self._opening_consumed = True
            self._safe_trace("user_reply_opening", opening_len=len(self._opening))
            return self._opening

        last_agent_content = ""
        for h in reversed(history):
            role = h.get("role") if isinstance(h, dict) else getattr(h, "role", None)
            if role == "agent":
                last_agent_content = (
                    h.get("content") if isinstance(h, dict) else h.content
                ) or ""
                break

        def _do_respond():
            from tau2.data_model.message import AssistantMessage, MultiToolMessage
            from tau2.user.user_simulator import UserSimulator
            am = AssistantMessage(role="assistant", content=last_agent_content, cost=0.0)
            self._messages.append(am)
            user_msg, new_state = self._user.generate_next_message(am, self._user_state)
            self._user_state = new_state
            self._messages.append(user_msg)

            # User-side tool-call drain. Mirrors native orchestrator's
            # USER->ENV / ENV->USER alternation: when the user simulator
            # emits tool_calls (banking_knowledge user_tools like
            # apply_for_credit_card, etc.), route each call through
            # env.get_response, append the ToolMessage(s) to the trajectory,
            # then feed them back to user.generate_next_message until the
            # user emits a text message or stops. Without this drain, the
            # user simulator's internal messages list carries unanswered
            # tool_call_ids and the next chat-completion call 400s.
            n_user_tool_rounds = 0
            n_user_tool_calls = 0
            while user_msg.tool_calls:
                n_user_tool_rounds += 1
                if n_user_tool_rounds > _MAX_USER_TOOL_ROUNDS:
                    raise RuntimeError(
                        f"user simulator emitted tool_calls in "
                        f"{n_user_tool_rounds} consecutive rounds without "
                        "a text message — refusing to loop further"
                    )
                tool_results = []
                for tc in user_msg.tool_calls:
                    n_user_tool_calls += 1
                    tm = self._env.get_response(tc)
                    tool_results.append(tm)
                    self._messages.append(tm)
                    # RECORDED FOR GRADING. These were executed and then dropped: the
                    # transcript carried only the user's text, so a grader reading tool
                    # calls saw none of them. 393 of tau3 telecom's 516 required actions
                    # are the USER's, so at most 24% of what had to be graded was
                    # visible — our pass rate read 3.3% against tau3's own 30% on the
                    # identical runs, and the policy-vs-stub delta collapsed to zero.
                    # Same defect as CompiledEnv had this morning.
                    self._user_actions.append({
                        "tool_name": getattr(tc, "name", None) or getattr(tc, "tool_name", ""),
                        "arguments": dict(getattr(tc, "arguments", None) or {}),
                        "actor": "user", "failed": False})
                tool_msg_for_user = (
                    MultiToolMessage(role="tool", tool_messages=tool_results)
                    if len(tool_results) > 1
                    else tool_results[0]
                )
                user_msg, new_state = self._user.generate_next_message(
                    tool_msg_for_user, self._user_state
                )
                self._user_state = new_state
                self._messages.append(user_msg)

            return (
                user_msg.content or "",
                UserSimulator.is_stop(user_msg),
                n_user_tool_rounds,
                n_user_tool_calls,
            )

        try:
            content, stopped, n_user_tool_rounds, n_user_tool_calls = (
                await asyncio.to_thread(_do_respond)
            )
        except Exception as exc:
            self._safe_trace("user_reply_error", error=str(exc))
            raise EnvError("user_sim_error", f"tau3 user sim raised: {exc}") from exc

        self._turn += 1
        self._safe_trace(
            "user_reply", turn=self._turn, stopped=stopped,
            reply_len=len(content), last_agent_len=len(last_agent_content),
            n_user_tool_rounds=n_user_tool_rounds,
            n_user_tool_calls=n_user_tool_calls,
        )
        if stopped:
            self._done = True
            self._safe_trace("done", reason="user_stop_marker")
        return content

    async def get_state(self) -> dict:
        reward = self._last_reward
        if self._done and not self._reward_computed:
            reward = await asyncio.to_thread(self._compute_reward)
            self._last_reward = reward
            self._reward_computed = True
        return {
            "data": (self._env.tools.db.model_dump()
                     if self._env and self._env.tools and self._env.tools.db else {}),
            "reward": reward,
            "task_id": self._task.id if self._task else None,
            "termination_reason": "user_stop" if self._done else None,
        }

    def pop_user_actions(self) -> list[dict]:
        out, self._user_actions = self._user_actions, []
        return out

    def done(self) -> bool:
        return self._done

    def _compute_reward(self) -> float:
        from tau2.data_model.simulation import SimulationRun, TerminationReason
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
        from tau2.orchestrator.modes import CommunicationMode

        now = datetime.now(timezone.utc).isoformat()
        run = SimulationRun(
            id=f"agentops-{self.journey_id or 'anon'}-{self._task.id}",
            task_id=self._task.id,
            trial=0,
            messages=list(self._messages),
            termination_reason=TerminationReason.USER_STOP,
            mode=CommunicationMode.HALF_DUPLEX,
            policy=self._env.get_policy(),
            start_time=now,
            end_time=now,
            duration=0.0,
        )
        info = evaluate_simulation(
            simulation=run, task=self._task,
            evaluation_type=EvaluationType.ALL, solo_mode=False,
            domain=self._domain, mode=CommunicationMode.HALF_DUPLEX,
        )
        reward = float(info.reward)
        self._safe_trace(
            "reward_computed", reward=reward, n_messages=len(self._messages),
            reward_breakdown=(info.reward_breakdown if info.reward_breakdown else {}),
        )
        return reward

    def _safe_trace(self, event: str, **fields: Any) -> None:
        if self._trace is None:
            return
        try:
            self._trace.log(event, **fields)
        except Exception:
            pass
