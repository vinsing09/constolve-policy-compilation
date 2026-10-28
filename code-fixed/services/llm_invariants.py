"""Shape invariants for OpenAI-format chat-completion messages.

Three separate fixes this week (commits 2c7310c, 010bf31, 4ed8347) all addressed
the same OpenAI 400: "An assistant message with 'tool_calls' must be followed
by tool messages." Each was at a different message-list builder. To keep a
fourth instance from shipping silently, every code path that constructs a
messages list for chat-completion should call `assert_tool_calls_resolved`
before sending.

This module validates list shape only. The tau³ user-simulator bug (commit
4ed8347) was a different shape — call-ordering on a list we don't own — and
has its own structural defense (drain loop + cap) inside
`Tau3BenchEnv.user_reply`.
"""
from __future__ import annotations

from typing import Any


class ToolCallsUnresolvedError(ValueError):
    """Raised when a chat-completion messages list violates the OpenAI
    contract that every `assistant.tool_calls[i].id` must be matched by a
    subsequent `role=tool` message with the same `tool_call_id`."""


def _tool_call_id(tc: Any) -> str | None:
    if isinstance(tc, dict):
        return tc.get("id")
    return getattr(tc, "id", None)


def assert_tool_calls_resolved(
    messages: list[dict], *, context: str = ""
) -> None:
    """Validate a chat-completion messages list before sending.

    For every assistant message carrying `tool_calls`, the immediately
    following messages (until the next non-tool role) must be `role=tool`
    messages whose `tool_call_id`s exactly cover the assistant's
    `tool_call.id`s — no missing, no extra, no duplicate.

    `context` is included in error messages to make the call site obvious.
    """
    label = context or "messages"
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i] or {}
        role = m.get("role")

        if role == "assistant":
            tool_calls = m.get("tool_calls") or []
            if not tool_calls:
                i += 1
                continue

            expected: list[str] = []
            for j, tc in enumerate(tool_calls):
                tcid = _tool_call_id(tc)
                if not tcid:
                    raise ToolCallsUnresolvedError(
                        f"{label}: assistant message at index {i} has "
                        f"tool_calls[{j}] with missing or empty `id`"
                    )
                expected.append(tcid)

            actual: list[str] = []
            k = i + 1
            while k < n and (messages[k] or {}).get("role") == "tool":
                tm = messages[k] or {}
                tcid = tm.get("tool_call_id")
                if not tcid:
                    raise ToolCallsUnresolvedError(
                        f"{label}: tool message at index {k} has missing "
                        f"or empty `tool_call_id` "
                        f"(responding to assistant at index {i})"
                    )
                actual.append(tcid)
                k += 1

            if sorted(actual) != sorted(expected):
                missing = sorted(set(expected) - set(actual))
                extra = sorted(set(actual) - set(expected))
                dup_expected = len(expected) != len(set(expected))
                dup_actual = len(actual) != len(set(actual))
                detail: list[str] = []
                if missing:
                    detail.append(f"missing tool responses for {missing}")
                if extra:
                    detail.append(f"unexpected tool_call_ids {extra}")
                if dup_expected:
                    detail.append("duplicate ids in assistant.tool_calls")
                if dup_actual and not extra and not missing:
                    detail.append("duplicate tool responses")
                raise ToolCallsUnresolvedError(
                    f"{label}: assistant.tool_calls at index {i} expects "
                    f"{sorted(expected)} but got {sorted(actual)} "
                    f"({'; '.join(detail) or 'count mismatch'})"
                )

            i = k
            continue

        if role == "tool":
            raise ToolCallsUnresolvedError(
                f"{label}: orphan tool message at index {i} "
                f"(no preceding assistant message with tool_calls)"
            )

        i += 1
