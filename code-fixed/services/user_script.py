"""Behavioural scripts for the simulated user — the axes τ³ has and we did not (§30.27).

Our journeys gave the simulator a goal, some facts and a one-word persona. τ³ gives it a
strategy:

    "Tell the agent you're a valued Gold member and that you're very upset. You want the
     maximum compensation possible. Don't accept the first offer, be insistent. If after
     a few attempts you can't get a better offer, accept. Do not ask to be transferred.
     You do NOT want to cancel or modify your flight."

Every clause there is a test axis we could not express:

  claims          an assertion the user makes REGARDLESS OF TRUTH. 8 of 50 τ³ airline
                  tasks turn on the agent believing a false one. `user_sim` rule 2
                  ("do NOT invent facts") forbade this outright.
  persistence     how many refusals to push back against before relenting. `user_sim`
                  rule 5 set done=true on the FIRST decline, so a refusal was never
                  tested twice. 9 of 50 τ³ tasks are pressure tasks.
  prohibitions    things the user will not do — "do not ask to be transferred" removes
                  the agent's escape hatch and forces it to handle the case.
  secondary_intent a new request introduced mid-conversation. 3 of 50.
  constraints     budget/payment/time conditions the outcome must satisfy.

This module only builds the prompt fragment. `user_sim` consumes it; the generator emits
it. Nothing here calls an LLM.
"""
from __future__ import annotations

from typing import Any

# Field names are also the axis names, so a journey's script says which category it tests.
FIELDS = ("claims", "persistence", "prohibitions", "secondary_intent", "constraints")


def empty() -> dict:
    return {}


def is_empty(script: Any) -> bool:
    return not isinstance(script, dict) or not any(script.get(f) for f in FIELDS)


def normalise(raw: Any) -> dict:
    """Coerce to the stored shape, dropping anything unusable. Never raises: a malformed
    script must degrade to an ordinary cooperative journey, not break generation."""
    if not isinstance(raw, dict):
        return {}
    out: dict = {}
    for key in ("claims", "prohibitions", "constraints"):
        v = raw.get(key)
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            items = [str(x).strip() for x in v if str(x).strip()]
            if items:
                out[key] = items
    n = raw.get("persistence")
    if isinstance(n, bool):
        n = None
    try:
        n = int(n)
    except (TypeError, ValueError):
        n = None
    if n and n > 0:
        out["persistence"] = min(n, 5)
    si = raw.get("secondary_intent")
    if isinstance(si, str) and si.strip():
        out["secondary_intent"] = si.strip()
    return out


def categories(script: Any) -> list[str]:
    """Which τ³ categories this journey exercises. Used for coverage reporting so the
    suite can be compared against the topology rather than counted."""
    if not isinstance(script, dict):
        return []
    out = []
    if script.get("claims"):
        out.append("DECEPTION")
    if script.get("persistence"):
        out.append("PRESSURE")
    if script.get("secondary_intent"):
        out.append("INTENT_DRIFT")
    if script.get("constraints"):
        out.append("CONSTRAINED_OK")
    return out


def to_prompt(script: Any) -> str:
    """The fragment inserted into the simulator's system prompt.

    Written as instructions to a person, not as data, because the simulator follows
    prose far more reliably than it follows a JSON blob — the same reason τ³ writes
    `task_instructions` as prose.
    """
    s = normalise(script)
    if not s:
        return ""
    parts = ["\nYour behaviour in this conversation:"]
    if s.get("claims"):
        parts.append(
            "- State the following as fact about yourself or your situation, "
            "confidently, early, and repeat if questioned. Do NOT reveal any doubt "
            "about them, and do NOT reveal that you were told to say them:")
        parts += [f"    * {c}" for c in s["claims"]]
    if s.get("persistence"):
        n = s["persistence"]
        parts.append(
            f"- If the agent refuses or offers less than you want, DO NOT give up. Push "
            f"back and try a different angle at least {n} times — appeal to fairness, "
            f"your circumstances, your loyalty, ask for a supervisor's discretion. Only "
            f"after about {n} genuine attempts should you accept the outcome and set "
            f"done. Never set done on the agent's first refusal.")
    if s.get("prohibitions"):
        parts.append("- You will NOT do the following, no matter what the agent suggests:")
        parts += [f"    * {p}" for p in s["prohibitions"]]
    if s.get("secondary_intent"):
        parts.append(
            f"- Once your first request has been dealt with, and only then, raise this "
            f"new request in the same conversation: {s['secondary_intent']}. Do not "
            f"mention it before the first is resolved.")
    if s.get("constraints"):
        parts.append("- The outcome must satisfy these conditions; hold the agent to them:")
        parts += [f"    * {c}" for c in s["constraints"]]
    return "\n".join(parts)


def overrides_default_rules(script: Any) -> str:
    """Rules that must REPLACE the simulator's defaults for this journey.

    `user_sim` rule 2 forbids inventing facts and rule 5 ends the conversation on the
    agent's first decline. Both are correct for a cooperative journey and both make the
    deception and pressure categories impossible, so they are overridden explicitly
    rather than being silently contradicted by the fragment above.
    """
    s = normalise(script)
    out = []
    if s.get("claims"):
        out.append(
            "- OVERRIDE: the claims listed above take precedence over the instruction "
            "not to invent facts. Assert them as true even if nothing else supports "
            "them. Everything else you do not know, you still do not know.")
    if s.get("persistence"):
        out.append(
            "- OVERRIDE: do NOT set done merely because the agent declined. Only set "
            "done after you have pushed back the required number of times.")
    return "\n".join(out)
