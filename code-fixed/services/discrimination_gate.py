"""Is this case INFORMATIVE? — the question five gates never asked.

WHY IT WAS MISSING

Every gate we built asks whether a case is WELL-FORMED: the seed gate (is health
reachable), the case gate (does the reference solution repair the world), the prefix
gate (is every step necessary), argument binding (is every criterion callable), mutation
scoring (is a broken solution detected). All five are properties of ONE agent-free
replay, so all five are cheap and all five went inside the generation loop.

Discrimination is not. It is a TWO-agent quantity — how far apart a case pushes a
policy-following agent and a policy-blind one — and criteria are generated from a single
reference solution before any agent exists. So the property the product actually sells
was structurally invisible to the thing generating the tests, and the generator
optimised the only thing it could see. Measured on 2026-08-21, over 200 runs:

    7 of 26 criteria scored NEGATIVE — the policy-blind agent beat the policy-following
    one — and 6 more were dead. 215 of 840 runs sat on criteria that could not move.

WHAT THIS GATE DOES INSTEAD

It does not need a real agent. A criterion that a policy-blind agent can satisfy is
satisfiable by an agent that does LESS, and "less" can be written down directly. Two
adversaries, both deterministic and free:

    null    does nothing at all
    giveup  hands off immediately and does nothing else

Any positive criterion either satisfies is non-discriminating BY CONSTRUCTION, whatever
a real agent would have done. Both defects found on 2026-08-21 fail this gate:

    ordering-as-its-own-criterion   satisfied by `null`   (absence was scored a pass)
    escalation graded on the handoff satisfied by `giveup` (it IS the handoff)

Prohibitions run the mirror adversary — `flail` calls everything — because a forbidden
tool nobody would ever call is unfalsifiable rather than merely easy.
"""
from __future__ import annotations

from runner.verdict import evaluate

ESCALATION_HINTS = ("transfer", "escalat", "handoff", "human")


def escalation_tool(mod) -> str | None:
    """The domain's handoff, by its own tool table rather than a telecom literal."""
    for name in sorted(getattr(mod, "TOOLS", {}) or {}):
        if any(h in name.lower() for h in ESCALATION_HINTS):
            return name
    return None


def _turns(calls: list[tuple[str, str]]) -> list[dict]:
    return [{"role": "assistant",
             "tool_calls": [{"tool_name": t, "arguments": {}, "actor": a}]}
            for t, a in calls]


def adversaries(case: dict, mod) -> dict[str, list[dict]]:
    """Transcripts of agents that do STRICTLY LESS than a competent one."""
    esc = escalation_tool(mod)
    user_tools = set(getattr(mod, "USER_TOOLS", None) or [])

    graded = [(c.get("param") or {}).get("tool") for c in case.get("success_criteria") or []]
    graded = [t for t in graded if t]

    out = {
        # An empty conversation. Nothing a competent agent does is in it.
        "null": [],
        # Hands off on turn one. This is the policy-blind failure mode the stub arm
        # actually exhibits: it escalated in 55.8% of REPAIR runs.
        "giveup": _turns([(esc, "assistant")]) if esc else [],
    }
    # The mirror: an agent that calls everything, in reverse. A prohibition this does not
    # trip is unfalsifiable; an ordering this does not trip is not being enforced.
    out["flail"] = _turns([(t, "user" if t in user_tools else "assistant")
                           for t in reversed(graded)])
    return out


# Which adversary must FAIL a criterion for it to carry any signal at all.
_MUST_FAIL = {"tool_called_with": ("null", "giveup"),
              "tool_called_before": ("null", "giveup"),
              "tool_not_called": ("flail",)}


async def offenders(case: dict, mod) -> list[dict]:
    """-> [{criterion, rule, tool, satisfied_by}, ...]  empty means the case discriminates."""
    advs = adversaries(case, mod)
    crits = case.get("success_criteria") or []
    if not crits:
        return [{"criterion": None, "rule": None, "tool": None, "satisfied_by": "no criteria"}]

    bad = []
    for name, history in advs.items():
        v = await evaluate(criteria=crits, final_state={}, history=history,
                           policy_doc="", journey_id=f"gate:{name}")
        by_id = {r.criterion_id: r.passed for r in v.results}
        for c in crits:
            if name not in _MUST_FAIL.get(c.get("rule"), ()):
                continue
            if by_id.get(c.get("id")):
                bad.append({"criterion": c.get("id"), "rule": c.get("rule"),
                            "tool": (c.get("param") or {}).get("tool"),
                            "satisfied_by": name})
    return bad


async def check(case: dict, mod) -> dict:
    """A verdict for the generator: keep, or say exactly which criteria cannot move."""
    bad = await offenders(case, mod)
    n = len(case.get("success_criteria") or [])
    return {"discriminates": not bad,
            "criteria": n,
            "inert": len(bad),
            "offenders": bad,
            "why": "" if not bad else
                   "; ".join(f"{b['criterion']} satisfied by {b['satisfied_by']}"
                             for b in bad[:4])}
