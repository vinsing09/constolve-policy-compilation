"""When is a policy decision actually live in this world?

WHAT WAS WRONG

`branches_telecom.json` carries 16 policy decisions across 7 kinds. The suite exercised
TWO of them. `live_branches` will only attach a branch whose precondition is true of the
case's world, and 14 of the 16 preconditions were prose:

    {"field": "line.contract_end_date", "value": "in the past"}
    {"field": "...", "value": "exceeds plan data limit"}
    {"field": "...", "value": "below excellent"}

No world ever holds the string "in the past", so every prohibition, ordering, eligibility,
refusal and substitution rule was silently skipped. The extraction was right about the
policy and unusable by the machine, which is the same defect as a criterion with no
arguments: a statement about the world written in a form the world cannot answer.

THE THREE-VALUED RESULT, WHICH IS THE POINT

    True     the world satisfies the condition
    False    the world contradicts it
    None     the world cannot answer

None is not a failure and must never be folded into False. A `before` comparison against
`$now` needs a clock, and the telecom environment bakes its date into the tool bodies
(`env_telecom.py:1215`) rather than carrying one in state. Treating that as False would
silently drop the contract-expiry rule — the single most policy-dependent decision in the
domain — and treating it as True would attach it to worlds where it does not hold, which
is the ungrounded-claim failure `scenario_check` exists to prevent. Reported as unknown,
it shows up as work to do instead of disappearing.
"""
from __future__ import annotations

import datetime

OPS = ("eq", "ne", "in", "not_in", "lt", "lte", "gt", "gte",
       "before", "after", "exists", "absent", "truthy", "falsy")


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip().replace(",", "").lstrip("$"))
    except (TypeError, ValueError):
        return None


def _date(v):
    """A date, or None. Only ISO-ish `YYYY-MM-DD[...]`, because a looser parser would
    read '2' or 'excellent' as a date and manufacture comparisons."""
    s = str(v or "").strip()[:10]
    try:
        return datetime.date(*(int(p) for p in s.split("-")))
    except (TypeError, ValueError):
        return None


def clock(world: dict):
    """The world's own current date, or None if it does not carry one.

    Returning None rather than `date.today()` is deliberate. A clock that follows the
    wall makes the same case pass today and fail tomorrow, and this suite exists to hold
    everything still except the agent.
    """
    meta = (world or {}).get("meta") or {}
    return _date(meta.get("now"))


def evaluate(cond: dict, world: dict, values: list) -> bool | None:
    """Does `cond` hold, given the values found at its field? -> True / False / None

    `values` is supplied by the caller rather than looked up here, so field resolution
    stays in one place — `scenario_check.find_field`, which already knows that a bare
    name can exist in two collections and that resolving to the wrong one manufactures
    contradictions.
    """
    op = str((cond or {}).get("op") or "").strip().lower()
    want = (cond or {}).get("value")

    if op not in OPS:
        return None                       # prose, or an operator we do not implement
    if op == "exists":
        return bool(values)
    if op == "absent":
        return not values
    if not values:
        return None                       # the field is not in this world at all

    if op in ("before", "after"):
        ref = clock(world) if str(want).strip() == "$now" else _date(want)
        if ref is None:
            # No clock in the world, or an unparseable literal. Unknown, never False:
            # see the module docstring.
            return None
        dates = [d for d in (_date(v) for v in values) if d is not None]
        if not dates:
            return None
        return any(d < ref for d in dates) if op == "before" else any(d > ref for d in dates)

    if op in ("lt", "lte", "gt", "gte"):
        ref = _num(want)
        nums = [n for n in (_num(v) for v in values) if n is not None]
        if ref is None or not nums:
            return None
        test = {"lt": lambda n: n < ref, "lte": lambda n: n <= ref,
                "gt": lambda n: n > ref, "gte": lambda n: n >= ref}[op]
        return any(test(n) for n in nums)

    if op in ("in", "not_in"):
        wanted = {str(x).strip().lower() for x in (want or [])}
        if not wanted:
            return None
        hit = any(str(v).strip().lower() in wanted for v in values)
        return hit if op == "in" else not hit

    if op == "truthy":
        return any(bool(v) and str(v).strip().lower() not in ("false", "0", "", "none")
                   for v in values)
    if op == "falsy":
        return all(not bool(v) or str(v).strip().lower() in ("false", "0", "", "none")
                   for v in values)

    hit = any(str(v).strip().lower() == str(want).strip().lower() for v in values)
    return hit if op == "eq" else not hit


def is_typed(cond) -> bool:
    """Can this precondition be evaluated at all?

    A branch that fails this is not attached and not discarded either: it is reported, so
    the count of untyped preconditions is visible rather than showing up as a suite that
    quietly tests two kinds out of seven.
    """
    return (isinstance(cond, dict) and bool(cond.get("field"))
            and str(cond.get("op") or "").strip().lower() in OPS)


def describe(cond: dict) -> str:
    if not isinstance(cond, dict):
        return "none"
    return f"{cond.get('field')} {cond.get('op')} {cond.get('value')!r}"
