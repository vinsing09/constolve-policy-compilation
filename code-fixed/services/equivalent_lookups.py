"""Two lookups that reveal the same thing are one requirement.

MEASURED, run21: `get_data_usage` scored policy 0.0 against stub 50.0 — MINUS 50, the
worst criterion in the suite. In 9 of the 10 policy-arm misses the agent had called
`get_details_by_id`, which returns the whole line record including its data usage. It
had the information and was marked wrong for reading it from the other endpoint.

The grader already supports this — `accept = set(param.get("any_of") or [tool])`, added
because "the grading hard-requires get_customer_by_phone, but the policy explicitly
authorizes three equivalent lookups". Nothing ever populated it for generated cases.

THE WORLD IS ASKED, NOT THE SOURCE. Reading field names out of the source finds nothing
here: `get_details_by_id` returns `dict(customer)`, naming no field at all. So both
tools are CALLED on the actual world and their outputs compared as flat key -> value
maps. That is ground truth by construction, the same choice `observable_values` made
after three attempts at parsing failed.

Only tools that write nothing are considered. A tool that changes the world is never
interchangeable with one that does not, however similar the output.
"""
from __future__ import annotations

import copy
import inspect
import json


def _flat(value, prefix: str = "") -> dict:
    out = {}
    if isinstance(value, dict):
        for k, v in value.items():
            out.update(_flat(v, f"{prefix}{k}."))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            out.update(_flat(v, f"{prefix}{i}."))
    else:
        out[prefix.rstrip(".")] = str(value)
    return out


def _calls(mod, name, world, caller) -> list:
    """Every output this tool can produce for THIS caller. Usually one.

    A generic endpoint takes an opaque `id` that resolve_args cannot ground - which is
    why the first version found no equivalents at all: `get_details_by_id` was never
    callable, so it could not be compared. The caller's own identifiers are tried, and
    an agent that called it with any of them has the information.
    """
    fn = (getattr(mod, "TOOLS", {}) or {}).get(name)
    if fn is None:
        return []
    params = [p for p in inspect.signature(fn).parameters if p not in ("state", "_world")]
    from services import case_gate
    base = case_gate.resolve_args(mod, name, world, caller) or {} if params else {}

    variants = [base]
    missing = [p for p in params if p not in base]
    if missing == ["id"]:
        variants = [dict(base, id=v) for v in
                    (caller.get("line_id"), caller.get("customer_id")) if v]
    elif missing:
        return []

    out = []
    for args in variants:
        try:
            out.append(fn(json.loads(json.dumps(copy.deepcopy(world), default=str)), **args))
        except Exception:  # noqa: BLE001
            continue
    return out


def equivalents(mod, tool: str, world: dict, caller: dict,
                min_facts: int = 1) -> list[str]:
    """-> read-only tools whose output CONTAINS everything `tool` reveals here.

    A superset, not an overlap: an agent that called one of these has learned at least
    what the graded call would have told it. `min_facts` stops a tool that reveals
    almost nothing from matching everything.
    """
    writes = getattr(mod, "WRITES", {}) or {}
    tools = getattr(mod, "TOOLS", {}) or {}
    if tool not in tools or writes.get(tool):
        return [tool]
    got_mine = _calls(mod, tool, world, caller)
    here = _flat(got_mine[0] if got_mine else {})

    # WHAT MATTERS IS THE ABNORMALITY, NOT EVERY FIELD. A strict superset finds nothing:
    # get_data_usage also reports plan_name and the data limit, which get_details_by_id
    # does not carry, so the tool that DOES reveal the fault fails a total-containment
    # test. The facts that differ from a healthy world are the diagnostic ones, and a
    # lookup that reveals those has told the agent what the graded call would have.
    healthy = _flat((_calls(mod, tool, getattr(mod, "SEED", {}) or {}, caller) or [{}])[0])
    mine = {k: v for k, v in here.items()
            if v not in ("", "None", "null") and healthy.get(k) != v}
    if len(mine) < min_facts:
        return [tool]

    out = {tool}
    for name in tools:
        if name == tool or writes.get(name):
            continue
        for got_raw in _calls(mod, name, world, caller):
            got = _flat(got_raw)
            # compare on the leaf name: the same fact is nested differently by
            # different endpoints, "data_usage" vs "line.data_usage".
            leaf = {k.split(".")[-1]: v for k, v in got.items()}
            if all(leaf.get(k.split(".")[-1]) == v for k, v in mine.items()):
                out.add(name)
                break
    return sorted(out)
