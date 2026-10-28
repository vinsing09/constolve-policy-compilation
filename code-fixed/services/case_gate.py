"""A case does not ship unless its own reference solution solves it.

WHAT WAS WRONG

Measured 2026-08-19 on the 45-case telecom graph suite: every graded criterion carried
`"arguments": {}`, and replaying each case's own graded actions against its own world left
38 of 45 still broken. A suite whose reference solution fails is not measuring the agent;
it is measuring nothing, and the deltas it produced (+1.1, -1.1, +1.1, +5.6, -4.4) are the
spread of an instrument wired to nothing.

The data needed to fix both was already present and thrown away. `reports/phase3_atoms_*`
records, for every break it proved:

    break         the call that damages the world
    predicates    exactly which predicates that break makes FALSE
    fix           the repairing call, WITH ITS ARGUMENTS
    fix_sequence  the full ordered repair, for repairs that take more than one call

`scripts/phase15_graph_suite.py` used `at["break"]` and discarded the other three, then
wrote `"arguments": {}` next to an atom that had the arguments in hand.

THE GATE

    a REPAIR case is solvable iff replaying its reference solution on its own world
        raises nothing and leaves the world FULLY healthy
    an ESCALATION case is solvable iff the same replay hands off, and the world still
        resists repair afterwards

Full health rather than "the declared predicates came back", because breaking cellular
service also breaks MMS: a case that repairs what it declared and leaves the cascade
broken is one the customer would still be complaining about. And a separate rule for
escalation, because that world CANNOT be repaired by construction — a locked SIM, a
suspension whose contract expired — so requiring health would reject exactly the shape
tau3 leans on hardest.

This is decidable here and nowhere else, because the world is compiled. Competing tools
mock the backend or hit a live one, so "is this test passable" is an opinion. Here it is a
replay.

WHY REJECTION IS THE RIGHT OUTCOME, NOT REPAIR

A case can be unsolvable because its world carries damage no graded step repairs —
measured: 14 of 45 cases had an outstanding bill that no criterion asked anyone to settle,
because a DIRECT_STATE mutation for a suspension scenario also flipped a bill and nothing
tracked it. Silently adding the missing repair would invent a test the policy graph never
described. Dropping the case is honest, and the count of dropped cases is the measurement
that says how good the generator is.
"""
from __future__ import annotations

import copy
import inspect
import json


# ---- arguments -------------------------------------------------------------

def _bills_for(world: dict, customer_id: str, statuses: tuple) -> list[dict]:
    return [b for b in (world.get("bills") or [])
            if b.get("customer_id") == customer_id
            and str(b.get("status") or "").strip().lower() in statuses]


def resolve_args(mod, tool: str, world: dict, caller: dict) -> dict | None:
    """Arguments for a graded call, resolved from the world rather than guessed.

    Used ONLY for fixes that come from the DIRECT_STATE table, which has no proved atom
    and therefore no recorded arguments. Whenever an atom exists its own `fix["args"]` is
    used verbatim and this function is not consulted: a recorded argument that was proved
    to repair the world beats one reconstructed from a signature.

    Every branch below resolves from state. None invents an identifier, because an
    argument that does not name a record in the world produces a criterion no agent can
    satisfy — the failure mode this module exists to end.
    """
    fn = (getattr(mod, "TOOLS", {}) or {}).get(tool)
    if fn is None:
        return None
    params = list(inspect.signature(fn).parameters)[1:]   # drop `state`
    cid = caller.get("customer_id")
    lines = [l for l in (world.get("lines") or []) if l.get("customer_id") == cid] \
        or (world.get("lines") or [])
    out: dict = {}
    for p in params:
        if p == "customer_id":
            out[p] = cid
        elif p == "line_id":
            out[p] = caller.get("line_id") or (lines[0].get("line_id") if lines else None)
        elif p == "bill_id":
            # the bill the case actually damaged, not simply the first one
            due = _bills_for(world, cid, ("overdue", "awaiting payment", "disputed"))
            out[p] = (due or _bills_for(world, cid, ("paid",)) or [{}])[0].get("bill_id")
        elif p == "summary":
            out[p] = "Escalating: the issue cannot be resolved within agent scope."
        elif p == "gb_amount":
            # the policy caps a refuel at 2GB; the cap is a branch to be tested
            # elsewhere, so the reference solution stays inside it.
            out[p] = 2
        elif p in ("app_name",):
            out[p] = "messaging"
        elif p in ("permission",):
            missing = [a for a in (world.get("app_permissions") or [])
                       if not a.get("granted")
                       and str(a.get("app_name") or "").lower() == "messaging"]
            out[p] = (missing[0].get("permission") if missing else "storage")
        elif p in ("reason",):
            out[p] = "customer request"
        else:
            sig = inspect.signature(fn).parameters[p]
            if sig.default is inspect.Parameter.empty:
                # A required parameter this resolver cannot ground. None, not {}: an
                # empty dict is indistinguishable from a tool that takes no arguments,
                # and conflating the two is how `"arguments": {}` came to sit next to
                # `transfer_to_human_agents`, whose `summary` is required.
                return None
    return out


def missing_required_args(mod, tool: str, args: dict) -> list[str]:
    """Required parameters the criterion fails to supply. Empty means the call is well formed."""
    fn = (getattr(mod, "TOOLS", {}) or {}).get(tool)
    if fn is None:
        return []
    out = []
    for name, p in list(inspect.signature(fn).parameters.items())[1:]:
        if p.default is inspect.Parameter.empty and name not in (args or {}):
            out.append(name)
    return out


# ---- the gate --------------------------------------------------------------

def replay(mod, world: dict, solution: list[dict]) -> tuple[dict, list[str]]:
    """Apply an ordered reference solution to a copy of the world. -> (world, errors)

    Copied, so a caller can gate a case without consuming the world it is gating.
    """
    w = json.loads(json.dumps(world, default=str))
    mod.derive(w)
    errors = []
    for step in solution:
        tool = step.get("tool")
        try:
            mod.apply(w, tool, **(step.get("args") or {}))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{tool}: {type(exc).__name__}: {str(exc)[:120]}")
    return w, errors


def check(mod, case: dict) -> dict:
    """Is this case solvable by the solution it ships with?

    TWO KINDS, AND ONLY ONE OF THEM IS SUPPOSED TO END HEALTHY

      repair      the reference solution must return the world to FULL health. Not
                  merely to "the declared predicates are true": breaking cellular
                  service also makes MMS fail, and a case that repairs its declared
                  predicate while leaving the cascade broken is one the customer would
                  still be complaining about. tau3's own acceptance works this way.
      escalation  the world CANNOT be repaired — a locked SIM, a suspension whose
                  contract has expired — and handing off is the correct answer. Health
                  after the reference solution is the wrong test; the test is that the
                  handoff happened and that the world genuinely resists repair.

    `broken` is what the case DECLARES. Recomputing it here would make the gate agree
    with the case by construction: it would be graded against whatever it happened to
    break rather than against what it claims to test, and collateral damage would
    silently become part of the expected repair.
    """
    world = (case.get("world_setup") or {}).get("entities") or {}
    declared = list((case.get("symptoms") or {}).get("broken") or [])
    solution = list(case.get("reference_solution") or [])
    kind = case.get("kind") or "repair"

    if not world or not declared or not solution:
        return {"solvable": False, "kind": kind,
                "why": "case declares no world, no symptoms or no reference solution",
                "declared_broken": declared, "residual": [], "errors": [],
                "unrepaired": [], "phantom": []}

    start = copy.deepcopy(world)
    mod.derive(start)
    actually_false = {k for k, v in mod.evaluate(start).items() if not v}

    # Declared broken but true in its own world. The case is testing something that is
    # not wrong, so no repair of it can be graded.
    phantom = sorted(set(declared) - actually_false)

    after, errors = replay(mod, world, solution)
    verdicts = mod.evaluate(after)

    residual = sorted(d for d in declared if not verdicts.get(d))
    # Anything still broken that the case never claimed. A CONSEQUENCE of the declared
    # break is fine and common — repairing cellular service repairs MMS with it — so
    # this is measured AFTER the repair, not before. What survives the repair is damage
    # no graded step addresses, and it is what made 14 of 45 cases unsolvable: a
    # DIRECT_STATE mutation for a suspension scenario also flipped a bill, and no
    # criterion asked anyone to settle it.
    unrepaired = sorted(k for k, v in verdicts.items() if not v and k not in declared)

    if kind == "escalation":
        # The world must genuinely resist repair, or escalating is the wrong answer and
        # the case rewards giving up. The declared predicates staying false after the
        # reference solution is the evidence; a case that repairs itself is not an
        # escalation case.
        handed_off = bool(getattr(mod, "OUTCOMES", {})) and any(
            fn(after) for fn in (getattr(mod, "OUTCOMES", {}) or {}).values())
        escalated = handed_off or any(
            s.get("tool") == escalation_tool_of(mod) for s in solution)
        solvable = bool(escalated) and not errors and not phantom and bool(residual)
        return {"solvable": solvable, "kind": kind, "declared_broken": sorted(declared),
                "phantom": phantom, "unrepaired": [], "residual": [],
                "errors": errors,
                "why": "" if solvable else
                       ("escalation case whose world repairs itself" if not residual
                        else "escalation case that never hands off")}

    return {
        "solvable": not residual and not unrepaired and not errors and not phantom,
        "kind": kind,
        "declared_broken": sorted(declared),
        "phantom": phantom,          # declared broken but true in its own world
        "residual": residual,        # declared broken and STILL broken after the repair
        "unrepaired": unrepaired,    # broken after the repair and never declared
        "errors": errors,            # a reference call that raises is not a solution
        "why": "",
    }


def escalation_tool_of(mod) -> str | None:
    """The handoff tool, by the same rule the rest of the pipeline uses."""
    for name in (getattr(mod, "TOOLS", {}) or {}):
        if "transfer" in name and "human" in name:
            return name
    return None


def report(results: list[dict], n_cases: int) -> str:
    """One block, so a generation run says plainly what it rejected and why."""
    def count(key):
        return sum(1 for r in results if r.get(key))
    solvable = sum(1 for r in results if r.get("solvable"))
    lines = [
        f"  cases checked                              {n_cases}",
        f"  SOLVABLE by their own reference solution   {solvable}/{n_cases}",
        f"    rejected: a reference call raises        {count('errors')}",
        f"    rejected: damage no graded step repairs  {count('unrepaired')}",
        f"    rejected: declared broken but not broken {count('phantom')}",
        f"    rejected: still broken after the repair  {count('residual')}",
    ]
    return "\n".join(lines)
