"""Grade an argument only if changing it breaks the repair.

THE DEFECT, measured on 240 traces from 2026-08-20.

    criterion wants  {'customer_id': 'CUS-4501283', 'line_id': 'LN-8830142',
                      'gb_amount': 1}
    agent called     {'customer_id': 'CUS-4501283', 'line_id': 'LN-8830142',
                      'gb_amount': 2}

Both refuels fix the customer's problem. The reference solution happened to record 1,
so the grader made 1 mandatory. 21 of refuel_data's 43 misses are this and nothing else:
the agent did the right thing with a different number.

`compare_args` already exists for free-TEXT arguments — a transfer summary is never
graded literally. This is the same idea for values that are free by CONSEQUENCE rather
than by type: an argument whose value the world does not constrain.

The test is mechanical and needs no model. Perturb one argument, replay the WHOLE
reference solution, and ask whether the world still ends healthy. If it does, that
argument was never load-bearing and grading it only punishes agents for choosing a
different legal value. Same shape as the prefix gate, which asks whether every STEP is
necessary; this asks whether every ARGUMENT is.

An identifier resists perturbation by construction — change the bill_id and the call
raises or repairs nothing — so ids stay graded without being special-cased.
"""
from __future__ import annotations

import copy

from services import case_gate

# Values tried in place of the recorded one. Enough to show the world does not care,
# cheap enough to run for every argument of every graded call.
def _alternatives(value):
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int) and not isinstance(value, bool):
        return [value + 1, value * 2 if value else 1]
    if isinstance(value, float):
        return [value + 1.0, value * 2 if value else 1.0]
    if isinstance(value, str):
        return []          # a string is either an id or free text; both already handled
    return []


def _healthy(mod, world, solution) -> bool:
    w, errors = case_gate.replay(mod, world, solution)
    if errors:
        return False
    return all(mod.evaluate(w).values())


def unconstrained_args(mod, world: dict, solution: list[dict],
                       index: int) -> list[str]:
    """-> arguments of solution[index] the world does not constrain."""
    step = solution[index]
    args = step.get("args") or {}
    if not args or not _healthy(mod, world, solution):
        # Only meaningful for a solution that repairs the world in the first place;
        # an escalation case has nothing to hold constant.
        return []

    free = []
    for name, value in args.items():
        for alt in _alternatives(value):
            trial = copy.deepcopy(solution)
            trial[index]["args"][name] = alt
            if _healthy(mod, world, trial):
                free.append(name)
                break
    return free


def graded_args(mod, world: dict, solution: list[dict], index: int,
                base: list[str] | None) -> list[str] | None:
    """Narrow an existing `compare_args` by dropping what the world does not constrain.

    `base` is what task_adapter.compare_args_for decided on TYPE grounds (None = all).
    Returning a list never widens it — a criterion can only become easier to satisfy in
    a way that is provably fair, never harder.
    """
    args = solution[index].get("args") or {}
    free = set(unconstrained_args(mod, world, solution, index))
    if not free:
        return base
    keep = [k for k in (base if base is not None else list(args)) if k not in free]
    return keep
