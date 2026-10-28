"""A fault the customer cannot describe is a fault the agent cannot be told about.

THE DEFECT, from a transcript on 2026-08-20.

    agent:    "...the next step is to check your network settings, specifically the
               APN (Access Point Name) settings"
    customer: "My APN name is fast.telecom.net and the MMSC URL is
               http://mms.telecom.net/mms/wapenc. I don't see any obvious errors."
    agent:    "The APN name and MMSC URL you listed appear to be correct, so the
               settings themselves may not be the issue."

The case required `reset_apn_settings`. The world WAS broken. What broke it was

    <field:phone.apn_settings_status> = ""

against a seed value of "correct". `check_apn_settings` duly returned an empty string,
and an empty string is not something a person says out loud. The customer described the
two fields that had values and the agent concluded the settings were fine.

5 of the 16 field-faults in telecom are empty-string like this, including
`phone.sim_status`. The tool-call faults do not have the problem — `toggle_airplane_mode`
leaves airplane mode ON, which the status bar states plainly.

WHAT THIS DOES

Substitutes a value a person can read aloud, then PROVES it broke exactly the same
predicates as the empty value did. Same damage, describable. If no candidate reproduces
the damage exactly, the original is kept rather than quietly changing what the case
tests.
"""
from __future__ import annotations

import copy

# Ordinary words a customer would read off a screen. Ordered so the most explicitly
# wrong comes first: "Not set" tells the agent more than "Unknown".
CANDIDATES = ("Not set", "Incorrect", "Unknown", "Invalid", "Missing", "Disabled")

EMPTY = ("", None, "none", "null", "n/a", [], {})


def is_silent(value) -> bool:
    return value in EMPTY or (isinstance(value, str) and not value.strip())


def _broken(mod, world) -> set:
    mod.derive(world)
    return {k for k, v in mod.evaluate(world).items() if not v}


def describable_value(mod, world: dict, apply_path, path: list, value):
    """-> a value a person can say, breaking the SAME predicates. Else `value`.

    `apply_path` is passed in rather than imported so this stays independent of the
    script that owns it.
    """
    if not is_silent(value):
        return value

    try:
        before = _broken(mod, copy.deepcopy(world))
        base = copy.deepcopy(world)
        apply_path(base, path, value)
        want = _broken(mod, base)
    except Exception:  # noqa: BLE001
        return value
    # THE DAMAGE THE BREAK CAUSES, not the total damage in the world. Comparing absolute
    # broken-sets made an irrelevant field look identical to a real fault, so a value was
    # substituted for a field that breaks nothing.
    if want == before:
        return value

    for cand in CANDIDATES:
        trial = copy.deepcopy(world)
        apply_path(trial, path, cand)
        try:
            got = _broken(mod, trial)
        except Exception:  # noqa: BLE001
            continue
        # EXACTLY the same damage. A candidate that breaks MORE would silently enlarge
        # what the case tests; one that breaks less would make it unsolvable-as-graded.
        if got == want:
            return cand
    return value


def _literals(mod, limit: int = 400) -> list[str]:
    """String values the environment itself uses. Its vocabulary, not ours."""
    import inspect
    import re as _re
    try:
        src = inspect.getsource(mod)
    except Exception:  # noqa: BLE001
        return []
    seen, out = set(), []
    for m in _re.finditer(r"[\"']([A-Za-z][A-Za-z /_-]{2,29})[\"']", src):
        v = m.group(1)
        low = v.lower()
        if low in seen:
            continue
        seen.add(low)
        out.append(v)
        if len(out) >= limit:
            break
    return out


def _seed_value(mod, path):
    """The field's healthy value, from the environment's own seed."""
    node = getattr(mod, "SEED", None)
    for p in path:
        if isinstance(node, list):
            node = node[p] if isinstance(p, int) and p < len(node) else (node[0] if node else None)
            continue
        node = (node or {}).get(p) if isinstance(node, dict) else None
        if node is None:
            return None
    return node


def _written(mod, world, apply_path, path, start, repair) -> list:
    """Values the repair WRITES to the field, starting the world at `start`."""
    w = copy.deepcopy(world)
    apply_path(w, path, start)
    try:
        mod.derive(w)
    except Exception:  # noqa: BLE001
        return []

    def read(state):
        node = state
        for p in path:
            node = node[p] if isinstance(node, list) else (node or {}).get(p)
            if node is None:
                return None
        return node

    out, before = [], read(w)
    for tool, args in repair or []:
        if not tool:
            continue
        try:
            mod.apply(w, tool, **(args or {}))
            mod.derive(w)
        except Exception:  # noqa: BLE001
            continue
        now = read(w)
        if str(now).strip().lower() != str(before).strip().lower():
            out.append(now)
        before = now
    return out


def _same(a, b) -> bool:
    return str(a).strip().lower() == str(b).strip().lower()


def sound_fault_value(mod, world: dict, apply_path, path: list, value,
                      repair: list | None = None):
    """A fault value that is describable AND is not a state the repair produces.

    THE DEFECT, from a run20 transcript.

        policy:   "Send the user a payment request for the overdue bill. This will
                   change the status of the bill to AWAITING PAYMENT."
                  "A user can only have one bill in the AWAITING PAYMENT status."
        our atom: <field:bills.status> = "awaiting payment"
        agent:    "Your bill is currently in 'awaiting payment' status, which means a
                   payment request has already been sent and the system will not allow
                   another one to be issued for the same bill."

    The fault IS the state the repair produces, so the repair was already half applied
    and the agent that obeyed the policy was the one marked wrong. All 6 required
    actions missed; the conversation ended at 7 turns.

    NEITHER EXISTING GATE COULD SEE IT. The tool is permissive - the reference solution
    replays without error and the world ends healthy - so case_gate passes it.

    THE TEST IS AN INVOLUTION CHECK, and it took three wrong attempts to find:

        toggle_airplane_mode   healthy False -> writes True;  fault True -> writes False
        send_payment_request   healthy Paid  -> writes Awaiting Payment;
                               fault Awaiting Payment -> writes Awaiting Payment again

    airplane_mode=True is a SOUND fault because the repair moves it back. "awaiting
    payment" is not, because the repair writes the same state and makes no progress.
    So: if the fault is a value the repair writes when started from healthy, it is
    sound only if starting from the fault the repair writes healthy back.
    """
    value = describable_value(mod, world, apply_path, path, value)
    if not repair:
        return value

    base = copy.deepcopy(world)
    apply_path(base, path, value)
    try:
        want = _broken(mod, base)
    except Exception:  # noqa: BLE001
        return value
    if not want:
        return value

    # DOES ANOTHER VALID FAULT'S REPAIR PASS THROUGH THIS ONE?
    #
    # No healthy baseline is usable: the healthy value of bills.status is "Paid", and
    # the repair cannot run from Paid at all, so starting there produces nothing to
    # compare against. Starting from the fault does not work either, because the repair
    # DOES eventually move it (send_payment_request writes Awaiting Payment, then
    # make_payment writes Paid).
    #
    # What identifies the defect is that "awaiting payment" sits IN THE MIDDLE of the
    # repair from a sound fault. Starting at "overdue", the field goes
    # overdue -> Awaiting Payment -> Paid. So a fault equal to Awaiting Payment has the
    # repair already half applied, which is exactly what the agent told the customer:
    # "a payment request has already been sent".
    #
    # airplane_mode=True survives this: no same-damage value's repair passes through
    # True, because the only value that would is the healthy one and that breaks nothing.
    for cand in list(CANDIDATES) + _literals(mod):
        if _same(cand, value):
            continue
        trial = copy.deepcopy(world)
        apply_path(trial, path, cand)
        try:
            if _broken(mod, trial) != want:
                continue                  # not the same fault; not a substitute
        except Exception:  # noqa: BLE001
            continue
        route = _written(mod, world, apply_path, path, cand, repair)
        if any(_same(value, x) for x in route):
            return cand                   # the fault was a waypoint of this repair
    return value
