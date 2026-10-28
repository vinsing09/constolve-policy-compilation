"""An atom must be filed under the fix its own break actually needs.

THE DEFECT, found by asking why our policy gap was DOUBLE tau3's.

    tau3 telecom:  6 of 114 tasks involve the payment chain   (5%)
    ours:         12 of  24 cases                            (50%)

Nine of those twelve had no payment step in the walk at all. They were SIM and signal
walks — `toggle_airplane_mode | reseat_sim_card | reset_apn_settings` — that became
payment cases because of one atom:

    break        <field:bills.status> = "awaiting payment"
    declared fix reseat_sim_card
    fix_sequence [reseat_sim_card, send_payment_request, make_payment]

Reseating a SIM has nothing to do with an unpaid bill. Because the atom is FILED under
reseat_sim_card, every walk containing a reseat step pulled a billing fault into a case
about signal, and graded the payment chain in it.

`prefix_gate.prune` removed the spurious reseat STEP from the solution, which was right
and not enough: the atom was still selected for reseat steps, so the billing FAULT came
with it. The mis-filing had to be fixed where the atom is chosen, not where the solution
is trimmed.

Exactly 1 of telecom's 28 atoms is mis-filed. It is re-filed under the first fix its
break genuinely needs rather than dropped, so reseat cases keep their own atom
(`<field:phone.sim_status>`) and payment cases keep this one.
"""
from __future__ import annotations

import copy


def _apply_break(mod, world, brk) -> bool:
    tool = str((brk or {}).get("tool") or "")
    try:
        if tool.startswith("<field:"):
            coll, field = tool[len("<field:"):-1].split(".", 1)
            target = world[coll]
            (target[0] if isinstance(target, list) else target)[field] = brk["args"]["value"]
        elif tool.startswith("<compound:"):
            return False              # applied differently; not audited here
        else:
            mod.apply(world, tool, **(brk.get("args") or {}))
        mod.derive(world)
        return True
    except Exception:  # noqa: BLE001
        return False


def _fix_name(atom) -> str | None:
    fx = atom.get("fix")
    return fx.get("tool") if isinstance(fx, dict) else fx


def audit(mod, atoms: list[dict]) -> list[dict]:
    """-> [{atom, declared, needed}] for atoms filed under a fix their break lacks."""
    from services import prefix_gate

    out = []
    for atom in atoms:
        declared = _fix_name(atom)
        if not declared:
            continue
        seq = [{"tool": c.get("tool"), "args": c.get("args") or {}}
               for c in (atom.get("fix_sequence") or [atom.get("fix") or {}])
               if isinstance(c, dict) and c.get("tool")]
        if not seq:
            continue
        world = copy.deepcopy(getattr(mod, "SEED", {}) or {})
        try:
            mod.derive(world)
        except Exception:  # noqa: BLE001
            continue
        if not _apply_break(mod, world, atom.get("break")):
            continue
        kept, _ = prefix_gate.prune(mod, world, seq)
        needed = [s["tool"] for s in kept]
        if needed and declared not in needed:
            out.append({"atom": atom, "declared": declared, "needed": needed})
    return out


def refile(mod, atoms: list[dict]) -> tuple[list[dict], list[str]]:
    """-> (atoms with mis-filed ones re-keyed, human-readable notes).

    Re-filed under the FIRST fix the break genuinely needs. Dropping instead would take
    the fault out of the suite entirely; re-filing keeps the coverage and puts it in the
    cases it belongs to.
    """
    problems = audit(mod, atoms)
    if not problems:
        return atoms, []
    by_id = {id(p["atom"]): p for p in problems}
    out, notes = [], []
    for atom in atoms:
        p = by_id.get(id(atom))
        if not p:
            out.append(atom)
            continue
        fixed = copy.deepcopy(atom)
        target = p["needed"][0]
        args = next((c.get("args") or {} for c in (atom.get("fix_sequence") or [])
                     if isinstance(c, dict) and c.get("tool") == target), {})
        fixed["fix"] = {"tool": target, "args": args}
        fixed["fix_sequence"] = [c for c in (atom.get("fix_sequence") or [])
                                 if isinstance(c, dict) and c.get("tool") in p["needed"]]
        out.append(fixed)
        notes.append(f"{(atom.get('break') or {}).get('tool')} re-filed "
                     f"{p['declared']} -> {target} (needs {p['needed']})")
    return out, notes
