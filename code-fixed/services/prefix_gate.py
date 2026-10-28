"""Every required action must be REQUIRED — the world still broken right before it.

WHY

Our solvability gate replays a case's reference solution and checks the world ends
healthy. That permits an overspecified case: if the world is already repaired at step 3
of 7, steps 4-7 are dead weight, and the agent is marked down for skipping work that was
never needed.

tau3 asserts this at build time and we did not (`telecom/tasks/manager.py`,
`verify_task`): it replays each expected action and, BEFORE each one, asserts the world is
still not fixed. That middle assertion is the whole difference.

MEASURED 2026-08-20 on the 24-case telecom suite: 0 overspecified. The check is kept
anyway, because an overspecified case is invisible in a pass rate — it looks exactly like
an agent that gave up early.
"""
from __future__ import annotations

import copy


def healthy(mod, state) -> bool:
    return all(mod.evaluate(state).values())


def check(mod, case: dict) -> dict:
    """-> {"ok", "verdict", "redundant_from", "n"}"""
    world = (case.get("world_setup") or {}).get("entities") or {}
    sol = list(case.get("reference_solution") or [])
    kind = case.get("kind") or "repair"
    if not world or not sol:
        return {"ok": False, "verdict": "no world or no reference solution",
                "redundant_from": None, "n": len(sol)}

    st = copy.deepcopy(world)
    mod.derive(st)
    if healthy(mod, st):
        return {"ok": False, "verdict": "world starts healthy",
                "redundant_from": 0, "n": len(sol)}

    for i, step in enumerate(sol):
        if healthy(mod, st):
            # everything from here on is unnecessary
            return {"ok": False, "verdict": f"redundant from step {i} of {len(sol)}",
                    "redundant_from": i, "n": len(sol)}
        try:
            mod.apply(st, step["tool"], **(step.get("args") or {}))
        except Exception:  # noqa: BLE001
            pass

    if not healthy(mod, st) and kind != "escalation":
        return {"ok": False, "verdict": "never fixed", "redundant_from": None,
                "n": len(sol)}
    return {"ok": True, "verdict": "every step needed", "redundant_from": None,
            "n": len(sol)}


def report(mod, cases: list[dict]) -> str:
    rows = [check(mod, c) for c in cases]
    bad = [(c, r) for c, r in zip(cases, rows) if not r["ok"]]
    out = [f"  cases {len(cases)}   every-step-needed {len(cases) - len(bad)}"
           f"   overspecified or broken {len(bad)}"]
    for c, r in bad[:8]:
        out.append(f"     {r['verdict']:34s} {str(c.get('source_task_id'))[:52]}")
    return "\n".join(out)


def unnecessary_steps(mod, world: dict, solution: list[dict]) -> list[int]:
    """-> indices of steps the repair does not need. Leave-one-out, not prefix.

    WHAT THE PREFIX CHECK ABOVE CANNOT SEE

    It asks whether the world is ALREADY healthy before a step, which finds trailing
    dead weight and nothing else. A useless step in the MIDDLE is invisible to it: the
    world is still broken before that step and still broken after, so the prefix reads
    as necessary the whole way down. It reported 0 overspecified on the 24-case suite
    and this was sitting in it:

        atom  break {"tool": "<field:bills.status>", "value": "awaiting payment"}
              fix_sequence [reseat_sim_card, send_payment_request, make_payment]

    Reseating a SIM has nothing to do with an unpaid bill. The atom miner recorded a
    spurious first step, and every case built from that atom graded `reseat_sim_card`.
    All 7 such cases had sim_status "active", while the policy conditions reseating on
    the SIM showing "Missing" — so a policy-FOLLOWING agent correctly declined and a
    policy-BLIND one reseated anyway, because it is a generic thing to try.

    Measured cost: reseat_sim_card scored policy 32.5 against stub 57.5. MINUS 25, an
    inverted criterion produced entirely by an action nobody should have taken.

    Dropping one step and replaying finds it. Steps are tested one at a time against the
    ORIGINAL solution rather than cumulatively, so two genuinely interchangeable repairs
    are not both discarded.
    """
    if not solution or len(solution) < 2:
        return []
    from services import case_gate

    def repairs(steps) -> bool:
        w, errors = case_gate.replay(mod, world, steps)
        return not errors and healthy(mod, w)

    if not repairs(solution):
        return []                 # not a repair case, or already unsolvable; not ours

    out = []
    for i in range(len(solution)):
        without = [s for j, s in enumerate(solution) if j != i]
        if without and repairs(without):
            out.append(i)
    return out


def prune(mod, world: dict, solution: list[dict]) -> tuple[list[dict], list[str]]:
    """-> (solution with dead steps removed, names of what was removed).

    GREEDY AND RE-VERIFIED, because `unnecessary_steps` tests each step against the
    ORIGINAL solution. Two interchangeable repairs are therefore BOTH reported — each is
    dispensable while the other is present — and removing both at once would break the
    repair. So one goes, the world is replayed, and the question is asked again.
    """
    from services import case_gate

    def repairs(steps) -> bool:
        w, errors = case_gate.replay(mod, world, steps)
        return not errors and healthy(mod, w)

    kept = list(solution)
    dropped: list[str] = []
    if not repairs(kept):
        return kept, []
    while True:
        idx = unnecessary_steps(mod, world, kept)
        if not idx:
            break
        i = idx[0]
        candidate = [s for j, s in enumerate(kept) if j != i]
        if not candidate or not repairs(candidate):
            break
        dropped.append(kept[i].get("tool"))
        kept = candidate
    return kept, dropped
