#!/usr/bin/env python3
"""Build cases by WALKING the policy graph, then grade policy agent vs stubbed agent.

WHY THIS REPLACES THE COMPOSER

Our composer takes the product over all atoms: 21,606 distinct repair sets against tau3's
95, two thirds of cases mixing unrelated symptom families in one call. The policy is a
troubleshooting manual organised BY FAMILY, so it has nothing to say about those
situations, and measured, it changed almost nothing about what the agent did.

A walk is a subset of ONE family's steps, kept in the order the policy prescribes. Every
case therefore sits on a path the policy actually describes.

TWO THINGS THE ATOM APPROACH COULD NOT REACH

Atoms come from perturbing TOOLS, so we only ever generated faults some tool causes. The
policy names states no tool produces:

    SIM "Locked"                 reseat_sim_card explicitly leaves it locked
    contract end date in the past resume_line refuses; the agent must escalate

Both exist in the compiled world (`phone.sim_status`, `lines[].contract_end_date`) and
both are set directly here. They are the escalation cases — 18% of tau3's suite and
0.2% of ours — and they are where the policy decides the answer.

    python3 scripts/phase15_graph_suite.py --build --cases 24
    python3 scripts/phase15_graph_suite.py --grade --cases 24 --repeats 2
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import importlib
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third_party/tau3-bench/src"))
for _l in (REPO / ".env").read_text().splitlines():
    if "=" in _l and not _l.strip().startswith("#"):
        _k, _, _v = _l.partition("=")
        os.environ.setdefault(_k.strip(), _v.strip())

from services import case_gate, meta_user, narrator, policy_graph  # noqa: E402
from services import suite_bench as B, task_adapter  # noqa: E402
from services.task_adapter import escalation_tool  # noqa: E402
from services import discrimination_gate  # noqa: E402
from services import describable_faults  # noqa: E402
from services import arg_gate  # noqa: E402
from services import prefix_gate  # noqa: E402
from services import equivalent_lookups  # noqa: E402
from services import atom_audit  # noqa: E402
from services.task_adapter import (  # noqa: E402
    readable_observables, state_field_names)
from services.user_sim import UserSim                               # noqa: E402

STUB = ("You are a customer support agent. Help the user with their request "
        "using the tools available to you.")

# tau3 telecom's own depth profile, MEASURED from reports/bench_tau3_telecom.json
# (30 tasks, 5 repeats, 300 runs): required graded actions per task, mean 5.04, with a
# tail to 11 where its solve rate is 0%. Used as the target shape for selection so the
# axis our suite is compared on is chosen rather than left to iteration order.
TAU3_DEPTH = {1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 1, 8: 2, 9: 1, 10: 2, 11: 1}

# tau3 telecom's mix of task kinds. Escalation is 18% of its suite and was 0.2% of our
# old composer's, which is the gap this generator was built to close.
TAU3_KIND = {"repair": 0.82, "escalation": 0.18}

# States the policy names that no tool of ours produces. Written here rather than
# discovered, and deliberately few: each one is a condition the policy explicitly
# describes and the compiled environment already models.
DIRECT_STATE = {
    # The billing flow. No tool of ours makes a bill overdue, so send_payment_request,
    # make_payment and the suspension they unblock were unreachable by any case — while
    # tau3 tests exactly that chain as `overdue_bill_suspension`. The compiled tools test
    # for "overdue" and "awaiting payment"; the seed's bills are all Paid.
    # EVERY SPEC DECLARES ITS OWN REPAIR. A mutation that damages the world without
    # saying how the damage is undone produced the largest single class of unsolvable
    # case: 14 of 45 carried an outstanding bill that no criterion asked anyone to
    # settle, because the suspension spec below also flips a bill and nothing tracked
    # it. `repair` is ordered, and it is the ONLY source of graded criteria for a
    # direct-state step, so a case cannot grade an action its own solution never takes.
    # THE SUSPENSION IS WHAT MAKES THE BILL DISCOVERABLE. These two specs used to set
    # the bill alone, leaving the line Active — so service worked, the customer had no
    # symptom, and the agent's own lookup reported "your line is active and not
    # suspended". 8 of 14 payment cases in the 2026-08-20 suite were built that way,
    # and send_payment_request was NEVER ASKED FOR in 96 of its 109 misses.
    #
    # The policy's billing flow is triggered by suspension, which is why the
    # `resume_line` spec below already couples them and calls it "the policy's own
    # chain". tau3 never separates them either: its task is `overdue_bill_suspension`.
    "send_payment_request": [
        {"why": "a bill is overdue and the line has been suspended for it",
         "path": ["bills", 0, "status"], "value": "Overdue",
         "also": [(["lines", 0, "status"], "Suspended")],
         "repair": ["send_payment_request", "make_payment", "resume_line"]},
    ],
    "make_payment": [
        {"why": "a bill is overdue and the line has been suspended for it",
         "path": ["bills", 0, "status"], "value": "Overdue",
         "also": [(["lines", 0, "status"], "Suspended")],
         "repair": ["send_payment_request", "make_payment", "resume_line"]},
    ],
    "resume_line": [
        {"why": "the line is suspended for an overdue bill",
         "path": ["lines", 0, "status"], "value": "Suspended",
         "also": [(["bills", 0, "status"], "Overdue")],
         # the policy's own chain: pay before the suspension can be lifted
         "repair": ["send_payment_request", "make_payment", "resume_line"]},
    ],
    # UNREPAIRABLE BY CONSTRUCTION. The only correct action is to hand off, so these
    # carry kind="escalation" and are graded on the handoff alone. Judging them by "the
    # world ends healthy" would reject the shape tau3 leans on hardest, and grading a
    # repair alongside the handoff would punish an agent that correctly recognised the
    # limit.
    "transfer_to_human_agents": [
        {"why": "SIM is locked after too many wrong PIN entries",
         "path": ["phone", "sim_status"], "value": "Locked",
         "repair": ["transfer_to_human_agents"], "kind": "escalation"},
        {"why": "line is suspended and its contract has already ended",
         "path": ["lines", 0, "status"], "value": "Suspended",
         "also": [(["lines", 0, "contract_end_date"], "2024-11-30")],
         "repair": ["transfer_to_human_agents"], "kind": "escalation"},
    ],
}


def apply_path(world, path, value):
    node = world
    for p in path[:-1]:
        node = node[p]
    node[path[-1]] = value


PLAIN = {
    "phone.signal_strength": "the signal bars at the top of your phone read {v}",
    "phone.network_type": "your phone shows it is on {v}",
    "phone.download_speed": "an internet speed check comes back {v}",
    "phone.connection_quality": "your connection quality reads {v}",
    "phone.can_send_mms": "a picture message actually sends",
}


def acceptance_from_world(mod, world: dict) -> list[str]:
    """What the customer must SEE before agreeing it is fixed.

    Read from a transcript: a case composed of an MMS fault over mobile-data and service
    faults was narrated as "my photos will not send", the customer accepted the moment a
    photo sent, and the two mobile-data steps were graded as failures for actions the
    conversation gave nobody a reason to take. The customer stopped early because their
    stated complaint was met while the world was still broken underneath.

    tau3 solves this with an acceptance standard tied to a check the user can run —
    "resolved only when the speed test returns excellent, and nothing else". Ours must
    cover EVERY fault the case grades, or the grading asks for more than the scenario
    justifies and both arms fail it equally.

    Derived by comparing what the customer can perceive now against the healthy world,
    so it needs no model and cannot drift from the case.
    """
    import json as _json
    from services.scenario_check import field_vocabulary
    healthy = _json.loads(_json.dumps(getattr(mod, "SEED", {}) or {}, default=str))
    try:
        mod.derive(healthy)
    except Exception:  # noqa: BLE001
        return []
    hv, cv = field_vocabulary(healthy, limit=4), field_vocabulary(world, limit=4)
    lines = []
    for obs in (getattr(mod, "OBSERVABLES", None) or []):
        h = (hv.get(obs) or [None])[0]
        c = (cv.get(obs) or [None])[0]
        if h is None or c is None or str(h) == str(c):
            continue
        tmpl = PLAIN.get(obs)
        if tmpl:
            lines.append("You do not agree the problem is fixed until "
                         + tmpl.format(v=h) + ".")
    return lines


def build_case(walk: dict, mod, atoms: list[dict], idx: int) -> dict | None:
    """A case, or None with the reason recorded on `build_case.rejected`.

    Rejections are ACCUMULATED rather than counted, because "skipped 300" says nothing
    about whether the generator is healthy. Which reason, and how often, is the number
    that says what to fix next.
    """
    user_side = set(getattr(mod, "USER_TOOLS", None) or [])
    by_fix = collections.defaultdict(list)
    for at in atoms:
        by_fix[at["fix"]["tool"]].append(at)

    world = json.loads(json.dumps(mod.SEED, default=str))
    mod.derive(world)
    direct_used = []
    # THE REFERENCE SOLUTION, accumulated as the world is damaged.
    #
    # Every break records how it is undone, at the moment it is applied, from the atom
    # that proved it or from the DIRECT_STATE spec that caused it. Before this the
    # criteria were written from `step["fix"]` with `"arguments": {}` while the atom
    # sitting in `by_fix` held the proved arguments and the full `fix_sequence`. 191 of
    # 191 criteria shipped with no arguments, and `transfer_to_human_agents` was graded
    # with none at all although its `summary` is required, so that criterion could not be
    # executed by anyone.
    solution: list[dict] = []
    ordered_pairs: list[tuple] = []
    kind = "repair"
    for step in walk["steps"]:
        fix = step["fix"]
        if by_fix.get(fix):
            brk = by_fix[fix][0]["break"]
            shell = {"breaks": [dict(brk)]}
            # applied onto the world we are accumulating, not a fresh seed
            tool = brk.get("tool") or ""
            try:
                if tool.startswith("<field:"):
                    coll, field = tool[len("<field:"):-1].split(".", 1)
                    tgt = world[coll]
                    # A FAULT THE CUSTOMER CANNOT DESCRIBE CANNOT REACH THE AGENT.
                    # 5 of telecom's 16 field-faults set a field to "", including
                    # apn_settings_status and sim_status. Measured consequence, from a
                    # 2026-08-20 transcript: the customer read out the two APN fields
                    # that HAD values, said "I don't see any obvious errors", and the
                    # agent concluded the settings were fine on a case that required
                    # reset_apn_settings. The substitute is proved to break exactly the
                    # same predicates before it is used.
                    _path = ([coll, 0, field] if isinstance(tgt, list) else [coll, field])
                    # AND NOT A STATE THE REPAIR PRODUCES. `<field:bills.status> =
                    # "awaiting payment"` is the status send_payment_request WRITES, so
                    # the fault pre-applied half the repair and a policy-following agent
                    # correctly refused to send another request. See sound_fault_value.
                    _at = by_fix[fix][0] if by_fix.get(fix) else {}
                    # THE WHOLE PROVED SEQUENCE, not the headline fix. The atom that
                    # sets bills.status carries fix=reseat_sim_card with fix_sequence
                    # [reseat_sim_card, send_payment_request, make_payment]; reseat does
                    # not touch bills.status, so passing only the headline detected
                    # nothing and this fix was inert on its first attempt.
                    _seq = _at.get("fix_sequence") or ([_at["fix"]] if _at.get("fix") else [])
                    _val = describable_faults.sound_fault_value(
                        mod, world, apply_path, _path, brk["args"]["value"],
                        repair=[(c.get("tool"), c.get("args") or {}) for c in _seq
                                if isinstance(c, dict)])
                    (tgt[0] if isinstance(tgt, list) else tgt)[field] = _val
                elif tool.startswith("<compound:"):
                    # A compound break is a field perturbation followed by a call —
                    # `<compound:phone.vpn_performance_level+connect_vpn>`. Skipping
                    # them left disconnect_vpn unreachable, so one of the policy's 15
                    # fixes could never be exercised by any case. Applied the same way
                    # broken_world does, onto the world being accumulated.
                    c = brk["compound"]
                    pert = c["pert"]
                    tgt = world[pert["collection"]]
                    row = tgt[0] if isinstance(tgt, list) else tgt
                    row[pert["field"]] = pert["value"]
                    mod.derive(world)
                    mod.apply(world, c["call"]["tool"], **(c["call"].get("args") or {}))
                else:
                    mod.apply(world, tool, **(brk.get("args") or {}))
                mod.derive(world)
            except Exception:  # noqa: BLE001
                return None
            # The proved repair, with the arguments that were proved to work. Prefer
            # `fix_sequence` — some repairs take more than one call and grading only the
            # last of them asks the agent to reach a state it was never told to build.
            at = by_fix[fix][0]
            seq = at.get("fix_sequence") or [at["fix"]]
            for call in seq:
                # Same dedupe as the direct-state branch: `reset_apn_settings` proves a
                # repair ending in `reboot_device`, and a walk containing both steps
                # appended it twice. The duplicate then satisfied the criterion after its
                # twin was dropped, so a DROPPED required call survived its own mutant.
                args = dict(call.get("args") or {})
                if any(x["tool"] == call["tool"] and x.get("args") == args
                       for x in solution):
                    continue
                solution.append({"tool": call["tool"], "args": args, "source": "atom"})
            # A PROVED MULTI-CALL REPAIR IS A STATED SEQUENCE. `resume_line` carries an
            # atom whose fix_sequence is the whole payment chain, so those calls entered
            # the solution through this branch and never through DIRECT_STATE — and the
            # order went ungraded. 13 of 24 selected cases required both
            # send_payment_request and make_payment with nothing asserting which came
            # first, and swap_order mutants were killed 0 times out of 21.
            for x, y in zip(seq, seq[1:]):
                pair = (x["tool"], y["tool"])
                if pair[0] != pair[1] and pair not in ordered_pairs:
                    ordered_pairs.append(pair)
        elif fix in DIRECT_STATE:
            spec = DIRECT_STATE[fix][idx % len(DIRECT_STATE[fix])]
            apply_path(world, spec["path"], spec["value"])
            for p, v in spec.get("also") or []:
                apply_path(world, p, v)
            mod.derive(world)
            direct_used.append(spec["why"])
            if spec.get("kind") == "escalation":
                kind = "escalation"
            for t in spec.get("repair") or []:
                # resolved against the DAMAGED world, so `bill_id` names the bill this
                # spec actually made overdue rather than whichever one happens to be first
                args = case_gate.resolve_args(mod, t, world, task_adapter.caller_record(world))
                if args is None:
                    return None      # an argument that cannot be grounded is not a repair
                # DO NOT APPEND A REPAIR ALREADY IN THE SOLUTION. A walk containing both
                # the `send_payment_request` step and the `resume_line` step makes each
                # spec append the same [send_payment_request, make_payment] chain, and
                # the field they set is the same field: one chain repairs both. Measured
                # 2026-08-19, 6 of 24 cases carried the pair twice, which inflated the
                # criteria and made a DROPPED call survive its own mutant because the
                # duplicate still satisfied the criterion.
                same = {"tool": t, "args": args, "source": "direct_state"}
                if not any(x["tool"] == t and x.get("args") == args for x in solution):
                    solution.append(same)
            # THE CHAIN IS THE POLICY'S STATED SEQUENCE. `repair` is ordered — pay before
            # the suspension is lifted, reset the APN before rebooting — and that order is
            # the rule, not an artefact of how the list was written. Recorded so the
            # criteria can grade it; without this the ordering rules came only from walk
            # steps, and these calls enter through a repair chain rather than as steps, so
            # the sequence was ungraded. Measured: swap_order mutants were killed 0 times
            # out of 21.
            chain = list(spec.get("repair") or [])
            for x, y in zip(chain, chain[1:]):
                if (x, y) not in ordered_pairs:
                    ordered_pairs.append((x, y))
        else:
            return None

    # A BILLING FAULT MUST PRESENT ITSELF.
    #
    # Coupling the two DIRECT_STATE payment specs to the suspension was not enough: most
    # payment cases do not come through DIRECT_STATE at all. `<field:bills.status> =
    # "awaiting payment"` is an ATOM, and the atom branch wins whenever an atom exists,
    # so 7 of 12 payment cases in the regenerated suite still ran on an Active line.
    #
    # A bill is only visible to an agent that goes looking, and nothing sends it looking
    # while service works normally. Measured on 240 traces: send_payment_request was
    # NEVER RAISED in 86 of its 91 misses. Applied here, after the world is built, so it
    # holds for every path that can produce an unpaid bill rather than for one table.
    _pay = {"send_payment_request", "make_payment"} & {s["tool"] for s in solution}
    if _pay:
        _caller = task_adapter.caller_record(world)
        _mine = [l for l in (world.get("lines") or [])
                 if l.get("customer_id") == _caller.get("customer_id")] \
            or (world.get("lines") or [])
        if _mine and str(_mine[0].get("status") or "").lower() != "suspended":
            _mine[0]["status"] = "Suspended"
            mod.derive(world)
            if not any(s["tool"] == "resume_line" for s in solution):
                _a = case_gate.resolve_args(mod, "resume_line", world, _caller)
                if _a is None:
                    return None
                solution.append({"tool": "resume_line", "args": _a,
                                 "source": "presentation"})
            for _x, _y in (("send_payment_request", "make_payment"),
                           ("make_payment", "resume_line")):
                if (_x, _y) not in ordered_pairs:
                    ordered_pairs.append((_x, _y))

    # A STEP THE REPAIR DOES NOT NEED MUST NOT BE GRADED.
    #
    # The prefix check asks whether the world is ALREADY healthy before a step, which
    # finds trailing dead weight and nothing else - it reported 0 overspecified on the
    # 24-case suite while this sat inside it:
    #
    #   atom  break {"tool": "<field:bills.status>", "value": "awaiting payment"}
    #         fix_sequence [reseat_sim_card, send_payment_request, make_payment]
    #
    # Reseating a SIM has nothing to do with an unpaid bill. Every case built from that
    # atom graded reseat_sim_card, all 7 of them with sim_status "active", while the
    # policy conditions reseating on the SIM showing "Missing". So the policy-FOLLOWING
    # agent correctly declined and the policy-BLIND one reseated anyway: policy 32.5
    # against stub 57.5, MINUS 25, an inverted criterion made entirely of an action
    # nobody should have taken.
    if kind != "escalation":
        solution, _dropped = prefix_gate.prune(mod, world, solution)
        if _dropped:
            build_case.pruned.extend(_dropped)

    if all(mod.evaluate(world).values()):
        return None                                  # nothing broke; not a test

    # IF THE WORLD CANNOT BE FIXED, THE ONLY CORRECT ACTION IS TO HAND OFF. A case that
    # required escalation AND three repairs is incoherent: the SIM is locked, so the
    # repairs cannot resolve anything and grading them punishes an agent that correctly
    # recognised the limit. tau3 builds it the same way — its
    # `airplane_mode_on|lock_sim_card_pin` task grades transfer_to_human_agents and
    # nothing else, even though airplane mode is fixable.
    esc = escalation_tool(mod)
    if kind == "escalation" or any(s["tool"] == esc for s in solution):
        kind = "escalation"

        # AN ESCALATION CASE MUST STILL GRADE THE DIAGNOSIS.
        #
        # Truncating the solution to the handoff alone left these cases grading TWO
        # criteria — call transfer, do not call one forbidden tool — where a repair case
        # grades 3 to 13. Five health predicates were broken and not one was graded, so
        # an agent that transferred on its first turn scored 100%.
        #
        # Measured over 200 runs on 2026-08-21, `transfer_to_human_agents` scored
        # policy 20.0% against stub 72.0% — MINUS 52 points, the worst criterion in the
        # suite. The same tool graded as FORBIDDEN on repair cases scored +14.7 over 95
        # runs. One behaviour, two criteria, opposite signs: 95(+14.7) + 25(-52.0) is
        # +0.8pp over 120 runs. Raising the escalation share from 4% to 21% to match
        # tau3 therefore converted a working discriminator into noise.
        #
        # The policy does not say "escalate"; it says escalate "if and only if the
        # request cannot be handled within the scope of your actions". Establishing that
        # requires looking. The REPAIRS stay out of the solution — they genuinely cannot
        # resolve a locked SIM, and grading them would punish an agent that correctly
        # recognised the limit, which was the 2026-08-18 contradiction. The CHECKS are
        # read-only and always callable, so requiring them costs a correct agent nothing
        # and costs a give-up agent everything.
        checks = []
        for st in walk["steps"]:
            ct = st.get("check")
            if not ct or ct == esc or any(x["tool"] == ct for x in checks):
                continue
            ca = case_gate.resolve_args(mod, ct, world, task_adapter.caller_record(world))
            if ca is None:
                continue          # an argument that cannot be grounded is not a check
            checks.append({"tool": ct, "args": ca, "source": "diagnosis"})
        solution = checks + [s for s in solution if s["tool"] == esc][:1]
        walk = dict(walk, steps=[st for st in walk["steps"] if st["fix"] == esc][:1])
        for c in checks:
            ordered_pairs.append((c["tool"], esc))

    # CRITERIA ARE GENERATED FROM THE REFERENCE SOLUTION, not alongside it.
    #
    # They were written independently from `step["fix"]`, so the graded actions and the
    # actions that actually repair the world were two different lists that nothing kept
    # in agreement. Measured: 191 of 191 criteria carried no arguments, and replaying
    # them left 38 of 45 worlds broken. Deriving one from the other makes a case that
    # grades an action its own solution never takes impossible to express.
    by_step = {s["fix"]: s for s in walk["steps"]}
    crits, graded = [], set()
    for i, call in enumerate(solution):
        tool = call["tool"]
        args = call.get("args") or {}
        missing = case_gate.missing_required_args(mod, tool, args)
        if missing:
            return None          # an uncallable criterion is not a test
        crits.append({
            "id": f"s{i}_{tool}", "type": "behavior_check",
            "rule": "tool_called_with",
            # WHO ACTS IS DECIDABLE, not a matter of the extractor's opinion. The graph
            # tagged make_payment as the agent's; it is in USER_TOOLS and tau3 grades it
            # requestor='user'. So we required the agent to call a tool it does not have,
            # and that criterion failed 15 times out of 15 in BOTH arms — pure dead
            # weight diluting the gap. The compiled module records the split.
            # GRADE AN ARGUMENT ONLY IF THE WORLD CONSTRAINS IT. compare_args_for
            # drops free-TEXT arguments; this drops values that are free by
            # CONSEQUENCE. The reference solution recorded gb_amount=1, so an agent
            # that refuelled 2GB - equally correct, equally repairing - was marked
            # wrong in 21 of refuel_data's 43 misses on 2026-08-20.
            # AN EQUIVALENT LOOKUP SATISFIES THE REQUIREMENT. get_data_usage scored
            # policy 0.0 / stub 50.0 in run21 — MINUS 50, the worst criterion — and in
            # 9 of the 10 policy-arm misses the agent had called get_details_by_id,
            # which carries the same line record. It had the fact and was marked wrong
            # for reading it from the other endpoint.
            "param": {"tool": tool, "arguments": args,
                      "any_of": (_alt := equivalent_lookups.equivalents(
                          mod, tool, world, task_adapter.caller_record(world))),
                      # EQUIVALENT TOOLS DO NOT SHARE A SIGNATURE. get_data_usage takes
                      # (customer_id, line_id) and get_details_by_id takes (id), so
                      # accepting the NAME and then comparing arguments rejected it
                      # anyway — the criterion stayed inverted at -30.0 with any_of
                      # correctly attached. Where a substitute is allowed, the CALL is
                      # what is graded, which is what tau3 does for its own equivalent
                      # lookups.
                      "compare_args": ([] if len(_alt) > 1 else arg_gate.graded_args(
                          mod, world, solution, i,
                          task_adapter.compare_args_for(args))),
                      "requestor": ("user" if tool in user_side else "assistant")},
            "description": str((by_step.get(tool) or {}).get("issue")
                               or f"the policy's repair for this fault")[:110]})
        graded.add(tool)
    for step in walk["steps"]:
        for req in (step.get("requires") or []):
            if (req, step["fix"]) not in ordered_pairs:
                ordered_pairs.append((req, step["fix"]))
    # ORDERING ATTACHES TO THE CALL IT CONSTRAINS. As a criterion of its own it was
    # satisfied by not acting — see runner/verdict.py for the five measured gaps, three
    # dead and two inverted. Folded into the must-call criterion, absence fails the
    # must-call half, so doing less can no longer score. It also stops the double count
    # that made each ordering duplicate the must-call failure mass in both arms.
    by_tool = {c["param"]["tool"]: c for c in crits
               if c["rule"] == "tool_called_with" and (c.get("param") or {}).get("tool")}
    for first, second in ordered_pairs:
        if first in by_tool and second in by_tool and first != second:
            bl = by_tool[first]["param"].setdefault("before", [])
            if second not in bl:
                bl.append(second)

    # ONE DIAGNOSIS CRITERION, NOT ONE PER CHECK.
    #
    # Requiring the diagnosis before a handoff moved transfer_to_human_agents from
    # -52.0 to -16.0 and produced the two strongest criteria in the suite
    # (check_status_bar +50.0, check_sim_status +36.0). Requiring EVERY check of every
    # step over-corrected: 8 of run18's 14 dead criteria are checks that NEITHER arm
    # ever ran, which cannot discriminate and drags the absolute score down for both.
    #
    # The policy says escalate only when the request "cannot be handled". Establishing
    # that takes looking, not looking exhaustively. `any_of` already exists in the
    # grader for exactly this - a policy that authorises three equivalent lookups must
    # not be graded on one of them - so a give-up agent still fails on zero checks
    # while a diligent one is not punished for choosing a different first check.
    if kind == "escalation":
        checks = [c for c in crits if c["rule"] == "tool_called_with"
                  and str((c.get("param") or {}).get("tool") or "").startswith("check_")]
        if len(checks) > 1:
            merged = checks[0]
            merged["id"] = "diagnosed_before_handoff"
            merged["param"] = {"tool": merged["param"]["tool"],
                               "any_of": sorted({c["param"]["tool"] for c in checks}),
                               "arguments": {}, "compare_args": [],
                               # either party may run it: some checks are the customer's
                               "requestor": None,
                               "before": [esc] if esc else []}
            merged["description"] = ("the policy allows escalation only once the request "
                                     "is shown to be outside what the agent can do")
            drop = {id(c) for c in checks[1:]}
            crits = [c for c in crits if id(c) not in drop]
    # THE BEHAVIOUR THE BASELINE SHOWS ACTUALLY SEPARATES THE ARMS. Measured on tau3's
    # own tasks through our grading: the stubbed agent calls transfer_to_human_agents in
    # 42% of runs against the policy agent's 29% — it gives up on problems it could have
    # fixed. The policy states the rule directly: escalate "if and only if the request
    # cannot be handled within the scope of your actions".
    #
    # The graph suite graded this ZERO times across 30 cases, so the one thing that most
    # distinguishes a policy-blind agent was invisible to it, and three iterations of
    # tuning acceptance and ordering could not have recovered it.
    #
    # Added only where the case is fixable — where escalation is itself the graded
    # answer, requiring and forbidding it would be the contradiction found on 2026-08-18.
    esc = escalation_tool(mod)
    if esc and esc not in {st["fix"] for st in walk["steps"]}:
        crits.append({
            "id": "no_premature_escalation", "type": "behavior_check",
            "rule": "tool_not_called", "param": {"tool": esc},
            "description": f"every fault here is fixable; the policy allows escalation "
                           f"only when the request cannot be handled"})

    case = {
        "source_task_id": f"[graph:{walk['family'][:28]}]"
                          + "|".join(s["fix"] for s in walk["steps"]),
        "kind": kind,
        "world_setup": {"entities": world,
                        "user_tools": task_adapter.user_tool_specs(mod)},
        # WHAT THIS CASE CLAIMS TO TEST, in the environment's own vocabulary. Without it
        # solvability cannot be checked at all: the gate would have to recompute what
        # broke and would then agree with the case by construction, so collateral damage
        # would silently become part of the expected repair.
        "symptoms": {
            "broken": sorted(k for k, v in mod.evaluate(world).items() if not v),
            "still_working": sorted(k for k, v in mod.evaluate(world).items() if v),
        },
        # The ordered calls that repair it, with the arguments proved to work. This is
        # the same list the criteria above were generated from.
        "reference_solution": solution,
        "goal": "",
        "known_facts": {"n_atoms": len(walk["steps"])},
        # THE NARRATOR READS `caller` (narrator.py:175) to give the customer a real
        # identity. Putting the record in known_info instead left it unread, and the
        # narrator wrote "no name, phone number or account details are offered" — so the
        # simulated customer had no identity at all, could not answer the very first
        # question the policy requires (customer lookup), and every case would have
        # stalled before any troubleshooting began.
        "caller": task_adapter.caller_record(world),
        "wont_volunteer": [], "will_reveal_if_asked": [],
        "family": walk["family"], "direct_state": direct_used,
        "acceptance": acceptance_from_world(mod, world),
        "success_criteria": crits,
    }

    # THE GATE. A case that its own reference solution cannot solve is not a test, and
    # shipping it grades the agent on a conversation that could not have succeeded.
    # Rejected rather than repaired: silently adding a missing repair would invent a
    # test the policy graph never described, and the count of rejections is the
    # measurement that says how good the generator is.
    verdict = case_gate.check(mod, case)
    if not verdict["solvable"]:
        build_case.rejected.append({
            "id": case["source_task_id"], "kind": kind,
            "errors": verdict.get("errors"), "residual": verdict.get("residual"),
            "unrepaired": verdict.get("unrepaired"), "phantom": verdict.get("phantom"),
            "why": verdict.get("why")})
        return None
    return case


# Default so `build_case` is callable outside a generation run, e.g. from tests.
build_case.rejected = []
build_case.pruned = []


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="telecom")
    ap.add_argument("--cases", type=int, default=24)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=6)
    # RAISED FROM 4 on 2026-08-19. The old value was set on the reasoning that "above
    # ~5 nothing passes under either condition, so the band that discriminates is 2-4".
    # tau3's own run records disprove it. Sliced by depth, its BINARY gap does collapse
    # in the tail (+8.9pp at n>7 against +21.0pp at n<=7) — but its PARTIAL-CREDIT gap
    # is the reverse and much larger: +30.2pp in the tail against +12.7pp shallow. At
    # n>=9 the policy-following agent completes 72.7% of the required actions and the
    # stubbed one 42.5%. Neither ever finishes; one gets most of the way and the other
    # does not, and a binary metric cannot see the difference.
    #
    # The cap was the only thing preventing the tail. With prerequisite families in the
    # pool — MMS depends on Mobile Data depends on Cellular Service, which is exactly how
    # tau3 layers its deep tasks — max_len 8 reaches depth 11, tau3's own range:
    #   max_len 4 -> depth 1-7,  0 cases at depth>=8
    #   max_len 6 -> depth 1-9,  1733
    #   max_len 8 -> depth 1-11, 6204
    ap.add_argument("--max-len", type=int, default=4,
                    help="max faults per case. Raised to 8 on 2026-08-19 to reach "
                         "tau3's depth range, then REVERTED: the deep run measured a "
                         "WORSE gap (+4.2pp against +8.7pp shallow) because nothing "
                         "above depth 3 passed in either arm, so 15 of 24 cases "
                         "contributed nothing. Depth is not the constraint; raise this "
                         "again only after a case at depth 8 can actually be passed.")
    ap.add_argument("--with-branches", action="store_true",
                    help="attach the policy decisions compiled by branch_compiler. They "
                         "had nowhere to land on the old composer's cases; the graph "
                         "generates the states their preconditions need")
    # 100, because that is what tau3 runs (tau2/run.py:82, runner/simulation.py:53).
    # We were running 20 — a fifth of the benchmark whose number we are chasing — so over
    # half of every measurement was a conversation cut off mid-repair, and it was cut off
    # UNEVENLY: the agent following the policy works through a diagnostic checklist and
    # loses more to the cap than one that goes straight at fixes. Raising it to 40 would
    # have been a patch; using the benchmark's own value is not.
    # 100 was tau3's value and was adopted on the reasoning that running at 20 made
    # "over half of every measurement a truncated conversation". Transcripts later showed
    # our conversations stopped doing work at ~turn 17 and then deadlocked to the cap, so
    # the cap was never the binding constraint: it bought no fidelity and multiplied the
    # cost of every run fivefold. With termination fixed, conversations end at 15-25 turns
    # and tau3's own median through this harness is 21. 30 leaves headroom over both.
    ap.add_argument("--max-turns", type=int, default=30)
    # A/B RUNS MUST GRADE THE SAME SUITE. --grade rebuilds and re-narrates every time, so
    # two runs differing in one code change also differed in the customer's opening lines.
    # The cases, worlds and criteria are deterministic; the narration is not.
    ap.add_argument("--reuse-suite", action="store_true",
                    help="grade the suite already on disk instead of rebuilding it")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--grade", action="store_true")
    a = ap.parse_args()

    mod = importlib.import_module(f"generated.env_{a.domain}")
    graph = json.load(open(REPO / "generated" / f"policy_graph_{a.domain}.json"))
    atoms = json.load(open(REPO / "reports" / f"phase3_atoms_{a.domain}.json"))["atoms"]
    # AN ATOM MUST BE FILED UNDER THE FIX ITS OWN BREAK NEEDS. One telecom atom breaks
    # bills.status and is filed under reseat_sim_card, so every walk containing a reseat
    # step pulled a billing fault into a case about signal. 9 of the 12 payment cases in
    # run22 had no payment step in the walk at all, which is why our policy gap came out
    # at DOUBLE tau3's: 50% of our cases involved the payment chain against tau3's 5%.
    atoms, _notes = atom_audit.refile(mod, atoms)
    for _n in _notes:
        print(f"  ATOM RE-FILED: {_n}")

    print("=" * 78)
    print(f"PHASE 15 — cases by walking the policy graph ({a.domain})")
    print("=" * 78)
    for g in graph:
        print(f"  {g['family'][:56]:56s} {len(g['steps'])} steps")

    envmod = importlib.import_module(f"tau2.domains.{a.domain}.environment")
    policy = envmod.get_environment().get_policy() or ""
    deps = policy_graph.prerequisites(policy, [g["family"] for g in graph])
    for k, v in deps.items():
        if v:
            print(f"  {k[-34:]:34s} requires {[x[-24:] for x in v]}")
    walks = policy_graph.walks(graph, max_len=a.max_len, deps=deps)
    # Difficulty is controlled by --max-len rather than by reweighting. An attempt to
    # sample walks to tau3's length distribution produced LONGER walks than the raw
    # pool, because short walks are scarce (40 of length 1 against thousands of length
    # 6) so the caps only ever bound the short end. max_len=4 already yields the 1-4
    # spread tau3 has and the band that discriminates; a reweighting that has to be
    # debugged is worse than a cap that does not.
    branch_defs = json.load(open(REPO / 'generated' / f'branches_{a.domain}.json'))
    build_case.rejected = []
    build_case.pruned = []
    built, skipped = [], 0
    for i, w in enumerate(walks):
        c = build_case(w, mod, atoms, i)
        (built.append(c) if c else None)
        skipped += 0 if c else 1
    # WHY cases were rejected, not merely how many. "skipped 300" says nothing about
    # whether the generator is healthy; the breakdown says what to fix next.
    rej = build_case.rejected
    print(f"\n{'-' * 78}\nSOLVABILITY GATE\n{'-' * 78}")
    print(f"  walks {len(walks)}  ->  cases {len(built)}   rejected {skipped}")
    print(f"    of which reached the gate (world built, args resolved): {len(rej)}")
    reasons = collections.Counter()
    for r in rej:
        for key in ("errors", "residual", "unrepaired", "phantom"):
            if r.get(key):
                reasons[key] += 1
        if r.get("why"):
            reasons[r["why"][:46]] += 1
    for k, v in reasons.most_common():
        print(f"      {k:48s} {v}")
    if rej:
        print("    examples:")
        for r in rej[:4]:
            detail = (r.get("errors") or r.get("unrepaired")
                      or r.get("residual") or r.get("phantom") or r.get("why"))
            print(f"      {str(r['id'])[:52]:52s} {str(detail)[:60]}")
    # COVER FIRST, then spread. Selecting the first case of each (family, length)
    # bucket left 6 of the 20 policy steps never exercised: itertools.combinations
    # emits in index order, so the early steps of every family dominated and
    # refuel_data, disconnect_vpn, enable_roaming, toggle_data_saver_mode,
    # grant_app_permission and resume_line appeared in no case at all. A suite that
    # silently omits a third of the procedure it claims to test is the same truncation
    # bias this project has now paid for three times in one day.
    def fixes_of(case):
        return {x["param"]["tool"] for x in case["success_criteria"]
                if x["rule"] == "tool_called_with"}

    want = {s["fix"] for g in graph for s in g["steps"]}
    picked, seen = [], set()
    # greedy set cover, TIES BROKEN TOWARD DEPTH. `max` returns the first maximum, so
    # ties went to whatever `itertools.combinations` emitted earliest — which is the
    # shortest walk. 7 of 23 selected cases graded a single call against tau3's 3 of 25,
    # so the suite was systematically shallower than the benchmark it is measured
    # against for a reason that had nothing to do with what it could generate.
    while len(picked) < a.cases:
        best = max((c for c in built if c not in picked),
                   key=lambda c: (len(fixes_of(c) - seen), len(fixes_of(c))),
                   default=None)
        if best is None or not (fixes_of(best) - seen):
            break
        picked.append(best)
        seen |= fixes_of(best)

    # Then fill toward tau3's MEASURED depth profile rather than round-robin from the
    # smallest bucket. Depth is the axis on which our suite and the benchmark are
    # compared, so it is chosen deliberately instead of falling out of iteration order.
    # DEPTH AND FAMILY BOTH, because optimising one alone collapses the other. A first
    # version filled purely by depth deficit and selected all 24 cases from a single
    # family, so the suite tested one section of the policy while looking well shaped.
    # The families are the sections of the troubleshooting manual; a suite drawn from one
    # is not a suite for the policy.
    per = collections.defaultdict(list)
    for c in built:
        if c not in picked:
            per[(c["family"], len(fixes_of(c)), c.get("kind") or "repair")].append(c)
    families = sorted({c["family"] for c in built})
    have_d = collections.Counter(len(fixes_of(c)) for c in picked)
    have_f = collections.Counter(c["family"] for c in picked)
    have_k = collections.Counter(c.get("kind") or "repair" for c in picked)
    target_total = sum(TAU3_DEPTH.values()) or 1
    while len(picked) < a.cases and any(per.values()):
        avail = [k for k, v in per.items() if v]
        n = max(1, len(picked))

        def score(key):
            fam, d, kind = key
            depth_deficit = TAU3_DEPTH.get(d, 0) / target_total - have_d[d] / n
            fam_deficit = 1 / max(1, len(families)) - have_f[fam] / n
            # tau3 telecom is ~18% escalation and ours was 5% of the built pool and 1 of
            # 24 selected. Escalation is where the POLICY decides the answer — the agent
            # must recognise a limit rather than act — so under-representing it removes
            # the cases most likely to separate a policy-following agent from a blind one.
            kind_deficit = TAU3_KIND.get(kind, 0) - have_k[kind] / n
            return (kind_deficit, fam_deficit, depth_deficit, d)

        k = max(avail, key=score)
        picked.append(per[k].pop(0))
        have_d[k[1]] += 1
        have_f[k[0]] += 1
        have_k[k[2]] += 1
    uncovered = want - seen
    if uncovered:
        print(f"  !! policy steps no case can exercise: {sorted(uncovered)}")

    print(f"\n  walks {len(walks)} | built {len(built)} | unbuildable {skipped}")
    print(f"  selected {len(picked)}")
    # NOT `family[:34]`. Three of telecom's four families share that exact prefix, so
    # the line read `{'Understanding and Troubleshooting ': 24}` and was mistaken for a
    # suite drawn from ONE family — a regression diagnosed, written up and "fixed" before
    # the label was checked. A display that merges distinct values manufactures findings.
    _fam = collections.Counter(c["family"] for c in picked)
    print(f"  by family: {len(_fam)} families")
    for _f, _n in _fam.most_common():
        print(f"      {_n:3d}  {_f}")
    print(f"  by kind  : {dict(collections.Counter(c.get('kind') or 'repair' for c in picked))}"
          f"   (tau3 telecom is ~18% escalation)")
    print(f"  carrying a policy-named state no tool produces: "
          f"{sum(1 for c in picked if c.get('direct_state'))}")
    print(f"  criteria per case: "
          f"{dict(sorted(collections.Counter(len(c['success_criteria']) for c in picked).items()))}")
    import collections as _c
    if build_case.pruned:
        print(f"  steps dropped as unnecessary (leave-one-out): "
              f"{dict(_c.Counter(build_case.pruned))}")
    ordering = sum(len((x.get("param") or {}).get("before") or [])
                   for c in picked for x in c["success_criteria"])
    print(f"  ordering constraints (attached to must-call): {ordering}")

    # IS THE CASE INFORMATIVE? The other five gates all ask whether it is WELL-FORMED,
    # which is a property of one agent-free replay. This one asks whether it can
    # separate two agents, by writing down the agent that does LESS. Reported rather
    # than rejected: the count is the measurement that says how good the generator is,
    # and silently dropping a third of the criteria would hide it.
    _inert = _inert_cases = 0
    for c in picked:
        _bad = {b["criterion"] for b in await discrimination_gate.offenders(c, mod)}
        _inert += len(_bad)
        _inert_cases += bool(_bad)
    _tot = sum(len(c["success_criteria"]) for c in picked)
    print(f"  DISCRIMINATION GATE — criteria an agent that does LESS still satisfies: "
          f"{_inert} of {_tot}"
          + (f"  ({100 * _inert / _tot:.1f}%)" if _tot else "")
          + f", in {_inert_cases} of {len(picked)} cases")
    if _inert:
        print(f"      the suite shipped on 2026-08-20 scored 58 of 168 (34.5%) here")

    # THE CUSTOMER HAS TO HAVE SOMETHING TO SAY. Every case was written with an empty
    # goal, so the simulated customer would have opened with nothing and the whole
    # benchmark would have measured an empty conversation. The narrator turns the broken
    # world into symptoms in a person's words, with its existing gates: no naming the
    # fix, a real identity from this world, no baked-in temperament.
    print(f"\n  narrating {len(picked)} cases...")
    sem = asyncio.Semaphore(a.concurrency)

    async def narrate_one(case):
        """The customer, built by `meta_user` rather than the narrator.

        The narrator was handed the device state and told to distribute it into
        `wont_volunteer` and `will_reveal_if_asked`. The customer then ANSWERED
        diagnostic questions from its script instead of running the check, so the tool was
        never called and every requestor:user criterion failed as "never called by the
        user" — 63% of what we grade. tau3 gives its customer identity and a symptom and
        nothing else, so the only way to learn a reading is to run the check.

        Both list fields are deliberately emptied. They have no tau3 equivalent and they
        are the mechanism of the defect.
        """
        async with sem:
            sc = await meta_user.build(mod=mod, domain=a.domain, case=case,
                                       graph=graph, branches=branch_defs)
            ins = sc.get("instructions") or {}
            if not (ins.get("reason_for_call") or "").strip():
                return None
            out = dict(case)
            out["scenario"] = sc
            out["goal"] = ins["reason_for_call"]
            out["known_facts"] = {**(case.get("known_facts") or {}),
                                  "known_info": ins.get("known_info") or ""}
            out["task_instructions"] = ins.get("task_instructions") or ""
            out["wont_volunteer"] = []
            out["will_reveal_if_asked"] = []
            return out

    if a.reuse_suite:
        # GRADE THE SUITE ALREADY ON DISK. --grade otherwise rebuilds and re-narrates, so
        # two runs differing in one code change also differed in the customer's opening
        # lines. Cases, worlds and criteria are deterministic; narration is not.
        kept = json.load(open(REPO / "generated" / f"graph_suite_{a.domain}.json"))
        print(f"  REUSING the suite on disk: {len(kept)} cases, no rebuild, no "
              f"re-narration — an A/B then differs only in the code under test")
        picked = kept
        narrated = kept
    else:
        narrated = await asyncio.gather(*[narrate_one(c) for c in picked])
    # CONTEXT THE WORLD CANNOT HOLD. The policy says to enable roaming "if a user is
    # travelling outside their home network", and the compiled world has
    # `roaming_enabled` but no location at all — so nothing in the scenario could ever
    # justify the action and enable_roaming failed in 15 of 15 runs under BOTH the policy
    # and the stub, adding equal dead weight to each. tau3 states it in the user's own
    # words ("You are currently abroad in France") rather than in the database, and so
    # must we. Added after narration because apply_narrative replaces known_facts.
    ROAM = {"enable_roaming", "toggle_roaming"}
    for c in narrated:
        if not c:
            continue
        graded = {x["param"].get("tool") for x in c["success_criteria"]}
        if graded & ROAM:
            c.setdefault("will_reveal_if_asked", []).insert(
                0, "I am travelling outside my home network at the moment.")
            c["known_facts"] = dict(c.get("known_facts") or {},
                                    travelling="outside home network")
    if not a.reuse_suite:
        kept = [c for c in narrated if c and (c.get("goal") or "").strip()]
    print(f"  narrated {len(kept)}/{len(picked)}"
          f"{' — REFUSING to write cases with no customer' if len(kept) < len(picked) else ''}")
    if not kept:
        print("  !! nothing narrated; not writing")
        return 1
    picked = kept

    if a.with_branches:
        from services import branch_compiler as BC
        bpath = REPO / "generated" / f"branches_{a.domain}.json"
        if bpath.exists():
            branches = json.load(open(bpath))
            before = sum(1 for c in picked if c.get("acceptance"))
            for c in picked:
                BC.apply_branches(c, branches)
            got = sum(1 for c in picked if c.get("acceptance"))
            kinds = collections.Counter(k for c in picked
                                        for k in (c.get("branch_kinds") or []))
            print(f"  branches attached: {got}/{len(picked)} cases "
                  f"(was {before}) | kinds {dict(kinds)}")
        else:
            print(f"  !! {bpath.name} missing; no branches attached")

    out = REPO / "generated" / f"graph_suite_{a.domain}.json"
    out.write_text(json.dumps(picked, indent=1, default=str))
    print(f"\nwritten: {out.relative_to(REPO)}")
    if not a.grade:
        return 0

    journeys = [B.journey_of(c, None, i) for i, c in enumerate(picked)]
    wa = B.agent_under_test(a.domain)
    wo = B.agent_under_test(a.domain, system_prompt=STUB)

    async def arm(agent, tag=""):
        res = []
        for _ in range(a.repeats):
            vals = await B.gather_bounded(
                # 53 of 90 policy runs and 48 of 90 stub runs were hitting a 20-turn
                # cap — median turns equalled the cap exactly, so more than half of every
                # measurement was a TRUNCATED conversation rather than a concluded one.
                # It also biased the comparison: the policy prescribes a diagnostic
                # sequence, so the agent following it spends turns on checks and loses
                # more to truncation than the stub that goes straight at fixes. That is
                # why repair criteria ran BACKWARDS (-12) and cancelled the escalation
                # criterion that was working (+5).
                [(lambda j=j: B.one_run(agent, a.domain, j, UserSim(),
                                        max_turns=a.max_turns,
                                        trace_tag=f"{tag}_r{len(res)}"))
                 for j in journeys], a.concurrency)
            res.append({j["id"]: v for j, v in zip(journeys, vals)})
        return res

    # KEEP THE CONVERSATIONS. Outcome numbers cannot say whether the agent failed to
    # instruct the user or the simulator ignored it, and those need opposite fixes.
    B.TRACE_DIR = REPO / "logs" / "traces" / f"{a.domain}"
    print(f"  traces -> {B.TRACE_DIR}")
    print(f"\n{'-' * 78}\nWITH POLICY\n{'-' * 78}")
    W = await arm(wa, "with")
    for k, rep in enumerate(W):
        print(f"    repeat {k+1}: {sum(1 for v in rep.values() if v['verdict']=='pass')}"
              f"/{len(rep)} pass | repair {B.repair_rate(list(rep.values())):.1f}%")
    print(f"\n{'-' * 78}\nWITHOUT POLICY\n{'-' * 78}")
    O = await arm(wo, "stub")
    for k, rep in enumerate(O):
        print(f"    repeat {k+1}: {sum(1 for v in rep.values() if v['verdict']=='pass')}"
              f"/{len(rep)} pass | repair {B.repair_rate(list(rep.values())):.1f}%")

    def r(P):
        p, q = [], []
        for rep in P:
            sc = [v for v in rep.values() if v["verdict"] in ("pass", "fail")]
            p.append(100.0 * sum(1 for v in sc if v["verdict"] == "pass") / max(1, len(sc)))
            q.append(B.repair_rate(sc))
        return p, q
    pw, rw = r(W)
    po, ro = r(O)
    print(f"\n{'-' * 78}\nRESULT — cases composed along the policy\n{'-' * 78}")
    print(f"  pass   with {statistics.mean(pw):5.1f}%  without {statistics.mean(po):5.1f}%"
          f"  gap {statistics.mean(pw)-statistics.mean(po):+5.1f}pp")
    print(f"  repair with {statistics.mean(rw):5.1f}%  without {statistics.mean(ro):5.1f}%"
          f"  gap {statistics.mean(rw)-statistics.mean(ro):+5.1f}pp")
    print(f"\n  our old composer measured +3.0pp pass, +11.0pp repair")
    print(f"  errors {len(B.ERRORS)}")

    # BY DEPTH, NEVER ONLY THE MEAN. Sliced this way, tau3's binary gap COLLAPSES in the
    # tail (+8.9pp at n>7 against +21.0pp at n<=7) while its partial-credit gap does the
    # opposite (+30.2pp against +12.7pp). One aggregate number hides both facts, and they
    # point to different things: the tail is where a suite stops discriminating on
    # pass/fail and starts discriminating on how far the agent got.
    print(f"\n{'-' * 78}\nBY DEPTH — required actions per case\n{'-' * 78}")
    depth_of = {j["id"]: len([c for c in (j.get("success_criteria") or [])
                              if c.get("rule") == "tool_called_with"]) for j in journeys}
    buckets = collections.defaultdict(lambda: {"wp": [], "op": [], "wr": [], "or": []})
    for rep_w, rep_o in zip(W, O):
        for jid, d in depth_of.items():
            b = buckets[d]
            for tag, rep in (("w", rep_w), ("o", rep_o)):
                v = rep.get(jid) or {}
                b[f"{tag}p"].append(1.0 if v.get("verdict") == "pass" else 0.0)
                if v.get("n"):
                    b[f"{tag}r"].append(v["k"] / v["n"])
    print(f"  {'depth':>5s} {'cases':>5s}   {'pass with':>9s} {'stub':>6s} {'gap':>7s}"
          f"   {'repair with':>11s} {'stub':>6s} {'gap':>7s}")
    for d in sorted(buckets):
        b = buckets[d]
        n_cases = sum(1 for k, v in depth_of.items() if v == d)
        pw_ = statistics.mean(b["wp"]) * 100 if b["wp"] else 0
        po_ = statistics.mean(b["op"]) * 100 if b["op"] else 0
        rw_ = statistics.mean(b["wr"]) * 100 if b["wr"] else 0
        ro_ = statistics.mean(b["or"]) * 100 if b["or"] else 0
        print(f"  {d:5d} {n_cases:5d}   {pw_:9.1f} {po_:6.1f} {pw_-po_:+7.1f}"
              f"   {rw_:11.1f} {ro_:6.1f} {rw_-ro_:+7.1f}")
    print(f"\n  tau3 for comparison:  n<=7  binary +21.0  partial +12.7")
    print(f"                        n>7   binary  +8.9  partial +30.2")

    rep_out = REPO / "reports" / f"phase15_graph_{a.domain}.json"
    rep_out.write_text(json.dumps({
        "cases": len(picked), "repeats": a.repeats,
        "pass": {"with": round(statistics.mean(pw),1), "without": round(statistics.mean(po),1)},
        "repair": {"with": round(statistics.mean(rw),1), "without": round(statistics.mean(ro),1)},
        "per_case": {j["id"]: {"with": [rep[j["id"]] for rep in W],
                               "without": [rep[j["id"]] for rep in O]} for j in journeys},
    }, indent=2, default=str))
    print(f"\nwritten: {rep_out.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
