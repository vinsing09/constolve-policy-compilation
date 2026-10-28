"""Turn a proved task from the compiled environment into a runnable test case.

WHY THIS EXISTS

`scripts/phase4_compose.py` emits tasks shaped `{breaks, graded_actions, predicates}`.
The runner needs `{scenario, goal, world_setup, success_criteria, known_facts}`. Nothing
bridged them, so 161,775 proved tasks could not be run against an agent, and surface 2
could not be measured on the compiled environment at all.

THREE PARTS, AND ONLY ONE OF THEM MAY USE A MODEL

  sample      which tasks to run. Deterministic, and it has to exist: 161,775 is not a
              suite, it is a space. tau3 samples at most 3 per (intent, size, persona)
              bin and excludes singletons; the same shape is used here.
  mechanics   the broken world, and the criteria. Deterministic. The graded actions are
              already proved to repair the world, so they become `tool_called_with`
              criteria directly, with tau3's `compare_args` to say which arguments are
              graded.
  narrative   what the user says and knows. A model writes this, ONCE per sampled task,
              and it is forbidden from touching the mechanics. If it could, the replay
              proof would no longer describe the case that runs.

That separation is the whole design. The proof covers the mechanics; the narrative is
presentation over a case that is already established as solvable.
"""
from __future__ import annotations

import collections
import json
from typing import Any

# Arguments that are free text and must not be graded literally. tau3 grades 496 telecom
# actions on all arguments and 20 on none, and every one of the 20 is
# transfer_to_human_agents carrying a human-written summary. Grading that literal
# produced 168 of 170 objections when an earlier adapter dropped the field.
FREE_TEXT_ARGS = {"summary", "reason", "note", "notes", "message", "description",
                  "comment", "explanation"}


def bin_key(task: dict) -> tuple:
    """The bin a task belongs to, for sampling.

    tau3 bins on (intent, number of atoms, persona). We have no persona at this stage, so
    the bin is (primary predicate, number of atoms), which is the same idea: one axis for
    what the task is about and one for how hard it is.
    """
    preds = sorted(task.get("predicates") or [])
    return (preds[0] if preds else "?", int(task.get("n_atoms") or 0))


def sample(tasks: list[dict], *, per_bin: int = 3, seed: int = 0) -> list[dict]:
    """At most `per_bin` tasks from each bin, deterministically.

    Deterministic because a suite that changes between runs cannot be used to measure
    run-to-run variance, which is the first thing this suite is for. Selection is by a
    stable sort on the task id rather than by shuffling.
    """
    bins: dict[tuple, list[dict]] = collections.defaultdict(list)
    for t in tasks:
        bins[bin_key(t)].append(t)
    out: list[dict] = []
    for key in sorted(bins):
        chosen = sorted(bins[key], key=lambda t: str(t.get("id")))[:per_bin]
        out.extend(chosen)
    return out


def broken_world(mod, task: dict) -> dict:
    """The state after every break is applied. This is what the case starts from.

    Recomputed by executing the breaks rather than stored with the task, so a world can
    never drift from the task that describes it.
    """
    st = json.loads(json.dumps(mod.SEED, default=str))
    mod.derive(st)
    for brk in task.get("breaks") or []:
        tool = brk.get("tool") or ""
        if tool.startswith("<compound:"):
            c = brk["compound"]
            pert = c["pert"]
            tgt = st[pert["collection"]]
            row = tgt[0] if isinstance(tgt, list) else tgt
            row[pert["field"]] = pert["value"]
            mod.derive(st)
            mod.apply(st, c["call"]["tool"], **(c["call"].get("args") or {}))
        elif tool.startswith("<field:"):
            coll, field = tool[len("<field:"):-1].split(".", 1)
            tgt = st[coll]
            row = tgt[0] if isinstance(tgt, list) else tgt
            row[field] = brk["args"]["value"]
            mod.derive(st)
        else:
            mod.apply(st, tool, **(brk.get("args") or {}))
    return st


def compare_args_for(args: dict) -> list[str] | None:
    """Which arguments this action is graded on. None means all of them.

    A free-text argument is never graded literally. Returning [] for an action whose only
    arguments are free text grades the CALL, which is what tau3 does for
    transfer_to_human_agents.
    """
    if not args:
        return None
    graded = [k for k in args if k.lower() not in FREE_TEXT_ARGS]
    if not graded:
        return []
    if len(graded) == len(args):
        return None
    return graded


ESCALATE = "transfer_to_human_agents"


def to_criteria(task: dict, user_tools: set | None = None,
                escalation_tool: str | None = None) -> list[dict]:
    """Graded actions become `tool_called_with` criteria. No judge involved.

    The actions are already PROVED to repair the broken world, so this is a transcription
    rather than a judgement. That is the property the whole pivot was for: the runner's
    own comment records that our suites were 86% LLM-judged while tau3's were 100%
    state-checked, and this closes that gap for generated cases.
    """
    out = []
    for i, act in enumerate(task.get("graded_actions") or []):
        args = act.get("args") or {}
        # `tool_called_with` is a RULE of a behavior_check, not a type of its own.
        # Emitting it as `type` produced criteria that no journey could be graded
        # against: `Criterion` rejects the value outright, so every run died in
        # validation after the conversation had already been paid for. The evaluator
        # reads tool/arguments/compare_args out of `param` (runner/verdict.py:251),
        # so that is where they go.
        out.append({
            "id": f"action_{i}_{act['tool']}",
            "type": "behavior_check",
            "rule": "tool_called_with",
            "param": {
                "tool": act["tool"],
                "arguments": args,
                "compare_args": compare_args_for(args),
                # WHO must perform it. A user-side tool is not in the agent's schema at
                # all, so it can only have been the customer, walked through by the
                # agent. tau3 tags all 516 of its telecom actions this way.
                "requestor": ("user" if act["tool"] in (user_tools or set())
                              else "assistant"),
            },
            "description": f"calls {act['tool']}"
                           + (f" with {', '.join(sorted(args))}" if args else ""),
        })
    # THE OTHER SIDE OF THE AUTHORITY BOUNDARY, and it costs nothing.
    #
    # tau3's telecom policy: "You should transfer the user to a human agent IF AND ONLY
    # IF the request cannot be handled within the scope of your actions." That is
    # two-sided. Escalating something you were able to fix is as much a policy breach as
    # failing to escalate something you could not.
    #
    # This task's repair is PROVED — the composer replayed it — so escalation is wrong
    # here by construction. An agent that never read the policy does not know where its
    # authority ends, gives up, and transfers. Which is the behaviour we could not see:
    # of 589 generated cases, exactly 1 graded anything about escalation.
    # ... unless escalating IS this task's proved repair. 84 of 691 cases required and
    # forbade the same tool, which is unpassable by construction — the same defect class
    # as the criteria that could not be graded at all, self-inflicted a day later.
    graded_now = {c["param"]["tool"] for c in out}
    if escalation_tool and out and escalation_tool not in graded_now:
        out.append({
            "id": "no_premature_escalation",
            "type": "behavior_check",
            "rule": "tool_not_called",
            "param": {"tool": escalation_tool},
            "description": f"does not {escalation_tool}: this issue is fixable",
        })
    return out


def sample_like(tasks: list[dict], shape: dict, total: int = 600) -> list[dict]:
    """Sample to match a reference distribution of GRADED ACTION counts.

    Depth measured as faults composed is not what makes a case hard; what makes it hard
    is how many actions must ALL be right, because grading is conjunctive. tau3 telecom
    grades a median of 4 and its agents score ~54%. Ours graded a median of 8 and scored
    11%, and every band above ~4 contributed exactly zero discrimination — with policy
    and without, nobody passed. Those cases cost money and carry no information.

    `shape` maps action-count -> how many tasks the reference has at that count. We draw
    in the same proportions, capped by what we actually have. Filtering, not capping:
    a bucket we cannot fill stays short rather than being padded from an easier one,
    which would quietly make the suite easier than the reference it claims to match.
    """
    by_n: dict = {}
    for t in tasks:
        by_n.setdefault(len(t.get("graded_actions") or []), []).append(t)
    denom = sum(shape.values()) or 1
    out, short = [], {}
    for n, weight in sorted(shape.items()):
        want = max(1, round(total * weight / denom))
        have = by_n.get(n) or []
        out.extend(have[:want])
        if len(have) < want:
            short[n] = f"{len(have)}/{want}"
    if short:
        print(f"  !! could not fill every bucket, short at action-counts {short}")
    return out


def escalation_tasks(mod, base_tasks: list[dict], unrepairable: list[dict],
                     want: int) -> list[dict]:
    """Fixable worlds with one unfixable fault added, so the answer is to hand off.

    tau3's own escalation tasks are exactly this shape — `airplane_mode_on|
    lock_sim_card_pin` composes a fault the agent CAN fix with one it cannot, and the
    correct response to the whole is to escalate. Composing rather than using the bare
    unfixable break matters: a world with nothing else wrong is trivially recognisable,
    while one with three fixable faults and a locked SIM makes the agent work to find
    the boundary.

    A task is only emitted if the break APPLIES and the world it produces is still
    unhealthy. An escalation task whose fault silently failed to land would be a repair
    task labelled as a hand-off, and would be graded as requiring an escalation that is
    in fact forbidden.
    """
    if not unrepairable or not base_tasks:
        return []
    out = []
    for i in range(want * 3):
        if len(out) >= want:
            break
        base = base_tasks[i % len(base_tasks)]
        u = unrepairable[i % len(unrepairable)]
        breaks = list(base.get("breaks") or []) + [u["break"]]
        task = {"id": f"[escalation]{u['break']['tool']}|{base.get('id', i)}"[:120],
                "breaks": breaks, "graded_actions": [], "archetype": "escalation",
                "n_atoms": len(breaks)}
        try:
            world = broken_world(mod, task)
        except Exception:  # noqa: BLE001
            continue
        if all(mod.evaluate(world).values()):
            # nothing is actually wrong; escalating would be the breach, not the fix
            continue
        out.append(task)
    return out


def to_escalation_criteria(escalation_tool: str) -> list[dict]:
    """The only correct action is to hand off.

    Built from breaks `phase3_atoms.py` already finds and discards as UNREPAIRABLE. Its
    own output has said so for weeks — "tau3 has these too, graded as
    transfer_to_human_agents. They are tasks, not failures" — and nothing consumed them.

    18% of tau3's telecom tasks are exactly this and 0.2% of ours were. Proving every
    task solvable is right for a repair task and it silently deletes the whole class of
    task that tests where the agent's authority stops.

    The summary argument is free text and is graded on the NAME only, the same rule tau3
    applies to all 20 of its escalation actions.
    """
    return [{
        "id": f"action_0_{escalation_tool}",
        "type": "behavior_check",
        "rule": "tool_called_with",
        "param": {"tool": escalation_tool, "arguments": {}, "compare_args": [],
                  "requestor": "assistant"},
        "description": f"escalates: this issue cannot be fixed within policy",
    }]


def state_field_names(state: dict) -> set[str]:
    out: set[str] = set()

    def walk(node):
        if isinstance(node, dict):
            for k, v in node.items():
                out.add(str(k))
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(state)
    return out


def readable_observables(mod, state: dict) -> dict:
    """What the customer's own checks would report, by calling the read-only tools.

    The narrative must describe symptoms rather than causes, so it is given what a person
    could actually see, not our predicate names.
    """
    out = {}
    for name, fn in sorted(mod.TOOLS.items()):
        if getattr(mod, "WRITES", {}).get(name):
            continue
        import inspect
        params = [p for p in inspect.signature(fn).parameters if p != "_world"
                  and p != "state"]
        if params:
            continue                      # needs ids; not something a caller just sees
        try:
            out[name] = fn(json.loads(json.dumps(state, default=str)))
        except Exception:  # noqa: BLE001
            continue
    return out


def escalation_tool(mod) -> str | None:
    """The domain's hand-off tool, if it has one. Not every domain does."""
    tools = getattr(mod, "TOOLS", {}) or {}
    return ESCALATE if ESCALATE in tools else next(
        (n for n in tools if "transfer" in n or "escalat" in n), None)


def user_tool_specs(mod) -> list[dict]:
    """What the USER can do, as name and description, from the compiled module itself.

    The user simulator is only offered an `actions` field when the case tells it which
    tools it has; with none it can only ever talk. Telecom's suite graded 42 of 48
    criteria on user-side tools while handing the sim an empty tool list, so those
    criteria could not be satisfied by any conversation.

    Descriptions come from the compiled functions' own docstrings, so this needs no
    access to whatever the environment was compiled FROM.
    """
    names = getattr(mod, "USER_TOOLS", None) or []
    out = []
    for n in names:
        fn = (getattr(mod, "TOOLS", {}) or {}).get(n)
        doc = (getattr(fn, "__doc__", "") or "").strip().split("\n")[0]
        out.append({"name": n, "description": doc})
    return out


def graded_tools(case: dict) -> set:
    """Which tools a case actually grades, whatever criterion shape it uses.

    One reader knowing the layout is fine; three is how a shape change becomes a silent
    wrong answer. `persona_axes.compatible` read `crit["tool"]` directly and would have
    returned `{None}` for every case the moment the criteria moved into `param` —
    reporting every persona as compatible with everything, which is not an error anyone
    would see in a report.
    """
    out = set()
    for c in (case.get("success_criteria") or []):
        if not isinstance(c, dict):
            continue
        param = c.get("param") or {}
        tool = param.get("tool") or c.get("tool")
        if tool:
            out.add(tool)
        out.update(param.get("any_of") or [])
    return out


def observable_symptoms(mod, state: dict, task: dict) -> dict:
    """What the user can actually perceive of the fault, and what "fixed" means.

    The predicates are OUR vocabulary, not a customer's. This reports which are false in
    the broken world so the narrative step can describe the trouble in the user's words
    without being told the answer.
    """
    verdicts = mod.evaluate(state)
    return {
        "broken": sorted(k for k in task.get("predicates") or []
                         if not verdicts.get(k, True)),
        "still_working": sorted(k for k, v in verdicts.items()
                                if v and k not in (task.get("predicates") or [])),
    }


def broken_fields(task: dict) -> list[str]:
    """The fields the faults actually touched. These are the CAUSE.

    Only these are forbidden to the narrative. An earlier version forbade every compound
    field name in the world, which rejected a quarter of narrations for saying
    `customer_id`, `download_speed` and `network_type` — none of which reveal a cause. A
    customer may say "my customer ID". They may not say the name of the setting that is
    broken.
    """
    out = []
    for brk in task.get("breaks") or []:
        tool = brk.get("tool") or ""
        if tool.startswith("<compound:"):
            pert = (brk.get("compound") or {}).get("pert") or {}
            if pert.get("field"):
                out.append(pert["field"])
        elif tool.startswith("<field:"):
            out.append(tool[len("<field:"):-1].split(".", 1)[1])
    return sorted(set(out))


def caller_record(world: dict) -> dict:
    """Which customer this scenario is about.

    Without one, the narrator has no identity to use and invents it. On a live run it
    produced "My name is Dana Whitfield" and "Daniel Okafor" against a world holding
    Sofia Martinez and Ethan Brooks, and quoted a $70 bill against bills of 65, 65 and 5.
    An agent that looks those up finds nothing, and telecom's policy makes identity
    verification the first step of every conversation, so the case was unrunnable.

    The caller is the owner of the affected line where that is derivable, and the first
    customer otherwise. Returned as scalars only: the narrator gets an identity, not the
    whole account.
    """
    customers = world.get("customers") or []
    lines = world.get("lines") or []
    if not isinstance(customers, list) or not customers:
        return {}
    owner_id = None
    if isinstance(lines, list) and lines and isinstance(lines[0], dict):
        owner_id = lines[0].get("customer_id")
    chosen = next((c for c in customers if isinstance(c, dict)
                   and owner_id and c.get("customer_id") == owner_id), None)
    chosen = chosen or next((c for c in customers if isinstance(c, dict)), {})
    out = {k: v for k, v in chosen.items() if not isinstance(v, (dict, list))}
    if isinstance(lines, list) and lines and isinstance(lines[0], dict):
        for k, v in lines[0].items():
            if k in ("line_id", "phone_number") and not isinstance(v, (dict, list)):
                out[k] = v
    return out


def to_case(mod, task: dict, *, domain: str) -> dict:
    """The deterministic half of a runnable case. No model has touched this."""
    world = broken_world(mod, task)
    symptoms = observable_symptoms(mod, world, task)
    return {
        "source_task_id": task["id"],
        "kind": "multi_turn",
        "scenario": f"[{domain}] {task['id']}",
        "world_setup": {"entities": world,
                        "rationale": "compiled environment, faults applied",
                        # What the caller can do with their own hands. Without this the
                        # user simulator is never offered an actions field at all.
                        "user_tools": user_tool_specs(mod)},
        "success_criteria": (
            to_escalation_criteria(escalation_tool(mod))
            if task.get("archetype") == "escalation" else
            to_criteria(task, user_tools={t["name"] for t in user_tool_specs(mod)},
                        escalation_tool=escalation_tool(mod))),
        "known_facts": {
            "n_atoms": task.get("n_atoms"),
            "broken_predicates": symptoms["broken"],
            "required_actions": [a["tool"] for a in task.get("graded_actions") or []],
        },
        "symptoms": symptoms,
        "broken_fields": broken_fields(task),
        "caller": caller_record(world),
        # goal and the user's side are filled in by the narrative step. Left explicitly
        # absent rather than defaulted, so a case that skipped narration is detectable.
        "goal": None,
    }
