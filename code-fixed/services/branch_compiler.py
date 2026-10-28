"""Compile BRANCHES from an agent's own policy, tools and observables.

WHAT A BRANCH IS

A point where the agent must choose and the correct choice depends on ITS OWN POLICY
rather than on general competence. Measured across all three tau3 domains on 2026-08-18,
a conditional branch is the only element present in the majority of tasks everywhere:
100% of telecom, 74% of airline, 66% of retail. Our 596 generated cases contained
approximately zero — every apparent match was narrative ("nothing loads unless I'm on
Wi-Fi") rather than a rule governing what the counterparty does at a decision point.

That absence is the concrete form of our suite moving 3.6 points when the whole policy is
deleted where tau3's moves 21.9.

THE TAXONOMY IS FIXED; WHICH KINDS A DOMAIN HAS IS DERIVED

tau3 telecom uses thresholds and authority limits, airline uses refusals and eligibility,
retail uses substitution. None of the three uses all seven. So a domain having a kind the
benchmark never wrote for it is a genuine addition rather than a rephrasing.

THE SPLIT, AS EVERYWHERE ELSE HERE

A model PROPOSES branches from the policy; the compiled environment DECIDES which are
real. Validation is the load-bearing half: a branch naming a tool that does not exist, an
observable that does not exist, or a value the world can never hold, is dropped rather
than repaired.
"""
from __future__ import annotations

import json
import os

import litellm

from services import llm_json

MODEL_ENV = "BRANCH_MODEL"

KINDS = ("threshold", "substitution", "refusal", "eligibility", "authority",
         "ordering", "prohibition")

_PROMPT = """\
A support agent follows the policy below. Find the DECISION POINTS in it: places where a
capable agent that had NOT read this policy would act differently from one that had.

POLICY:
{policy}

TOOLS the agent can call:
{tools}

WHAT THE CUSTOMER CAN PERCEIVE, as `field: [values it can take]`. A branch about
"when is this resolved" must be phrased in these terms, because the customer can only
judge what they can see:
{observables}

Each decision point is one of these kinds:

- threshold     the customer accepts only a certain level of an observable, and near
                misses do not count
- substitution  their first choice is unavailable and they will take a stated alternative
- refusal       they decline unless some condition holds
- eligibility   the policy permits the action only under stated conditions
- authority     it is outside what the agent may do; the policy says to escalate
- ordering      the policy requires one action before another
- prohibition   the policy forbids an action

Only include a decision point where the POLICY decides the answer. If ordinary common
sense settles it, leave it out — it cannot distinguish an agent that read the policy from
one that did not.

For each, give:
- `kind`
- `rule`: the policy's requirement, in one sentence, quoting the policy where you can
- `counterparty_line`: ONE sentence written in second person for the customer to follow,
  stating what they will accept, refuse, or take instead. This is what they say and do,
  never what the agent should do.

  IT MUST NOT ASSERT THE SITUATION, ONLY THE DISPOSITION. Write what the customer would
  agree to IF it arises, never that it has arisen. "If the agent tells you a bill must be
  settled first, you accept and ask them to take the payment" — NOT "you accept that you
  must first pay every overdue bill", which tells the customer they HAVE an overdue bill.
  That is account state the agent is supposed to look up and tell them, and handing it to
  the customer removes their reason to ask. tau3 words these as pure dispositions: "If the
  agent asks you to pay a bill, you accept", "you are willing to refuel 2 GB if necessary,
  but you do not want to change your plan".
- `observable`: the perceivable field it is judged on, or null
- `accept_values` / `reject_values`: for a threshold, which values of that observable
  count as resolved and which explicitly do not
- `precondition`: the situation that must ALREADY be true for this decision to arise, as
  {{"field": "<collection.field from the records>", "op": "<operator>", "value": <value>}},
  or null if it arises in any conversation. A customer must never be made to say "my
  contract has ended" or "I locked my SIM" in a world where that is not so.

  THE OPERATOR IS REQUIRED AND THE VALUE MUST BE A VALUE, NEVER A DESCRIPTION. A previous
  extraction wrote {{"field": "line.contract_end_date", "value": "in the past"}}. No world
  holds the string "in the past", so that rule was skipped by every case and five of the
  seven kinds below were never tested at all. Write the comparison instead:

      {{"field": "lines.contract_end_date", "op": "before", "value": "$now"}}
      {{"field": "phone.apn_settings_status", "op": "in", "value": ["incorrect", "not set"]}}
      {{"field": "lines.data_usage_gb", "op": "gt", "value": 50}}
      {{"field": "lines.status", "op": "eq", "value": "Suspended"}}

  Operators: eq, ne, in, not_in, lt, lte, gt, gte, before, after, exists, absent,
  truthy, falsy. `in` and `not_in` take a LIST. `before` and `after` take an ISO date or
  the literal "$now". Use the field names and the value vocabulary shown in the records
  above; a value the field never takes makes the rule dead.
- `required_tools` / `forbidden_tools`: tool names the policy requires or forbids here
- `before` / `after`: for ordering, the two tool names

Return JSON only:
{{"branches": [{{"kind": "...", "rule": "...", "counterparty_line": "...",
  "observable": null, "accept_values": [], "reject_values": [], "precondition": null,
  "required_tools": [], "forbidden_tools": [], "before": null, "after": null}}]}}
"""


def observable_values(mod, tasks: list[dict] | None = None, sample: int = 150) -> dict:
    """`field: [values it can take]` — asked of the environment, not scraped from it.

    Three attempts at reading the value space out of the module's SOURCE all failed:
    matching `==` alone found one value per field and would have made every threshold
    vacuous; scanning whole lines pulled in field names; a membership pattern matched
    across unrelated code and gave every field an identical list. Regex-tuning a parser
    is the wrong tool.

    So the world is asked instead. Start from the seed, apply the recorded breaks from
    real composed tasks, and record what each perceivable field actually holds. The
    healthy value comes from the seed; the values a threshold must be able to REJECT come
    from the broken worlds. That is ground truth by construction and it needs no parsing.
    """
    import json as _json
    from services.scenario_check import field_vocabulary
    from services import task_adapter

    names = list(getattr(mod, "OBSERVABLES", None) or [])
    seen: dict[str, set] = {n: set() for n in names}

    def record(world):
        vocab = field_vocabulary(world, limit=20)
        for n in names:
            for v in (vocab.get(n) or []):
                if str(v).strip():
                    seen[n].add(str(v))

    base = _json.loads(_json.dumps(getattr(mod, "SEED", {}) or {}, default=str))
    try:
        mod.derive(base)
    except Exception:  # noqa: BLE001
        pass
    record(base)

    for t in (tasks or [])[:sample]:
        try:
            record(task_adapter.broken_world(mod, t))
        except Exception:  # noqa: BLE001
            continue
    return {n: sorted(v)[:12] for n, v in seen.items() if v}


def validate(branch: dict, mod, obs: dict) -> tuple[bool, str]:
    """Does the compiled world support this branch? -> (ok, why not)

    Dropped, never repaired. A branch that survives here is one the environment can
    actually produce both sides of; anything else is a sentence that reads well and
    grades nothing.
    """
    from services import branch_predicate as BP
    from services.scenario_check import find_field

    # THE PRECONDITION MUST NAME A FIELD THAT EXISTS AND CARRY AN OPERATOR.
    #
    # Measured 2026-08-19: after typing was introduced, two of thirteen branches still
    # attached to nothing because the extractor invented `lines.roaming_status` and
    # `phone.messaging_app_permissions`. The real fields are `roaming_enabled` and the
    # `app_permissions` collection. A precondition naming a field no world holds is
    # unanswerable on every case, so the rule is dead while looking present — the same
    # shape as the prose preconditions this replaces, one level down.
    pre = branch.get("precondition")
    if pre:
        if not BP.is_typed(pre):
            return False, f"precondition is not evaluable: {BP.describe(pre)}"
        base = json.loads(json.dumps(getattr(mod, "SEED", {}) or {}, default=str))
        try:
            mod.derive(base)
        except Exception:  # noqa: BLE001
            pass
        if not find_field(base, str(pre.get("field"))):
            return False, f"precondition names no field in the world: {pre.get('field')}"

    kind = branch.get("kind")
    if kind not in KINDS:
        return False, f"unknown kind {kind!r}"
    if not (branch.get("counterparty_line") or "").strip():
        return False, "no counterparty line"

    tools = getattr(mod, "TOOLS", {}) or {}
    for key in ("required_tools", "forbidden_tools"):
        for t in branch.get(key) or []:
            if t not in tools:
                return False, f"{key} names {t!r}, which the environment cannot execute"
    for key in ("before", "after"):
        t = branch.get(key)
        if t and t not in tools:
            return False, f"{key} names {t!r}, which the environment cannot execute"

    if kind == "threshold":
        field = branch.get("observable")
        if field not in obs:
            return False, f"observable {field!r} is not perceivable in this environment"
        space = set(obs.get(field) or [])
        acc = {str(v) for v in (branch.get("accept_values") or [])}
        rej = {str(v) for v in (branch.get("reject_values") or [])}
        if not acc or not rej:
            return False, "threshold needs both accepted and rejected values"
        if not acc <= space:
            return False, f"accepted values {sorted(acc - space)} cannot occur"
        # THE VACUITY CHECK. If the rejected values can never occur, the customer can
        # never be made to reject anything and the branch is decorative.
        if not rej & space:
            return False, f"rejected values {sorted(rej)} can never occur"
        if acc & rej:
            return False, "a value cannot be both accepted and rejected"
        if len(acc) >= len(space) - 1 and len(space) > 2:
            return False, (f"accepts {len(acc)} of {len(space)} possible values — too "
                           f"permissive to reject anything")
    else:
        # DECORATIVE OBSERVABLES, stripped. Read at checkpoint C1: the "max 2GB refuel"
        # branch came back judged on connection_quality with everything except "no
        # connection" accepted. The model attaches a plausible-looking observable to
        # kinds that do not need one, and an accept set covering almost the whole space
        # cannot reject anything. Only a threshold is judged on an observable.
        branch.pop("observable", None)
        branch.pop("accept_values", None)
        branch.pop("reject_values", None)
    if kind == "ordering" and not (branch.get("before") and branch.get("after")):
        return False, "ordering needs both before and after"
    if kind == "prohibition" and not (branch.get("forbidden_tools") or []):
        return False, "prohibition names no tool"
    return True, ""


def to_criteria(branch: dict, idx: int) -> list[dict]:
    """The gradeable half. Uses only rules the evaluator already supports."""
    out = []
    for t in branch.get("required_tools") or []:
        out.append({"id": f"branch{idx}_req_{t}", "type": "behavior_check",
                    "rule": "tool_called_with",
                    "param": {"tool": t, "arguments": {}, "compare_args": []},
                    "description": f"policy requires {t}: {branch.get('rule', '')[:90]}"})
    for t in branch.get("forbidden_tools") or []:
        out.append({"id": f"branch{idx}_forb_{t}", "type": "behavior_check",
                    "rule": "tool_not_called", "param": {"tool": t},
                    "description": f"policy forbids {t}: {branch.get('rule', '')[:90]}"})
    if branch.get("kind") == "ordering":
        out.append({"id": f"branch{idx}_order", "type": "behavior_check",
                    "rule": "tool_called_before",
                    "param": {"tool": branch["before"], "after": branch["after"]},
                    "description": f"{branch['before']} must precede {branch['after']}"})
    return out


async def compile_branches(policy: str, mod, *, tasks: list[dict] | None = None,
                           tool_names: list[str] | None = None,
                           model: str | None = None) -> tuple[list[dict], list[tuple]]:
    """-> (surviving branches, [(dropped_branch, why), ...])

    Both halves are returned. A compiler that reports only what survived hides whether
    the model produced nothing or produced twenty unusable things.
    """
    model = model or os.getenv(MODEL_ENV, "claude-opus-5")
    obs = observable_values(mod, tasks)
    names = tool_names or sorted(getattr(mod, "TOOLS", {}) or {})
    try:
        resp = await litellm.acompletion(
            model=model, timeout=240, max_tokens=8000,
            messages=[{"role": "user", "content": _PROMPT.format(
                policy=(policy or "")[:40_000],
                tools=json.dumps(names, indent=1)[:6000],
                observables=json.dumps(obs, indent=1)[:4000])}])
        choice = resp.choices[0]
        raw = (choice.message.content or "").strip()
        if not raw:
            print(f"  [branch] empty response, "
                  f"finish_reason={getattr(choice, 'finish_reason', '')}")
            return [], []
    except Exception as exc:  # noqa: BLE001
        print(f"  [branch] request failed: {type(exc).__name__}: {exc}")
        return [], []
    data = llm_json.loads_or_none(raw, want=dict, label="branch")
    if not data:
        return [], []
    kept, dropped = [], []
    for b in data.get("branches") or []:
        if not isinstance(b, dict):
            continue
        if b.get("kind") == "threshold" and b.get("observable") in obs:
            # The world decides what counts as unacceptable, not the model. It listed
            # good/fair/poor for download_speed and omitted "no connection", so the
            # threshold never fired on the dead-connection worlds that need it most.
            space = {str(v) for v in obs[b["observable"]]}
            acc = {str(v) for v in (b.get("accept_values") or [])}
            b["reject_values"] = sorted(space - acc)
        ok, why = validate(b, mod, obs)
        (kept if ok else dropped).append(b if ok else (b, why))
    return kept, dropped


def _observed(world: dict, dotted: str):
    from services.scenario_check import find_field
    hits = find_field(world, dotted)
    return str(hits[0][1]) if hits else None


def live_branches(case: dict, branches: list[dict]) -> list[dict]:
    """Which branches this case actually presents. Mechanical, no model.

    Three gates, each added because reading the output showed what happens without it:

    PRECONDITION  a branch may only attach where its situation is TRUE of this world.
                  Without this the customer was made to say "I entered my PIN wrong too
                  many times" and "my contract already ended" in cases where neither
                  held — the ungrounded-claim failure `scenario_check` exists to stop,
                  reintroduced through a side door.
    THRESHOLD     live only when the world currently shows a REJECTED value. That is
                  exactly when the customer has something to refuse.
    RELEVANCE     otherwise, live when it touches a tool this case already grades.
    """
    from services.scenario_check import find_field
    from services import branch_predicate as BP
    world = (case.get("world_setup") or {}).get("entities") or {}
    # REQUIRED tools only. Counting forbidden ones too made every branch touching
    # transfer_to_human_agents relevant to every case, because every case carries the
    # no-premature-escalation criterion — so all 20 cases got the identical branch.
    graded = {c["param"]["tool"] for c in (case.get("success_criteria") or [])
              if c.get("rule") == "tool_called_with" and (c.get("param") or {}).get("tool")}
    out = []
    for b in branches:
        pre = b.get("precondition") or None
        if isinstance(pre, dict) and pre.get("field"):
            hits = find_field(world, str(pre["field"]))
            verdict = BP.evaluate(pre, world, [v for _, v in hits])
            # THREE-VALUED. True attaches, False skips, and UNKNOWN also skips but is
            # recorded on the function so a caller can report how many rules the world
            # could not answer. Folding unknown into False is what hid 14 untyped
            # preconditions and left five of seven kinds untested.
            if verdict is None:
                live_branches.unanswerable.append(
                    {"kind": b.get("kind"), "precondition": BP.describe(pre)})
                continue
            if verdict is False:
                continue
        if b.get("kind") == "threshold":
            now = _observed(world, b.get("observable") or "")
            if now is not None and now in {str(v) for v in (b.get("reject_values") or [])}:
                out.append(b)
            continue
        tools = set(b.get("required_tools") or []) | set(b.get("forbidden_tools") or [])
        tools |= {b.get("before"), b.get("after")} - {None}
        if tools & graded:
            out.append(b)
    return out


live_branches.unanswerable = []


def untyped(branches: list[dict]) -> list[dict]:
    """Branches whose precondition the world cannot answer. Reported, never dropped."""
    from services import branch_predicate as BP
    return [b for b in branches
            if b.get("precondition") and not BP.is_typed(b.get("precondition"))]


def apply_branches(case: dict, branches: list[dict], max_lines: int = 3) -> dict:
    """Add the customer's decision rules and the criteria they imply.

    NEVER FORBIDS A TOOL THE CASE REQUIRES. Read at checkpoint C2: a case whose proved
    repair is `reseat_sim_card` was given a branch forbidding `reseat_sim_card`, making
    it unpassable by construction — the same defect as this morning arriving by a new
    route. A conflict means the branch does not apply here, so it is dropped.

    Required-tool criteria are left out entirely: the case's proved repair already grades
    what must happen, and piling on more required calls is how 42 of 48 criteria became
    unpassable. A branch earns its place by changing what the customer accepts and by
    catching what the policy forbids.

    Capped, because six decision rules on one customer produced three different stances
    on being transferred to a human in a single conversation.
    """
    required = {c["param"]["tool"] for c in (case.get("success_criteria") or [])
                if c.get("rule") == "tool_called_with" and (c.get("param") or {}).get("tool")}
    live = [b for b in live_branches(case, branches)
            if not (set(b.get("forbidden_tools") or []) & required)]
    if not live:
        return case
    live = live[:max_lines]
    # MERGED, not replaced. The case already carries the acceptance derived from its own
    # broken world — what the customer must SEE before agreeing it is fixed — and
    # overwriting it here silently discarded that, leaving the customer free to stop as
    # soon as their stated complaint was met while the world was still broken underneath.
    case["acceptance"] = sorted(set(case.get("acceptance") or [])
                                | {b["counterparty_line"] for b in live
                                   if b.get("counterparty_line")})
    case["branch_kinds"] = sorted({b["kind"] for b in live})
    # deduped BY TOOL, not by id: two criteria forbidding the same tool with different
    # ids both appeared on one case.
    forbidden = {c["param"]["tool"] for c in (case.get("success_criteria") or [])
                 if c.get("rule") == "tool_not_called"}
    for i, b in enumerate(live):
        for crit in to_criteria(b, i):
            if crit["rule"] == "tool_called_with":
                continue
            tool = crit["param"].get("tool")
            if crit["rule"] == "tool_not_called":
                if tool in forbidden or tool in required:
                    continue
                forbidden.add(tool)
            case.setdefault("success_criteria", []).append(crit)
    return case
