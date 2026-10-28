"""Build the simulated customer's prompt from the policy, the graph and the world.

WHY THIS EXISTS

Our narrator was handed the device state and told to distribute it into the customer's
script — `wont_volunteer` gets "the diagnostic detail", `will_reveal_if_asked` is "what
they can check on request, in their words". Measured on 630 transcripts: the customer then
ANSWERS diagnostic questions instead of running the check, so the tool is never called and
every `requestor: user` criterion fails as "never called by the user". 63% of everything we
grade is user-side.

tau3 does the opposite, and its 114 telecom scenarios are the reference this module is
gated against:

    known_info   86 chars, three shapes only. Identity, plus ONE context fact that
                 activates a policy branch ("currently abroad in France", 64 of 114).
                 NO device state at any point.
    unknown_info exists in their schema and is used ZERO times. Ignorance is created by
                 OMISSION, not by declaration.
    the customer therefore cannot answer "what does your status bar show" from memory.
    It must call the tool.

THE INVARIANT THIS MODULE ENFORCES

    the customer's prompt contains no value that a user tool returns.

That is mechanically checkable against the compiled world, which is why it is the gate
rather than a similarity score. There is nothing to tune toward: either a value the world
holds appears in the prompt text or it does not.

WHAT IS GENERATED AND WHAT IS ASSEMBLED

tau3's `task_instructions` is 1071 chars of four stacked layers, and reading all 114 shows
how little of it is bespoke:

    epistemic rule   114/114 VERBATIM. A constant.
    consent           85/114. "willing to refuel 2.0 GB, but you do not want to change
                      your plan" — a threshold branch and a prohibition branch, spoken.
    acceptance        36/114. "resolved only when the speed test returns excellent and
                      nothing else" — an observable, its check tool, its healthy value.
    capability hints  29/114. "always check the status bar if the agent asks for status
                      information" — the family's own check steps.

Only `reason_for_call` needs prose. Five of six components come from artifacts already in
the repo: the caller record, `branches_*.json` counterparty lines, OBSERVABLES with their
reporting tools, and the policy graph's per-family checks.
"""
from __future__ import annotations

import inspect
import json
import os

import litellm

from services import llm_json
from services import narration_gate

MODEL_ENV = "NARRATOR_MODEL"


# tau3 repeats this block identically in all 114 telecom scenarios. It is the rule that
# forces a tool call instead of a recited answer, and it is the single thing our scenarios
# have never carried.
GROUNDING = (
    "Whenever the agent asks you about your device or your account, always ground your "
    "responses on the results of tool calls. Never make up the results of a tool call. "
    "If you have not run the relevant check, you do not know the answer — say so, or run "
    "the check.\n"
    "If a tool call does not return updated status information, you may need to perform "
    "another tool call to get the updated status.\n"
    "If you are unsure whether an action is necessary, ask the agent for clarification.\n"
    "If the agent asks you to perform several actions at once, say you can only do one at "
    "a time and ask them to take you through it step by step."
)


def reporting_tools(mod) -> dict:
    """observable field -> the user tools that REPORT it. -> {field: [tool, ...]}

    A tool reports an observable when it reads the field and writes nothing: that is a
    check, not a fix. Derived from the compiled module rather than listed, so a recompiled
    domain cannot silently drift from its own scenarios.
    """
    writes = getattr(mod, "WRITES", {}) or {}
    user_tools = set(getattr(mod, "USER_TOOLS", None) or [])
    out: dict[str, list[str]] = {}
    for obs in (getattr(mod, "OBSERVABLES", None) or []):
        field = obs.split(".")[-1]
        found = []
        for name in sorted(user_tools):
            fn = (getattr(mod, "TOOLS", {}) or {}).get(name)
            if fn is None or writes.get(name):
                continue                       # it changes the world; not a check
            try:
                src = inspect.getsource(fn)
            except (OSError, TypeError):
                continue
            if field in src:
                found.append(name)
        if found:
            out[obs] = found
    return out


def healthy_values(mod) -> dict:
    """What each observable reads on a healthy world. -> {field: value}"""
    import copy
    state = copy.deepcopy(mod.SEED)
    mod.derive(state)

    def read(dotted):
        node = state
        for part in dotted.split("."):
            if isinstance(node, list):
                node = node[0] if node else {}
            node = (node or {}).get(part) if isinstance(node, dict) else None
        return node
    return {o: read(o) for o in (getattr(mod, "OBSERVABLES", None) or [])}


# ---- the five mechanical components -----------------------------------------

def known_info(caller: dict, world: dict) -> str:
    """Identity, plus the ONE context fact that activates a policy branch.

    tau3's three shapes, in order of frequency across its 114 telecom tasks:
        "You are <name> with phone number <num>. You are currently abroad in France."
        "You are <name> with phone number <num>."
        "You are <name> with phone number <num>. You are currently at home in the <X>."

    The location clause is not colour. Roaming is a policy branch, so where the customer
    is decides which rules apply. Nothing else about the device or the account appears.
    """
    name = caller.get("full_name") or caller.get("name") or ""
    phone = caller.get("phone_number") or ""
    parts = [f"You are {name} with phone number {phone}."]
    city = caller.get("address_city")
    lines = [l for l in (world.get("lines") or [])
             if l.get("customer_id") == caller.get("customer_id")]
    abroad = any(l.get("roaming_enabled") is False for l in lines) if lines else False
    if abroad:
        parts.append("You are currently travelling outside your home network.")
    elif city:
        parts.append(f"You are currently at home in {city}.")
    return " ".join(parts)


def consent_lines(branches: list[dict], case: dict, mod) -> list[str]:
    """The customer's side of each policy decision that is LIVE in this world.

    tau3 writes these by hand — "You are willing to refuel 2.0 GB of data if necessary,
    but you do not want to change your mobile data plan" is a threshold branch and a
    prohibition branch spoken in the first person. We already compile the equivalent as
    `counterparty_line`, gated by a typed precondition, so this is a lookup rather than a
    generation.
    """
    from services import branch_compiler as BC
    out = []
    for b in BC.live_branches(case, branches or []):
        line = (b.get("counterparty_line") or "").strip()
        if not line or line in out:
            continue
        out.append(_as_disposition(line))
    return out


def _as_disposition(line: str) -> str:
    """A consent line must say what the customer WOULD accept, not what is true.

    Measured by the gate: our compiled `counterparty_line` reads "you accept that you must
    first pay every overdue bill before the suspension is lifted", which tells the customer
    they HAVE an overdue bill. That is account state the agent is supposed to look up and
    tell them, and stating it removes the reason to look. tau3's equivalent asserts
    nothing: "If the agent asks you to pay a bill, you accept."

    Hedged rather than rewritten, because the prose carries the policy decision and
    paraphrasing it with a model would put an unreviewed sentence between the branch and
    the customer.
    """
    line = line.rstrip(".")
    return f"If it comes up: {line[0].lower() + line[1:]}. Do not raise it yourself " \
           f"and do not assume it applies until the agent tells you it does."


def acceptance_lines(mod, broken: list[str], case_world: dict) -> list[str]:
    """When the customer believes it is fixed — stated as a TOOL RESULT.

    tau3: "You will consider the issue resolved only when the speed test returns
    excellent internet speed and nothing else. If it returns poor, fair or good, you will
    not consider the issue resolved."

    The frame is the whole point. Ours said "until the signal bars read excellent", which
    is a fact about the world the customer can assert without checking. Naming the tool
    and its healthy return means they cannot know whether they are satisfied until they
    run it.
    """
    reps, healthy = reporting_tools(mod), healthy_values(mod)
    import copy
    now = copy.deepcopy(case_world)
    try:
        mod.derive(now)
    except Exception:  # noqa: BLE001
        pass

    def read(dotted, state):
        node = state
        for part in dotted.split("."):
            if isinstance(node, list):
                node = node[0] if node else {}
            node = (node or {}).get(part) if isinstance(node, dict) else None
        return node

    # ONE condition, not one per observable. A first version emitted a line per broken
    # observable and produced "resolved only when check_network_status reports
    # 'excellent', and nothing else" immediately followed by the same sentence demanding
    # '5G' — two absolutes that cannot both hold. tau3 states exactly one, which is also
    # why its task_instructions is 1071 chars where ours ran to 1915.
    broken_obs = [(o, t) for o, t in reps.items()
                  if healthy.get(o) is not None and str(healthy[o]) != str(read(o, now))]
    if not broken_obs:
        return []
    # the check the customer would naturally reach for: the one covering the most
    # broken observables, tie-broken by name so the choice is reproducible
    from collections import Counter
    counts = Counter(t for _, tools in broken_obs for t in tools)
    tool = sorted(counts, key=lambda t: (-counts[t], t))[0]
    covered = [o for o, tools in broken_obs if tool in tools]
    wants = ", ".join(f"{healthy[o]!r}" for o in covered)
    natural = tool.replace("_", " ")
    return [f"You will consider the issue resolved only when the {natural} check reports "
            # NOT "however confident the agent sounds" — `confident` is a BEHAVIOUR word,
            # and the gate caught all 12 scenarios carrying it against tau3's 0 of 114.
            # Behaviour belongs to the persona axis, applied at run time; a scenario that
            # fixes it cannot be replayed by a different caller.
            f"{wants}, and nothing else. Any other result means it is not fixed, "
            f"whatever the agent says."]


def capability_hints(graph: list[dict], family: str, mod) -> list[str]:
    """What the customer can be asked to check, named as the ACTION not the answer.

    tau3: "Always check the status bar if the agent asks you for status information."
    It names the capability. It never states what the status bar would show.
    """
    user_tools = set(getattr(mod, "USER_TOOLS", None) or [])
    checks = []
    for fam in graph or []:
        if family and fam.get("family") != family:
            continue
        for step in fam.get("steps") or []:
            c = step.get("check")
            if c and c in user_tools and c not in checks:
                checks.append(c)
    if not checks:
        return []
    return [f"You can run these checks on your own device when the agent asks: "
            f"{', '.join('`' + c + '`' for c in checks)}. Run the check rather than "
            f"guessing what it would say."]


# ---- the one generated component --------------------------------------------

_SYMPTOM_PROMPT = """\
A customer is calling support. Write the ONE OR TWO SENTENCES they open with.

WHAT IS ACTUALLY WRONG, in internal terms the customer would never use:
{broken}

WHAT THEY CAN PERCEIVE — the difference between a working device and theirs. This is the
only evidence they have, and they cannot name the cause:
{perceived}

HOW BAD IT IS. Say it at this strength and no other. "gone" means the thing does not work
AT ALL — write that it is dead, not that it is slow, weak, patchy or unreliable, because
those describe a different fault and send the agent down the wrong path:
{severity}

Rules:
- Never name a tool, a setting, a field, or any internal term. They know the symptom, not
  the cause. "My texts with pictures won't send", never "my MMSC URL is empty".
- Do not describe the fix or suggest a next step. Diagnosing is the agent's job.
- Do NOT state any specific reading, setting value, or status. They have not checked
  anything yet — they only know what they noticed. "I have no signal", never "the status
  bar shows a crossed-out signal icon and 87% battery". Saying the connection is GONE is
  not a reading; it is what the person noticed, and it is required when severity says so.
- Write nothing about temperament, patience or technical confidence. Behaviour is applied
  separately at run time.

Return JSON only: {{"reason_for_call": "..."}}
"""


async def reason_for_call(broken: list[str], perceived: list[str],
                          model: str | None = None, observable: dict | None = None,
                          attempts: int = 2) -> str:
    """The symptom, in the customer's words. The only field needing a model.

    THE DEGREE HAS TO BE GIVEN. `perceived` deliberately carries field NAMES without
    their values so the customer still has to run the check — and that is right. But
    "signal_strength differs" does not say whether the signal is weak or absent, so the
    model wrote the commoner support complaint: every airplane-mode case in the
    2026-08-20 suite opened with "barely showing any bars" on a phone whose status bar
    read "Airplane Mode  No Signal". The agent then troubleshot speed, and the customer,
    keeping faith with its own story, told it "airplane mode is off".

    Severity is not a reading. It is computed from the environment's own observable
    values, says only gone-or-degraded, and leaks no setting.
    """
    model = model or os.getenv(MODEL_ENV, "claude-opus-5")
    dead = narration_gate.outage_evidence(observable or {})
    severity = ("gone — the connection does not work at all" if dead
                else "degraded — it works, but not properly")
    prompt = _SYMPTOM_PROMPT.format(
        broken=json.dumps(broken, indent=1)[:2000],
        perceived=json.dumps(perceived, indent=1)[:2000],
        severity=severity)

    for attempt in range(attempts):
        try:
            resp = await litellm.acompletion(
                model=model, timeout=120, max_tokens=600,
                messages=[{"role": "user", "content": prompt}])
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            print(f"  [meta_user] request failed: {type(exc).__name__}")
            return ""
        data = llm_json.loads_or_none(raw, want=dict, label="meta_user")
        text = str((data or {}).get("reason_for_call") or "").strip()
        if not text:
            return ""
        wrong = narration_gate.contradictions(text, observable or {})
        if not wrong:
            return text
        if attempt + 1 < attempts:
            prompt += "\n\n" + narration_gate.correction_note(wrong)
            continue
        print(f"  [meta_user] opening contradicts the device: {wrong[0][:90]}")
        return text          # kept, and reported, rather than losing the case silently
    return ""


# ---- assembly ----------------------------------------------------------------

def perceived_difference(mod, case_world: dict) -> list[str]:
    """Which observables differ from healthy — the customer's evidence, as FIELD NAMES.

    Passed to the symptom writer so it knows WHAT they noticed without being handed the
    reading itself. `signal_strength differs` is enough to write "I have no bars"; the
    actual value stays out of the prompt so the customer still has to check.
    """
    import copy
    healthy = healthy_values(mod)
    now = copy.deepcopy(case_world)
    try:
        mod.derive(now)
    except Exception:  # noqa: BLE001
        pass

    def read(dotted, state):
        node = state
        for part in dotted.split("."):
            if isinstance(node, list):
                node = node[0] if node else {}
            node = (node or {}).get(part) if isinstance(node, dict) else None
        return node
    return [obs for obs in healthy
            if str(healthy[obs]) != str(read(obs, now))]


def assemble(*, domain: str, caller: dict, case_world: dict, symptom: str,
             consent: list[str], acceptance: list[str], hints: list[str],
             persona: str | None = None) -> dict:
    """The tau3-shaped scenario. Same field names, so the gate compares like with like."""
    blocks = [GROUNDING]
    if hints:
        blocks.append("\n".join(hints))
    if consent:
        blocks.append("\n".join(consent))
    if acceptance:
        blocks.append("\n".join(acceptance))
    blocks.append("If the agent suggests actions that do not immediately fix the issue, "
                  "follow their guidance.")
    return {
        "persona": persona,
        "instructions": {
            "domain": domain,
            "reason_for_call": symptom,
            "known_info": known_info(caller, case_world),
            "unknown_info": None,
            "task_instructions": "\n\n".join(b for b in blocks if b.strip()),
        },
    }


async def build(*, mod, domain: str, case: dict, graph: list[dict],
                branches: list[dict], persona: str | None = None,
                model: str | None = None) -> dict:
    """A complete customer scenario for one case. -> tau3-shaped dict."""
    world = (case.get("world_setup") or {}).get("entities") or {}
    caller = case.get("caller") or {}
    broken = list((case.get("symptoms") or {}).get("broken") or [])
    perceived = perceived_difference(mod, world)
    # The customer's OWN readings, used only to fix the DEGREE of the complaint. The
    # values never enter the prompt; `outage_evidence` reduces them to gone-or-degraded.
    from services import task_adapter as _ta
    try:
        observable = _ta.readable_observables(mod, world)
    except Exception:  # noqa: BLE001
        observable = {}
    symptom = await reason_for_call(broken, perceived, model=model,
                                    observable=observable)
    return assemble(
        domain=domain, caller=caller, case_world=world, symptom=symptom,
        consent=consent_lines(branches, case, mod),
        acceptance=acceptance_lines(mod, broken, world),
        hints=capability_hints(graph, case.get("family") or "", mod),
        persona=persona,
    )
