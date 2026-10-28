"""The policy, read as a graph, so cases can be composed along it.

WHY

Our composer takes the product over all 28 atoms. It produced 21,606 distinct repair
sets against tau3's 95, and two thirds of our cases present several unrelated symptom
families at once — a payment problem and a VPN problem and an app-permission problem in
one call. The policy is a troubleshooting manual organised BY FAMILY, so no branch of it
covers that situation, and an agent carrying the policy gains nothing. Measured: the
policy changed almost nothing about what the agent did in our cases (premature
escalation 15 failures with it against 14 without; repair criteria 140 against 144).

The telecom policy's own headings are already the structure we need:

    Understanding and Troubleshooting Your Phone's Cellular Service
        Common Service Issues and Their Causes
        Diagnosing Service Issues
        Troubleshooting Service Problems -> Airplane Mode / SIM Card / APN / Suspension
    Understanding and Troubleshooting Your Phone's Mobile Data
        ... same shape
    (and MMS)

So a FAMILY is a section, an ISSUE is a subsection, and the diagnostic text gives the
order they are meant to be worked through. A case composed by walking one family follows
a path the policy actually describes; a case composed across families does not.

This does not invent structure. It reads the structure that is already written down.
"""
from __future__ import annotations

import json
import os
import re

import litellm

from services import llm_json

MODEL_ENV = "POLICY_GRAPH_MODEL"


def sections(policy: str) -> list[dict]:
    """Split the policy at its own headings. -> [{title, level, text}]"""
    out, pos = [], []
    for m in re.finditer(r"^(#{1,4})\s+(.+)$", policy, re.M):
        pos.append((len(m.group(1)), m.group(2).strip(), m.start()))
    for i, (lvl, title, start) in enumerate(pos):
        end = pos[i + 1][2] if i + 1 < len(pos) else len(policy)
        out.append({"title": title, "level": lvl, "text": policy[start:end]})
    return out


def families(policy: str, tool_names: list[str] | None = None) -> list[dict]:
    """The troubleshooting families, each with its whole subtree.

    Matched on the heading text rather than on position: the telecom policy has two
    top-level documents concatenated (agent rules, then a user-facing troubleshooting
    manual), so section INDEX means nothing and only the titles are stable.
    """
    secs = sections(policy)
    tools = set(tool_names or [])
    out = []
    for i, s in enumerate(secs):
        # A section qualifies two ways. TROUBLESHOOTING sections are the fault families.
        # PROCEDURE sections are the business rules — "Overdue Bill Payment", "Data
        # Refueling", "Line Suspension" — and they hold the eligibility and threshold
        # content the troubleshooting half does not: the 2GB refuel cap, "lift the
        # suspension only after the bills are paid, and not if the contract has ended".
        # tau3 has tasks on exactly those flows (overdue_bill_suspension repairs with
        # send_payment_request, make_payment, resume_line, reboot_device) and our graph
        # could not produce any of them.
        #
        # Matched on NAMING AT LEAST TWO TOOLS rather than on a list of titles, so this
        # carries to a domain whose sections are called something else.
        is_trouble = bool(re.search(r"troubleshoot", s["title"], re.I)) and s["level"] <= 2
        named = sum(1 for t in tools if t in s["text"])
        bullets = len(re.findall(r"^\s*[-*]\s+\S", s["text"], re.M))
        # A PROCEDURE has steps and few tools. A CATALOGUE has many tools and no steps —
        # "Diagnostic Actions (Read-only)" names sixteen and prescribes nothing, and
        # counting tools alone pulled both catalogues in while missing Line Suspension
        # and Data Refueling, which describe their steps in prose without ever writing a
        # tool's exact name.
        # A procedure PRESCRIBES. The policy says so in its own words — "you should",
        # "you can help", "to do so", "you are (not) allowed". A definition merely
        # describes: "Customer", "Line" and "Bill" list their fields as bullets and
        # prescribe nothing, and a bullet count alone pulled all three in.
        prescribes = bool(re.search(
            r"\byou (?:should|can|are allowed|are not allowed|must|need to)\b"
            r"|\bto do (?:so|this)\b|\bfollow these steps\b", s["text"], re.I))
        is_procedure = (not is_trouble and s["level"] <= 3 and prescribes
                        and named <= 6 and 250 < len(s["text"]) < 4000)
        if not (is_trouble or is_procedure):
            continue
        body = [s]
        for t in secs[i + 1:]:
            if t["level"] <= s["level"]:
                break
            body.append(t)
        out.append({"family": s["title"], "start": s_start(secs, i),
                    "kind": "troubleshooting" if is_trouble else "procedure",
                    "text": "".join(x["text"] for x in body),
                    "subsections": [x["title"] for x in body[1:]],
                    "span": len("".join(x["text"] for x in body))})
    # OUTERMOST ONLY. The telecom policy concatenates two documents with different
    # heading levels, so "Cellular Service" (level 1) and its own child "Troubleshooting
    # Service Problems" (level 2) BOTH match, and each family would be generated twice —
    # once whole and once as its own subsection.
    kept = []
    for f in out:
        if any(g is not f and g["start"] <= f["start"]
               and f["start"] + f["span"] <= g["start"] + g["span"] for g in out):
            continue
        kept.append(f)
    return kept


def s_start(secs, i):
    return sum(len(x["text"]) for x in secs[:i])


_PROMPT = """\
This is one section of a support agent's policy. Read it as a PROCEDURE.

Cover EVERY distinct procedure the section describes, not just the last one. A section
about billing, suspension, refuelling and escalation should yield steps for all of them.

{text}

TOOLS that exist. Use these names exactly, and never invent one:
{tools}

Return the ordered steps this section prescribes. A step is one issue the agent checks
for and what it does about it:

- `issue`: the fault, in the policy's own words
- `check`: the tool that DETECTS it, or null if the section names none
- `fix`: the tool that REPAIRS it. ONE tool name. If the policy authorises several
  equivalent tools, put the first in `fix` and the rest in `alternatives`
- `by`: "user" if the customer must perform the fix on their device, "agent" otherwise
- `requires`: tool names that must happen BEFORE this fix, in order, or []

Give them in the order the section says to work through them. That order is the point:
it is what an agent without the policy would not know.

Return JSON only:
{{"family": "...", "steps": [{{"issue": "...", "check": null, "fix": "...",
  "alternatives": [], "by": "user", "requires": []}}]}}
"""


async def compile_family(fam: dict, tool_names: list[str],
                         writes: dict | None = None,
                         model: str | None = None) -> dict | None:
    """One family's ordered steps, validated against the tools that exist."""
    model = model or os.getenv(MODEL_ENV, "claude-opus-5")
    try:
        resp = await litellm.acompletion(
            model=model, timeout=240, max_tokens=6000,
            messages=[{"role": "user", "content": _PROMPT.format(
                text=fam["text"][:24_000],
                tools=json.dumps(sorted(tool_names), indent=1)[:5000])}])
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # noqa: BLE001
        print(f"  [graph] {fam['family'][:40]}: {type(exc).__name__}")
        return None
    data = llm_json.loads_or_none(raw, want=dict, label="policy_graph")
    if not data:
        return None
    known = set(tool_names)
    steps, dropped = [], []
    for s in data.get("steps") or []:
        if not isinstance(s, dict) or not s.get("fix"):
            continue
        if s["fix"] not in known:
            # The model answers "a / b / c" when the policy authorises equivalents —
            # "you must first identify the customer" allows three lookup tools. Dropping
            # the whole step as an unknown tool cost the entire business-rules family.
            alts = [t.strip() for t in re.split(r"[/,]| or ", str(s["fix"]))
                    if t.strip() in known]
            if not alts:
                dropped.append((s.get("issue"), f"fix {s['fix']!r} does not exist"))
                continue
            s["fix"], s["alternatives"] = alts[0], alts[1:]
        s["alternatives"] = [t for t in (s.get("alternatives") or []) if t in known]
        # A FIX MUST CHANGE THE WORLD. The extractor offered `get_details_by_id` as the
        # repair for "user wants to change plan" — a read-only lookup. Grading it as a
        # repair asks the agent to fix something by looking at it, and the case can never
        # be satisfied by any action. The compiled module already records which tools
        # write, so this is decidable rather than a matter of judgement.
        if writes is not None and not (writes.get(s["fix"]) or []):
            dropped.append((s.get("issue"), f"fix {s['fix']!r} writes nothing"))
            continue
        s["check"] = s.get("check") if s.get("check") in known else None
        s["requires"] = [r for r in (s.get("requires") or []) if r in known]
        steps.append(s)
    return {"family": fam["family"], "steps": steps, "dropped": dropped}


def prerequisites(policy: str, family_names: list[str]) -> dict:
    """family -> the families it depends on, read from the policy's own words.

    The telecom policy states these under headings literally named "Prerequisites":

        Prerequisites for Mobile Data
            "the user must first have cellular service"
        Prerequisites for MMS
            "the user must have cellular service and mobile data (any speed)"

    So the families are not three disjoint lists, they are a graph. A customer
    complaining about MMS can be blocked by a CELLULAR SERVICE fault, and composing MMS
    walks in isolation cannot produce that case. tau3 has such tasks — measured, 9 of
    the 78 repair sets we could not generate span families this way, and every one is a
    prerequisite relation the policy spells out.
    """
    deps: dict[str, set] = {f: set() for f in family_names}
    for m in re.finditer(r"^#+\s*Prerequisites for (.+)$", policy, re.M | re.I):
        subject = m.group(1).strip()
        body = policy[m.end():m.end() + 900].split("\n#")[0]
        target = next((f for f in family_names
                       if subject.lower() in f.lower()), None)
        if not target:
            continue
        for other in family_names:
            if other == target:
                continue
            # the section names the guide it defers to, so match on its distinctive tail
            tail = re.sub(r"^Understanding and Troubleshooting\s*", "", other)
            key = re.sub(r"[^a-z ]", " ", tail.lower()).split("(")[0].strip()
            key = " ".join(key.split()[-2:])
            if key and key in body.lower():
                deps[target].add(other)
    # transitive: MMS requires mobile data, which requires cellular service
    for _ in range(len(family_names)):
        for f in family_names:
            for g in list(deps[f]):
                deps[f] |= deps.get(g, set())
    return {k: sorted(v) for k, v in deps.items()}


def walks(graph: list[dict], max_len: int = 3, deps: dict | None = None) -> list[dict]:
    """Paths through a family AND the families it depends on. -> [{family, steps}]

    A walk is still anchored to ONE complaint — that is what keeps a case coherent and
    on a path the policy describes. But the causes available to it include those of its
    prerequisites, because the policy says a mobile-data problem can be a cellular
    service problem underneath.

    Order is preserved: prerequisite steps come first, since the policy resolves the
    dependency before the complaint.
    """
    import itertools
    by_name = {g["family"]: g for g in graph}
    out = []
    for fam in graph:
        pool, seen = [], set()
        for dep in (deps or {}).get(fam["family"], []):
            for s in (by_name.get(dep, {}) or {}).get("steps", []):
                if s["fix"] not in seen:
                    pool.append(s); seen.add(s["fix"])
        for s in fam.get("steps") or []:
            if s["fix"] not in seen:
                pool.append(s); seen.add(s["fix"])
        for k in range(1, min(max_len, len(pool)) + 1):
            for combo in itertools.combinations(range(len(pool)), k):
                steps = [pool[i] for i in combo]
                if not coherent(steps):
                    continue
                out.append({"family": fam["family"], "steps": steps})
    return out


def alternatives(steps: list[dict]) -> list[tuple]:
    """Pairs of steps that are competing diagnoses of ONE symptom. -> [(fixA, fixB)]

    Two steps sharing a non-null `check` are branches of the same diagnostic question,
    not stages of one repair. `check_sim_status` answers either "Missing" — reseat the
    card — or "Locked" — escalate. A walk containing both demands two mutually exclusive
    repairs of a SIM that is in exactly one state, and grading them punishes an agent
    that correctly does only the one the world calls for.

    A NULL check is not a shared question and must not be grouped: `reboot_device`,
    `resume_line` and `enable_roaming` all have no check and legitimately co-occur.
    Grouping on null would discard most valid walks, which is why the rule is stated on
    the presence of the check rather than on equality alone.
    """
    out = []
    for i, a in enumerate(steps):
        for b in steps[i + 1:]:
            chk = a.get("check")
            if chk and chk == b.get("check") and a.get("fix") != b.get("fix"):
                out.append((a["fix"], b["fix"]))
    return out


def coherent(steps: list[dict]) -> bool:
    """Can one world require ALL of these repairs at once?

    `itertools.combinations` alone enumerated every subset of a family, so it produced
    walks demanding contradictory repairs. Measured on the 2026-08-19 telecom suite: one
    case required both `reseat_sim_card` and `transfer_to_human_agents` on a world whose
    `sim_status` was neither Missing nor Locked.

    Escalation is not filtered here. It is terminal rather than incompatible, and
    `build_case` reduces a walk containing it to the handoff alone, which is how tau3
    builds the same shape — its `airplane_mode_on|lock_sim_card_pin` task grades
    `transfer_to_human_agents` and nothing else even though airplane mode is fixable.
    """
    return not alternatives(steps)
