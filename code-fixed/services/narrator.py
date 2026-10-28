"""Write the customer's side of a case whose mechanics are already proved.

WHERE THIS SITS, AND WHY THE BOUNDARY IS ABSOLUTE

`services/task_adapter.py` produces the mechanics: a broken world recomputed by executing
the faults, and graded actions already proved to repair it. This module writes only what
the customer says and knows.

It may not touch the world, the graded actions or the criteria. If it could, the replay
proof would stop describing the case that actually runs, and "every test is proved
solvable" would quietly become false. The proof covers the mechanics; this is presentation
laid over a case already established as solvable.

THREE GATES

A narrative that names the fix hands the agent the answer. "My APN settings are wrong,
please run reset_apn_settings" is not a diagnosis test. So the narration is REJECTED if it
mentions any graded tool, any internal predicate name, or any field name from the state.
That check is mechanical and runs on every case, because it is exactly the sort of leak
that reads fine to a human reviewer and destroys the measurement.

The customer describes SYMPTOMS. What they can perceive is what the read-only tools would
report, not what our predicates are called.

The second gate is that the customer's identity must be REAL. The narrator is given the
caller's actual record and may not invent a name, number, amount or date. On a live run
without this it produced "My name is Dana Whitfield" against a world holding Sofia
Martinez, and quoted a $70 bill against bills of 65, 65 and 5. An agent that looks those
up finds nothing, and telecom's policy makes identity verification the first step, so the
conversation could not happen at all.

The third gate is that the scenario must carry no TEMPERAMENT. Persona is a separate axis
applied at run time, and MM-tau-p2 measured it as worth up to 37 points of goal
achievement, so it has to be varied rather than baked in. A scenario saying "you are
anxious" is the same defect class as one naming the fix: both decide something the
experiment is supposed to vary.
"""
from __future__ import annotations

import json
import os
import re

import litellm

from services import llm_json
from services import narration_gate

MODEL_ENV = "NARRATOR_MODEL"


_PROMPT = """\
Write the customer's side of a support conversation for a test case.

The situation is already fixed and cannot be changed by you. Your only job is to write how
a real person would describe it and what they would know.

WHAT THIS PERSON'S PHONE AND ACCOUNT SHOW RIGHT NOW. This is the ground truth and your
story must match it. If a reading says there is NO signal, NO connection or NO service,
then the customer has NONE — do not write that things are slow, weak or patchy, because
that is a different problem and it sends the agent down the wrong path:
{observable}

WHAT IS ACTUALLY WRONG, in internal terms the customer would never use. These are our
labels, not readings — a label like "service is unhealthy" can mean anything from
degraded to completely dead, so take the DEGREE from the readings above, never from here:
{broken}

WHO YOU ARE. These are the real records for this caller. Use these exact details and
invent NOTHING about your identity, your account or your bills:
{caller}

THE POLICY the support agent follows, for the domain's vocabulary and the account details
a customer would know about themselves:
{policy}

Write the customer. Rules, and the first is checked mechanically:

- NEVER name a tool, a setting name, a field name, or any internal term from the list of
  what is wrong. The customer does not know the cause. They know the symptom. Write
  "my texts with pictures won't send", not "my MMSC URL is empty".
- Do not describe the fix or suggest a next step. Diagnosing is the agent's job and a
  customer who names the fix has destroyed the test.
- `reason_for_call` is one or two sentences in their own words, the way somebody actually
  opens a support call.
- `known_info` is what they can state about themselves without being prompted: their name,
  their number, where they are. Take these from WHO YOU ARE above, verbatim. Never invent
  a name, a phone number, an amount or a date. A caller whose details are not in the
  records cannot be found by the agent, and identity verification is the first thing the
  policy asks for, so an invented name makes the whole conversation impossible.
- `wont_volunteer` is true information they will give ONLY if the agent asks for it
  specifically. Put the diagnostic detail here, because making the agent ask is the point.
- `will_reveal_if_asked` is what they can check on request, in their words.
- `acceptance` is what they need to see before they believe it is fixed.
- Write NOTHING about temperament, patience, tone or technical confidence. Do not say the
  customer is anxious, brisk, chatty or confused. Behaviour is applied separately at run
  time so the same situation can be played by different callers, and a scenario that
  fixes the behaviour makes that impossible.

Return JSON only:
{{"reason_for_call": "...", "known_info": "...", "wont_volunteer": ["..."],
  "will_reveal_if_asked": ["..."], "acceptance": "..."}}
"""

# Words that describe HOW someone behaves rather than WHAT happened. A scenario carrying
# these has fixed the persona at authoring time, which is tau3's three-places warning
# being violated: behaviour then lives in the scenario AND in the persona slot, and the
# two can contradict each other. It also makes persona un-varied, since a case authored
# with a patient caller cannot be re-run as an anxious one.
BEHAVIOUR_WORDS = {
    "anxious", "impatient", "patient", "frustrated", "angry", "calm", "flustered",
    "worried", "nervous", "brisk", "chatty", "talkative", "terse", "polite", "rude",
    "confused", "tech-savvy", "nontechnical", "non-technical", "elderly", "young",
    "irritated", "annoyed", "friendly", "hostile", "confident", "hesitant",
}


def behaviour_leaks(text: str) -> list[str]:
    """Temperament words in a scenario that should carry only facts."""
    blob = re.sub(r"[^a-z0-9-]+", " ", (text or "").lower())
    return sorted({w for w in BEHAVIOUR_WORDS
                   if re.search(rf"\b{re.escape(w)}\b", blob)})


def leak_terms(case: dict, state_keys: set[str] | None = None) -> set[str]:
    """Words the customer must not say: the fix, our predicate names, the broken fields.

    NOT every field in the world. Forbidding all of them rejected 3 of 12 narrations on a
    live run for saying `customer_id`, `download_speed` and `network_type`, none of which
    reveal a cause. What must stay hidden is the FIX and the CAUSE:

      graded tools        naming one hands the agent the answer
      broken predicates   our internal vocabulary, which no customer has
      broken fields       the specific setting that is wrong, which is the diagnosis

    `state_keys` is accepted for callers that still pass it and is ignored, so a wider
    net cannot be reintroduced by accident.
    """
    # Via the shared accessor: this read `c["tool"]` and raised KeyError the moment
    # criteria moved into `param`. It is the leak gate, so a shape it cannot read
    # is not a crash to route around — an empty term set would let the narrator
    # name the fix in the customer's own words and nothing would object.
    from services.task_adapter import graded_tools
    terms = set(graded_tools(case))
    terms |= set(case.get("symptoms", {}).get("broken") or [])
    terms |= set(case.get("known_facts", {}).get("required_actions") or [])
    terms |= set(case.get("broken_fields") or [])
    return {t.lower() for t in terms if t}


def find_leaks(text: str, terms: set[str]) -> list[str]:
    """Which forbidden terms appear. Matched on word stems, not exact strings.

    `reset_apn_settings` leaks just as badly written as "reset APN settings", so the
    comparison is on the underscore-free form as well as the literal.
    """
    blob = re.sub(r"[^a-z0-9]+", " ", (text or "").lower())
    hits = []
    for t in terms:
        literal = t.lower()
        spaced = re.sub(r"[^a-z0-9]+", " ", literal).strip()
        if not spaced or len(spaced) < 4:
            continue
        if re.search(rf"\b{re.escape(spaced)}\b", blob):
            hits.append(t)
    return sorted(set(hits))


async def narrate(case: dict, policy: str, observable: dict, state_keys: set[str], *,
                  model: str | None = None, attempts: int = 2) -> dict | None:
    """The customer's side, or None. Never mutates `case`.

    Retries once with the leaked terms named, because a leak is usually one careless noun
    rather than a misunderstanding of the task.
    """
    model = model or os.getenv(MODEL_ENV, "claude-opus-5")
    terms = leak_terms(case, state_keys)
    prompt = _PROMPT.format(
        broken=json.dumps(case.get("symptoms", {}).get("broken") or [], indent=1),
        caller=json.dumps(case.get("caller") or {}, indent=1, default=str)[:1500],
        observable=json.dumps(observable, indent=1, default=str)[:3000],
        policy=(policy or "")[:24000])

    for attempt in range(attempts):
        try:
            resp = await litellm.acompletion(
                model=model, timeout=180, max_tokens=4000,
                messages=[{"role": "user", "content": prompt}])
            choice = resp.choices[0]
            raw = (choice.message.content or "").strip()
            if not raw:
                reason = getattr(choice, "finish_reason", "") or "empty"
                print(f"  [narrator] no content, finish_reason={reason}")
                return None
        except Exception as exc:  # noqa: BLE001
            print(f"  [narrator] request failed: {type(exc).__name__}")
            return None

        data = llm_json.loads_or_none(raw, want=dict, label="narrator")
        if not data or not data.get("reason_for_call"):
            return None

        spoken = " ".join(str(data.get(k) or "") for k in
                          ("reason_for_call", "known_info", "acceptance"))
        spoken += " " + " ".join(str(x) for x in (data.get("wont_volunteer") or []))
        spoken += " " + " ".join(str(x) for x in (data.get("will_reveal_if_asked") or []))
        leaks = find_leaks(spoken, terms)
        behaviour = behaviour_leaks(spoken)
        # DOES THE STORY MATCH THE DEVICE? Every airplane-mode case in the 2026-08-20
        # suite narrated "barely any bars" onto a phone whose status bar read
        # "Airplane Mode  No Signal". The narrator had the reading and wrote around it.
        wrong = narration_gate.contradictions(str(data.get("reason_for_call") or ""),
                                              observable)
        if not leaks and not behaviour and not wrong:
            return data
        if attempt + 1 < attempts:
            note = []
            if leaks:
                note.append(f"named these, which the customer cannot know: {leaks}. "
                            f"Describe the symptom instead of the cause")
            if behaviour:
                note.append(f"described temperament: {behaviour}. State only what "
                            f"happened and what is known, never how the person feels "
                            f"or behaves")
            if wrong:
                note.append(narration_gate.correction_note(wrong))
            prompt += "\n\nYour previous answer " + "; and ".join(note) + "."
            continue
        why = (f"leaks the answer: {leaks}" if leaks
               else f"fixes the persona: {behaviour}" if behaviour
               else f"contradicts the device: {wrong}")
        print(f"  [narrator] rejected, {why}")
        return None
    return None


def apply_narrative(case: dict, narrative: dict) -> dict:
    """Attach the customer's side. Mechanics are copied through untouched."""
    out = dict(case)
    out["goal"] = narrative["reason_for_call"]
    out["known_facts"] = dict(case.get("known_facts") or {},
                              known_info=narrative.get("known_info", ""),
                              acceptance=narrative.get("acceptance", ""))
    out["wont_volunteer"] = list(narrative.get("wont_volunteer") or [])
    out["will_reveal_if_asked"] = list(narrative.get("will_reveal_if_asked") or [])
    # persona_id is deliberately NOT set here. Behaviour is an overlay applied at run
    # time, so one case runs under every persona. Setting it at authoring time fixes the
    # caller and makes the persona dimension unmeasurable.
    return out
