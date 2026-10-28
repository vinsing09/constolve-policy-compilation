"""Is a generated scenario TRUE of the world it runs in?

THE QUESTION NOBODY ELSE CAN ANSWER

A scenario is a person saying things: "my picture messages will not send", "I am at home",
"I have had this phone about two years". Every one of those is a claim about the world. If
the world says the caller is abroad and the scenario says they are at home, the test is
incoherent, and the agent will be graded on a conversation that could not happen.

Competing tools cannot check this because their backend is mocked by the customer or hit
live, so there is no ground truth to compare a claim against. We compiled the world, so
there is. That makes scenario validity a MEASUREMENT here and an opinion everywhere else.

THE SPLIT, SAME AS EVERYWHERE ELSE IN THIS PIPELINE

    a model EXTRACTS claims from prose        it is good at this and nothing else will do
    the environment DECIDES each one          deterministic, and it is the ground truth

A claim is graded one of three ways, and the third matters as much as the first two:

    grounded      the world agrees
    contradicted  the world disagrees. The scenario is wrong and must not ship
    unverifiable  the world has no opinion. "I have had this phone two years" is not in
                  the state and is not a defect, it is colour. Counting it as a failure
                  would push the narrator toward writing nothing, and counting it as a
                  pass would hide real contradictions

Reporting all three separately is the whole point. A single "validity score" that folds
unverifiable into either side is the same defect as counting refuter errors as coverage.
"""
from __future__ import annotations

import json
import os

import litellm

from services import llm_json

MODEL_ENV = "SCENARIO_CHECK_MODEL"


_EXTRACT_PROMPT = """\
Extract the factual claims a customer is making, so each can be checked against the
system's records.

WHAT THE CUSTOMER SAID:
{spoken}

THE SYSTEM'S RECORDS, as `collection.field` with the values actually present. Use these
exact qualified names, and use the vocabulary shown rather than your own words:
{schema}

A claim is anything the customer asserts as true about themselves, their device or their
account. Break compound sentences into separate claims.

For each claim give:
- `text`: the words they used
- `field`: the QUALIFIED `collection.field` this is about, or null if the records hold
  nothing about it. A bare name is not enough: `amount` exists on line items while a
  bill's total is `bills.amount_due`, and picking the wrong one turns a true statement
  into a false one.
- `expected`: a SINGLE value the field would have if the claim is true, TAKEN FROM THE
  VALUES SHOWN for that field. Never a list. If the customer mentions several things,
  write one claim each. If the field's values are "excellent" and "poor", a customer
  saying "strong signal" means "excellent", not "strong". Use null if you cannot say.

Do not judge whether the claim is right. Only say what it asserts.

Return JSON only:
{{"claims": [{{"text": "...", "field": "<field or null>", "expected": <value or null>}}]}}
"""


def find_field(world, field: str):
    """Every value that field holds, with the collection it came from. -> [(coll, value)]

    The collection matters. A bare name can exist in two unrelated places, and resolving
    to the wrong one manufactures contradictions: on a live run "anything with a picture
    fails" was matched to `sent`, which belongs to a payment request rather than a
    message, and "my last bill for $70" was matched to a line item's `amount` of 65 while
    the bill's own `amount_due` was in fact 70. Both were reported as the scenario being
    false when the scenario was right and the lookup was wrong.

    Accepts a qualified `collection.field` and, when given a bare name, reports every
    collection it appears in so the caller can refuse to judge an ambiguous one.
    """
    want_coll, _, want_field = field.rpartition(".")
    found = []

    def walk(node, coll):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == want_field and not isinstance(v, (dict, list)):
                    if not want_coll or coll == want_coll:
                        found.append((coll, v))
                walk(v, k if isinstance(v, (dict, list)) else coll)
        elif isinstance(node, list):
            for v in node:
                walk(v, coll)
    walk(world, "")
    return found


# A customer saying "no service" and a world holding 0 are the same fact. Kept short and
# only for absence, because a broad synonym table would start excusing real mismatches.
_ZERO = {"0", "0.0", "none", "no", "zero", "nil", "no connection", "no service"}

_TRUE = {"true", "yes", "on", "1", "enabled", "active"}
_FALSE = {"false", "no", "off", "0", "disabled", "inactive"}


def as_bool(v):
    """A boolean, or None if this value does not denote one.

    `bool("False")` is True in Python, because a non-empty string is truthy. Comparing a
    real False against a model-written "False" that way reports a FALSE claim as
    grounded, which is the one direction of error this whole module exists to prevent.
    """
    if isinstance(v, bool):
        return v
    t = str(v).strip().lower()
    if t in _TRUE:
        return True
    if t in _FALSE:
        return False
    return None


def as_number(v):
    """A float, or None if this is not a number."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip().replace(",", "").lstrip("$"))
    except (TypeError, ValueError):
        return None


def same(a, b) -> bool:
    """Loose equality, because a model writes `true` where the world holds `"Active"`."""
    if a is None or b is None:
        return a is b
    if str(a).strip().lower() in _ZERO and str(b).strip().lower() in _ZERO:
        return True
    ba, bb = as_bool(a), as_bool(b)
    if ba is not None and bb is not None:
        return ba == bb
    if isinstance(a, bool) != isinstance(b, bool):
        # one side is a genuine boolean and the other is not boolean-like at all.
        # "Active" against True is not a match; say so rather than coercing.
        return False
    # NUMBERS COMPARE AS NUMBERS. The world stores 70.0 and a model writes 70, and
    # comparing "70.0" to "70" as strings marks a true claim contradicted. Every numeric
    # claim was exposed to that.
    na, nb = as_number(a), as_number(b)
    if na is not None and nb is not None:
        return abs(na - nb) < 1e-9
    return str(a).strip().lower() == str(b).strip().lower()


def verify(claims: list[dict], world: dict) -> dict:
    """Grade every claim against the compiled world. No model involved.

    An AMBIGUOUS claim is unverifiable, never contradicted. If a bare field name lives in
    two collections we cannot tell which record the customer meant, and guessing produces
    a false accusation against a scenario that may be perfectly true. Same rule as never
    counting a transport error as a coverage result: uncertainty is reported as
    uncertainty.
    """
    grounded, contradicted, unverifiable = [], [], []
    for c in claims:
        field = c.get("field")
        if not field or c.get("expected") is None:
            unverifiable.append(c)
            continue
        if isinstance(c.get("expected"), (list, dict, set)):
            # A claim is about ONE value. The extractor sometimes answers "which apps are
            # installed" with the whole list, and comparing a list against the world's
            # individual values reported six false contradictions in one run. Not
            # checkable as stated, so it is not judged.
            unverifiable.append({**c, "why": "claim is not about a single value"})
            continue
        hits = find_field(world, field)
        if not hits:
            unverifiable.append({**c, "why": "field not present in the world"})
            continue
        colls = {coll for coll, _ in hits}
        if any(same(v, c["expected"]) for _, v in hits):
            grounded.append(c)
        elif "." not in field and len(colls) > 1:
            unverifiable.append({**c, "why": f"ambiguous: {field} exists in "
                                             f"{sorted(colls)}",
                                 "actual": [v for _, v in hits][:3]})
        else:
            contradicted.append({**c, "actual": [v for _, v in hits][:3]})
    return {"grounded": grounded, "contradicted": contradicted,
            "unverifiable": unverifiable}


def field_vocabulary(world: dict, limit: int = 6) -> dict:
    """`collection.field` -> the values actually present. -> {qualified: [values]}

    Given bare field names the extractor produced `signal_strength='strong'` against a
    world whose vocabulary is "excellent", and mapped "$70 bill" to a line item's
    `amount`. Both were reported as the scenario being false. Showing the real value
    space removes the guesswork rather than trying to repair it afterwards.
    """
    out: dict[str, set] = {}

    def walk(node, coll):
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    walk(v, k)
                elif coll:
                    out.setdefault(f"{coll}.{k}", set()).add(v)
        elif isinstance(node, list):
            for v in node:
                walk(v, coll)
    walk(world, "")
    return {k: sorted(v, key=str)[:limit] for k, v in sorted(out.items())}


async def extract(spoken: str, schema: dict, *, world: dict | None = None,
                  model: str | None = None) -> list[dict]:
    """Claims from prose. The only model call in this module."""
    model = model or os.getenv(MODEL_ENV, "claude-opus-5")
    if world:
        fields = field_vocabulary(world)
    else:
        fields = sorted({f for spec in
                         list((schema.get("collections") or {}).values())
                         + list((schema.get("singletons") or {}).values())
                         for f in ((spec or {}).get("fields") or {})})
    try:
        resp = await litellm.acompletion(
            model=model, timeout=180, max_tokens=4000,
            messages=[{"role": "user", "content": _EXTRACT_PROMPT.format(
                spoken=(spoken or "")[:4000],
                schema=json.dumps(fields, indent=1, default=str)[:9000])}])
        choice = resp.choices[0]
        raw = (choice.message.content or "").strip()
        if not raw:
            print(f"  [scenario] no content, "
                  f"finish_reason={getattr(choice, 'finish_reason', '')}")
            return []
    except Exception as exc:  # noqa: BLE001
        print(f"  [scenario] request failed: {type(exc).__name__}")
        return []
    data = llm_json.loads_or_none(raw, want=dict, label="scenario")
    if not data:
        return []
    return [c for c in (data.get("claims") or []) if isinstance(c, dict) and c.get("text")]


def spoken_text(case: dict) -> str:
    """Everything the customer says, in one blob, for extraction."""
    parts = [str(case.get("goal") or "")]
    kf = case.get("known_facts") or {}
    parts += [str(kf.get("known_info") or ""), str(kf.get("acceptance") or "")]
    parts += [str(x) for x in (case.get("wont_volunteer") or [])]
    parts += [str(x) for x in (case.get("will_reveal_if_asked") or [])]
    return " ".join(p for p in parts if p)


def symptom_consistent(mod, case: dict) -> dict:
    """Does the trouble the customer reports match what is actually broken?

    A separate check from claim grounding, and a sharper one. The scenario exists because
    certain predicates are false; if the customer describes something else entirely, the
    case is coherent in its facts and wrong in its subject. No model is involved: the
    broken predicates are known and the world is known.
    """
    world = (case.get("world_setup") or {}).get("entities") or {}
    if not world:
        return {"checked": False}
    verdicts = mod.evaluate(json.loads(json.dumps(world, default=str)))
    declared = set(case.get("symptoms", {}).get("broken") or [])
    actually_false = {k for k, v in verdicts.items() if not v}
    return {
        "checked": True,
        "declared_broken": sorted(declared),
        "actually_false": sorted(actually_false),
        # every predicate the case claims is broken must really be false in its own world
        "consistent": declared.issubset(actually_false),
        "phantom": sorted(declared - actually_false),
    }
