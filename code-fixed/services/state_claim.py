"""Did the customer state a device reading they never took?

`action_claim` catches a claim to have ACTED — "I just turned it off" with no action.
This catches a claim about STATE — "Airplane mode is off" — which is a different
sentence and nothing looked at it.

MEASURED, 240 traces from 2026-08-20: 126 explicit device-state claims, of which 81
(64%) were made without ever calling a tool that reports the field. The cost is not
theoretical. On a case whose world has airplane mode ON:

    agent:    "...I'll guide you through some quick checks and fixes"
    customer: "Airplane mode is off, mobile data is on, and I don't have a VPN
               running. Data Saver is off too."   [no tool call]
    agent:    "Thank you for confirming those settings."

The agent believed it, moved on to APN settings and network mode, and
toggle_airplane_mode was never raised again in that conversation.

tau3 has no equivalent problem because its user simulator may either speak OR call a
tool in a turn, and what it knows about the device comes back as a tool RESULT. Ours
may do both in one reply, so the reply can assert a reading no tool produced.

THE FIELDS ARE THE ENVIRONMENT'S OWN. They are discovered by calling each read-only,
no-argument tool on the seed and recording the keys it returns, so nothing here is
telecom vocabulary.
"""
from __future__ import annotations

import inspect
import json
import re

# "airplane mode is off", "my sim status shows active", "data saver: on"
_ASSERTS = r"(?:\s*(?:is|are|shows?|says?|reads?|displays?|:)\s+)"
# A reading the customer would be repeating back. Deliberately narrow — a question
# ("is airplane mode the problem?") must not be read as a claim.
_VALUES = (r"(on|off|enabled|disabled|active|inactive|none|no|yes|good|poor|weak|"
           r"correct|incorrect|blank|empty|set|unset|connected|disconnected|"
           r"working|not working|\d+)")

_QUESTION = re.compile(r"\?\s*$")


def reporting_tools(mod) -> dict[str, set[str]]:
    """-> {field the customer could read: {tools that report it}}.

    Only no-argument read-only tools: those are the ones a person can just look at.
    """
    seed = getattr(mod, "SEED", None)
    if not isinstance(seed, dict):
        return {}
    out: dict[str, set[str]] = {}
    for name, fn in sorted((getattr(mod, "TOOLS", {}) or {}).items()):
        if (getattr(mod, "WRITES", {}) or {}).get(name):
            continue
        params = [p for p in inspect.signature(fn).parameters if p not in ("state", "_world")]
        if params:
            continue
        try:
            got = fn(json.loads(json.dumps(seed, default=str)))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(got, dict):
            for key in got:
                out.setdefault(key, set()).add(name)

        # A TOOL ALSO REPORTS WHAT IT READS. `check_status_bar` returns one formatted
        # string — {"status_bar": "Airplane Mode  No Signal ..."} — so by return keys
        # alone it reports nothing, and a customer who HAD checked the status bar would
        # still be told they never looked. That rejection would push the simulator to
        # check again, which is the sibling-ambiguity failure `action_claim` already
        # hit once. Reading the field counts as reporting it.
        try:
            src = inspect.getsource(fn)
        except Exception:  # noqa: BLE001
            continue
        for group in seed.values():
            rows = group if isinstance(group, list) else [group]
            for row in rows:
                if not isinstance(row, dict):
                    continue
                for key in row:
                    if f'"{key}"' in src or f"'{key}'" in src:
                        out.setdefault(key, set()).add(name)
    return out


# A FIELD NAME IS NOT HOW A PERSON SAYS IT. `data_saver_enabled` is spoken as "Data
# Saver is off", so the trailing state word has to come off or the phrase never matches
# — which is exactly why the first version caught `airplane_mode` and missed the "Data
# Saver is off too" in the same sentence.
_SUFFIXES = ("_enabled", "_connected", "_restricted", "_active", "_granted", "_on")


def _phrases(field: str) -> list[str]:
    out = [field.replace("_", " ").strip()]
    for suf in _SUFFIXES:
        if field.endswith(suf) and len(field) > len(suf) + 3:
            out.append(field[: -len(suf)].replace("_", " ").strip())
    return [p for p in dict.fromkeys(out) if len(p) >= 5]


def unobserved_claims(mod, message: str, observed_tools) -> list[str]:
    """-> fields the message states a reading for, that no called tool reported."""
    text = str(message or "")
    if not text.strip():
        return []
    seen = set(observed_tools or ())
    fields = reporting_tools(mod)
    out = []
    for field, tools in fields.items():
        if tools & seen:
            continue                       # they looked; they may report it
        phrases = _phrases(field)
        if not phrases:
            continue                       # too short to match without false positives
        done = False
        for sentence in re.split(r"(?<=[.!?,])\s+", text):
            low = sentence.lower()
            if _QUESTION.search(sentence.strip()):
                continue                   # asking is not claiming
            for phrase in phrases:
                if phrase not in low:
                    continue
                if re.search(re.escape(phrase) + _ASSERTS + _VALUES, low):
                    out.append(field)
                    done = True
                    break
            if done:
                break
    return sorted(set(out))


def correction_note(fields: list[str], mod) -> str:
    tools = reporting_tools(mod)
    how = {f: sorted(tools.get(f, ()))[:1] for f in fields}
    listed = ", ".join(f"`{f}`" for f in fields)
    which = ", ".join(sorted({t for v in how.values() for t in v}))
    return (
        "YOUR LAST REPLY WAS REJECTED. You stated a reading you have not taken: "
        f"{listed}. You have not looked at your phone for this, so you cannot know it, "
        "and the agent will believe you and stop investigating.\n"
        f"Reply again. Either put the check in `actions` (for example: {which}) and "
        "report what it returned, or say plainly that you do not know and ask the agent "
        "where to look. Never state a setting you have not checked."
    )
