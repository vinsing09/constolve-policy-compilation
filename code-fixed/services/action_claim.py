"""Did the customer SAY they did something they did not DO?

THE DEFECT

Read from a transcript, 20 August. The agent correctly identifies airplane mode. The
customer replies:

    "I just turned off Airplane Mode. Now the status bar shows one bar, and it says LTE"

and emits NO action. The world never changed. The phone stayed broken, so the agent spent
the next eleven turns troubleshooting a fault it had already diagnosed and instructed
correctly. The graded action `toggle_airplane_mode` was missed in 34 of 35 runs.

The pattern across the suite is exact, and it is about the KIND of tool:

    checks  (check_status_bar, run_speed_test, check_sim_status)   performed, ~0% missed
    changes (toggle_airplane_mode, reseat_sim_card, reboot_device) narrated,  35-97% missed

The simulator reads its own rule — "never claim in `message` that you did something you
did not put in `actions`" — and does not follow it, because nothing checks.

WHY THIS IS GENERAL, NOT A TELECOM PATCH

Nothing here knows what a SIM card is. A tool that WRITES to the world is a change; one
that writes nothing is a check. That split comes from the compiled environment's own
`WRITES` table, so it holds for any domain we compile. The words searched for are derived
from the tool's own name, and the completion markers are ordinary English tense, not
domain vocabulary.

tau3 avoids the whole problem structurally: "At each turn you can either send a message
to the agent OR make a tool call. You cannot do both." A turn that acts carries no prose
to lie in. We allow both in one reply, so we check instead.
"""
from __future__ import annotations

import re

# "I just turned it off", "done", "I've restarted it". Ordinary completion language —
# no domain words, so this transfers to any compiled domain.
_CLAIMED = re.compile(
    r"\b(i (just |already |'ve |have )?(did|done|turned|switched|toggled|reset|restarted|"
    r"rebooted|removed|reseated|re-?inserted|enabled|disabled|changed|set|updated|paid|"
    r"granted|installed|connected|disconnected)"
    r"|(just |now )?(did|done|turned|switched|toggled|reset|restarted|rebooted|changed|set)"
    r"\b.{0,40}\b(it|that|this|off|on|back|again)"
    r"|it('s| is| has been) (now |already )?(done|off|on|reset|restarted|changed|enabled|disabled))",
    re.I)


def changing_tools(mod) -> dict:
    """User tools that CHANGE the world -> the words that name them.

    A tool with an empty write-set is a check: the customer may report what it returned.
    A tool that writes is a change: the customer may only report it if it actually ran.
    """
    writes = getattr(mod, "WRITES", {}) or {}
    out = {}
    for name in (getattr(mod, "USER_TOOLS", None) or []):
        if not writes.get(name):
            continue
        words = [w for w in name.split("_") if len(w) > 2]
        # `toggle_airplane_mode` -> "airplane mode"; `reseat_sim_card` -> "sim card"
        subject = " ".join(words[1:]) if len(words) > 1 else " ".join(words)
        out[name] = [w for w in {subject, " ".join(words), name.replace("_", " ")} if w]
    return out


# "I haven't turned it off yet", "I could not reset it", "I'm not able to reboot".
# A REFUSAL IS THE HONEST ANSWER we are asking for — flagging it would punish the exact
# behaviour the correction demands and push the simulator back toward claiming.
_NEGATED = re.compile(
    r"\b(haven'?t|have not|hasn'?t|has not|didn'?t|did not|don'?t|do not|can'?t|cannot|"
    r"could not|couldn'?t|won'?t|will not|not (yet|able|sure)|unable|no idea|"
    r"not managed|failed to)\b", re.I)


def unsupported_claims(mod, message: str, actions: list) -> list[str]:
    """Tools the message claims were performed, that no action performed. -> [tool, ...]

    Both halves must hold: the message names the thing AND asserts it was done. Naming it
    alone is fine — "is airplane mode the problem?" is a question, not a claim.
    """
    text = str(message or "")
    if not text.strip() or not _CLAIMED.search(text):
        return []
    performed = {(a.get("tool") if isinstance(a, dict) else getattr(a, "tool", None))
                 for a in (actions or [])}
    low = text.lower()
    tools = changing_tools(mod)
    # TWO TOOLS CAN SHARE A SUBJECT. `reset_apn_settings` and `set_apn_settings` both
    # reduce to "apn settings", so a customer who correctly performed the reset was
    # flagged for not performing the set — 54 of 85 flags in the 2026-08-20 run, all
    # false. Worse, the rejection made the simulator repeat the actions it had already
    # taken, which is visible in the transcript as reset+reboot three times over.
    #
    # If any tool that was PERFORMED shares the phrase, the sentence is accounted for.
    performed_phrases = {p.lower() for t in performed for p in tools.get(t, [])}
    out = []
    for tool, phrases in tools.items():
        if tool in performed:
            continue
        for p in phrases:
            if not p or p.lower() not in low:
                continue
            if p.lower() in performed_phrases:
                break            # a tool the customer DID perform owns this phrase
            # judged on the SENTENCE the subject appears in, so "I checked the status
            # bar. I have not turned off airplane mode." is not read as a claim because
            # of the first sentence.
            i = low.index(p.lower())
            start = max(0, low.rfind(".", 0, i) + 1)
            end = low.find(".", i)
            sentence = low[start: end if end > 0 else len(low)]
            if _NEGATED.search(sentence):
                break
            if _CLAIMED.search(sentence):
                out.append(tool)
                break
    return out


def correction_note(tools: list[str]) -> str:
    """What to tell the simulator when it claimed without acting."""
    listed = ", ".join(f"`{t}`" for t in tools)
    return (
        "YOUR LAST REPLY WAS REJECTED. You said you had done something and did not put it "
        f"in `actions`: {listed}. On your phone nothing changed, so the agent is now "
        "troubleshooting a fault you told them was fixed.\n"
        "Reply again. Either put the action in `actions` and describe what you observed "
        "after it, or say plainly that you have not done it yet. Do not claim an action "
        "you are not performing."
    )
