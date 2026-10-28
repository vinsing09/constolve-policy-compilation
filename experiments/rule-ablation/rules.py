"""The six removed rules: the exact policy text each removes, and which graded criteria it ties to.

Tie rule (pre-registered): a criterion is tied to a rule when the removed text tells the
agent to make the criterion's tool call (its tool or one of its any_of tools) under a
condition the scenario meets. Ordering components ("before": [...]) are not tied
separately. See prereg.md for the per-criterion policy mapping.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SUITE = PROJECT / "code-original" / "generated" / "graph_suite_telecom.json"
N_CASES = 24
CONTROL_SHA = "065b48503bf780153fc671455cdfddb85b4c6ffb11e2ea93bd940233d804c3e9"
COPY_CHECKSUM = "684554a5be984056d2bb6b2222189de548cf50f205049c91f31f71f82a2cc02c"  # full-fixed-2
# full-fixed-2's tree (35d1fbb9…) with only the .env API-key line changed (commit 60e0cb8)
COPY_TREE = "e83f11aaf7d0bddf4c48443de9ab55851f95d348df1086ac2015f3fbc9263db9"

APN_CHECK = ('If `check_apn_settings()` shows "Incorrect", guide the user to use '
             '`reset_apn_settings()` to reset the APN settings.\n')
MMSC_CHECK = ('If `check_apn_settings()` shows MMSC URL is not set, guide the user to use '
              '`reset_apn_settings()` to reset the APN settings.\n')
APN_REBOOT = ('After resetting the APN settings, the user must be instructed to use '
              '`reboot_device()` for the changes to apply.\n')

RULES = {
    "R1_airplane": {
        "aspect": "cellular service: airplane mode (### Airplane Mode)",
        "spans": ["If it is ON, guide the user to use `toggle_airplane_mode()` to turn it OFF.\n"],
    },
    "R2_apn": {
        "aspect": "network settings: APN reset then reboot (### Incorrect APN Settings, "
                  "### Verifying APN (MMSC URL) for MMS)",
        "spans": [APN_CHECK + APN_REBOOT, MMSC_CHECK + APN_REBOOT],
    },
    "R3_payment": {
        "aspect": "billing: overdue-bill payment request and payment (## Overdue Bill Payment)",
        "spans": [
            "- Send the user a payment request for the overdue bill.\n"
            "    - This will change the status of the bill to AWAITING PAYMENT.\n"
            "- Inform the user that a payment request has been sent. They should:\n"
            "    - Check their payment requests using the check_payment_request tool.\n"
            "- If the user accepts the payment request, use the make_payment tool to make "
            "the payment.\n",
            "- The send payement request tool will not check if the bill is overdue. You "
            "should always check that the bill is overdue before sending a payment request.\n"],
    },
    "R4_suspension": {
        "aspect": "line suspension: permission to lift it (## Line Suspension)",
        "spans": ["You are allowed to lift the suspension after the user has paid all their "
                  "overdue bills.\n"],
    },
    "R5_escalate_unfixable": {
        "aspect": "escalation, specific: check SIM; escalate a locked SIM or a suspension the "
                  "agent cannot lift (### SIM Card Issues, ### Line Suspension)",
        "spans": [
            "Use `check_sim_status()` to check the status of the SIM card.\n"
            "If it shows \"Missing\", guide the user to use `reseat_sim_card()` to ensure the "
            "SIM card is correctly inserted.\n"
            "If it shows \"Locked\" (due to incorrect PIN or PUK entries), **escalate to "
            "technical support for assistance with SIM security**.\n"
            "If it shows \"Active\", the SIM itself is likely okay.\n",
            "*   If the suspension cannot be lifted by the agent (e.g., due to contract end "
            "date as mentioned in general policy, or other reasons not resolvable by the "
            "agent), **escalate to technical support**.\n"],
    },
    "R6_transfer_scope": {
        "aspect": "escalation, general: transfer only when out of scope, after trying "
                  "everything (main policy preamble, tech-support introduction)",
        "spans": [
            "You should transfer the user to a human agent if and only if the request cannot "
            "be handled within the scope of your actions. ",
            "You should try your best to resolve the issue for the user before transferring "
            "the user to a human agent.\n\n",
            "Make sure you try all the possible ways to resolve the user's issue before "
            "transferring to a human agent.\n\n"],
    },
}
PAIR = ("R5_escalate_unfixable", "R6_transfer_scope")
ARMS = ("control", *RULES)


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def remove(policy: str, rule: str) -> str:
    """The policy with the rule's spans cut out; every span must occur exactly once."""
    out = policy
    for span in RULES[rule]["spans"]:
        if policy.count(span) != 1:
            raise ValueError(f"{rule}: span occurs {policy.count(span)} times: {span[:60]!r}")
        out = out.replace(span, "", 1)
    return out


def tools(crit: dict) -> set[str]:
    p = crit["param"]
    return {p["tool"], *(p.get("any_of") or [])}


def tied(rule: str, crit: dict, case: dict) -> bool:
    t, must = tools(crit), crit["rule"] == "tool_called_with"
    case_tools = {x for c in case["success_criteria"] for x in tools(c)}
    escalation = case.get("kind") == "escalation"
    return {
        "R1_airplane": must and "toggle_airplane_mode" in t,
        "R2_apn": must and ("reset_apn_settings" in t or
                            ("reboot_device" in t and "reset_apn_settings" in case_tools)),
        "R3_payment": must and bool(t & {"send_payment_request", "make_payment"}),
        "R4_suspension": must and "resume_line" in t,
        "R5_escalate_unfixable": must and escalation and
        ("check_sim_status" in t or "transfer_to_human_agents" in t),
        "R6_transfer_scope": "transfer_to_human_agents" in t,
    }[rule]


def suite() -> list[dict]:
    return json.loads(SUITE.read_text())[:N_CASES]


def ties() -> dict[str, list[list]]:
    """rule -> [[scenario index, criterion id], ...] over the first 24 scenarios."""
    cases = suite()
    return {r: [[i, c["id"]] for i, case in enumerate(cases)
                for c in case["success_criteria"] if tied(r, c, case)] for r in RULES}
