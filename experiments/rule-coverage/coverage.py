"""Policy rule coverage: which atomic rules in the telecom policy at least one graded criterion checks.

Tie (the rule-ablation prereg's rule, made strict for any_of): criterion c in scenario s checks
rule r when r directs a call to c's tool under a condition s meets, and every tool that can pass
c (its any_of) is one r directs. tool_called_with ties to a "must" clause, tool_not_called to a
"must_not" clause. Order components ("before") are not tied separately. Rules only observable in
what the agent says or thinks are "not checkable": the grader reads tool calls only.
Run: PYTHONDONTWRITEBYTECODE=1 python3 -B coverage.py   (writes coverage.csv and coverage.md)
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
POLICY = PROJECT / "experiments" / "rule-ablation" / "policies" / "control.md"
sys.path.insert(0, str(PROJECT / "experiments" / "rule-ablation"))
import rules as ablation  # noqa: E402

NOW = "2025-02-25"
LINE = "LN-8830142"


def ent(case):
    return case["world_setup"]["entities"]


def phone(case):
    return ent(case)["phone"]


def line(case):
    return next(x for x in ent(case)["lines"] if x["line_id"] == LINE)


def broken(case, sym):
    return sym in case["symptoms"]["broken"]


def suspended(case, liftable):
    ln = line(case)
    return ln["status"].lower() == "suspended" and (ln["contract_end_date"] >= NOW) == liftable


def overdue(case):
    return any(b["status"].lower() == "overdue" for b in ent(case)["bills"])


def over_limit(case):
    ln = line(case)
    plan = next(p for p in ent(case)["plans"] if p["plan_id"] == ln["plan_id"])
    return ln["data_usage_gb"] > plan["data_limit_gb"]


def apn_bad(case):
    return str(phone(case)["apn_settings_status"]).lower() in ("incorrect", "not set")


def messaging_lacks_perm(case):
    granted = {p["permission"] for p in ent(case)["app_permissions"]
               if p["app_name"] == "messaging" and p["granted"]}
    return not {"storage", "sms"} <= granted


ANY = lambda c: True  # noqa: E731
ESCALATION = lambda c: c["kind"] == "escalation"  # noqa: E731
REPAIR = lambda c: c["kind"] == "repair"  # noqa: E731
LOOKUP = {"get_customer_by_phone", "get_customer_by_id", "get_customer_by_name"}

# rule id -> list of (mode, tools the rule directs, condition on the scenario), or a string:
# "speech: ..." = not checkable by a tool-call grader; any other string = no criterion type can tie it.
SPEC = {
    "G01": "speech: content of the agent's replies",
    "G02": "speech: content of the agent's replies",
    "G03": "no criterion type checks parallel tool calls",
    "G04": "speech: whether a reply accompanies a tool call",
    "G05": "no criterion checks a refusal of an out-of-policy request",
    "G06": [("must", {"transfer_to_human_agents"}, ESCALATION),
            ("must_not", {"transfer_to_human_agents"}, REPAIR)],
    "G07": "speech: the fixed hand-off message",
    "G08": [("must_not", {"transfer_to_human_agents"}, REPAIR)],
    "L01": [("must", {"get_customer_by_name"}, ANY)],
    "T01": [("must", LOOKUP, ANY)],
    "B01": [("must", {"get_bills_for_customer", "get_details_by_id"}, overdue)],
    "B02": [("must", {"get_bills_for_customer", "get_details_by_id"}, overdue)],
    "B03": [("must", {"send_payment_request"}, overdue)],
    "B04": "speech: telling the user a request was sent",
    "B05": [("must", {"make_payment"}, overdue)],
    "B06": "speech: timed against telling the user the bill is paid",
    "B07": [("must_not", {"send_payment_request"},
             lambda c: any(b["status"].lower() == "awaiting payment" for b in ent(c)["bills"]))],
    "S01": [("must", {"resume_line"}, lambda c: suspended(c, True))],
    "S02": [("must_not", {"resume_line"}, lambda c: suspended(c, False))],
    "S03": [("must", {"reboot_device"}, lambda c: suspended(c, True))],
    "D01": "no criterion compares gb_amount (refuel criteria compare customer_id, line_id only)",
    "D02": "speech: asking the amount",
    "D03": "speech: confirming the price",
    "D04": [("must", {"refuel_data"}, over_limit)],
    "P01": "no scenario asks for a plan change; the agent has no plan-change tool",
    "P02": [("must", {"get_available_plan_ids"}, ANY)],
    "P03": "speech: asking the user to choose",
    "P04": "speech: internal price calculation, said to the user",
    "P05": "speech: confirming the price",
    "P06": "the agent's tool set (tau2 telecom) has no plan-change tool",
    "R01": [("must", {"get_details_by_id"}, lambda c: not line(c)["roaming_enabled"])],
    "R02": [("must", {"enable_roaming"}, lambda c: not line(c)["roaming_enabled"])],
    "C01": [("must", {"toggle_airplane_mode"}, lambda c: phone(c)["airplane_mode"] is True)],
    "C02": [("must", {"check_sim_status"}, lambda c: broken(c, "has_cellular_service"))],
    "C03": [("must", {"reseat_sim_card"}, lambda c: phone(c)["sim_status"].lower() == "missing")],
    "C04": [("must", {"transfer_to_human_agents"},
             lambda c: phone(c)["sim_status"].lower() == "locked")],
    "C05": [("must", {"reset_apn_settings"}, apn_bad)],
    "C06": [("must", {"reboot_device"}, apn_bad)],
    "C07": [("must", {"get_details_by_id"}, lambda c: broken(c, "has_cellular_service"))],
    "C08": [("must", {"check_status_bar", "check_network_status"},
             lambda c: suspended(c, True))],
    "C09": [("must", {"transfer_to_human_agents"}, lambda c: suspended(c, False))],
    "M01": [("must", {"toggle_data"}, lambda c: phone(c)["mobile_data_enabled"] is False)],
    "M02": [("must", {"toggle_roaming"}, lambda c: phone(c)["roaming_enabled"] is False)],
    "M03": [("must", {"toggle_data_saver_mode"}, lambda c: phone(c)["data_saver_enabled"] is True)],
    "M04": [("must", {"disconnect_vpn"}, lambda c: bool(phone(c)["vpn_connected"])
             and str(phone(c)["vpn_performance_level"]).lower() == "poor")],
    "M05": [("must", {"set_network_mode_preference"},
             lambda c: phone(c)["network_mode_preference"][:2] in ("2g", "3g"))],
    "X01": [("must", {"check_status_bar", "check_network_status", "run_speed_test"},
             lambda c: broken(c, "mms_working"))],
    "X02": [("must", {"set_network_mode_preference"},
             lambda c: phone(c).get("network_type") == "2G")],
    "X03": [("must", {"reset_apn_settings"}, lambda c: not phone(c).get("mmsc_url"))],
    "X04": [("must", {"toggle_wifi_calling"}, lambda c: phone(c)["wifi_calling_enabled"] is True)],
    "X05": [("must", {"grant_app_permission"}, messaging_lacks_perm)],
}

# Why each rule that has a tie clause still ties to nothing (checked against the suite by the tests).
WHY = {
    "L01": "no criterion grades get_customer_by_name",
    "T01": "no criterion grades a customer lookup (get_customer_by_*)",
    "B01": "no criterion grades the bill lookup before send_payment_request",
    "B02": "no criterion grades the bill lookup",
    "B07": "no scenario starts with a bill in AWAITING PAYMENT",
    "S02": "scenario 23 meets the condition but has no tool_not_called(resume_line)",
    "P02": "no criterion grades get_available_plan_ids",
    "R01": "no criterion grades the line lookup alone (s*_get_data_usage also passes on get_data_usage)",
    "C03": "no scenario has a missing SIM",
    "C07": "no criterion grades the line lookup alone",
    "C08": "no criterion grades a service check after resume_line",
    "M05": "only network-mode criterion (scenario 2) starts at 4g_only, not 2G/3G",
    "X01": "diagnosed_before_handoff also passes on check_sim_status",
    "X02": "no scenario is on a 2G network",
    "X03": "every scenario has an MMSC URL set",
}


def load_rules() -> list[dict]:
    with open(HERE / "rules.csv", newline="") as f:
        return list(csv.DictReader(f))


def line_numbers(spec: str) -> list[list[int]]:
    """'16;171' -> [[16], [171]]; '110-111' -> [[110, 111]]: one group per ' | ' text piece."""
    groups = []
    for part in spec.split(";"):
        a, _, b = part.partition("-")
        groups.append(list(range(int(a), int(b or a) + 1)))
    return groups


def check_text(rule: dict, policy_lines: list[str]) -> None:
    """Each text piece must occur verbatim in its stated line(s) of the policy."""
    pieces, groups = rule["text"].split(" | "), line_numbers(rule["lines"])
    if len(groups) == 1 and len(pieces) == len(groups[0]):
        groups = [[n] for n in groups[0]]
    if len(pieces) != len(groups):
        raise ValueError(f"{rule['id']}: {len(pieces)} text pieces, {len(groups)} line groups")
    for piece, nums in zip(pieces, groups):
        if piece not in " ".join(policy_lines[n - 1].strip() for n in nums):
            raise ValueError(f"{rule['id']}: text not found at lines {nums}: {piece[:50]!r}")


def ties(rule_id: str, cases: list[dict]) -> list[tuple[int, str]]:
    spec = SPEC[rule_id]
    if isinstance(spec, str):
        return []
    out = []
    for i, case in enumerate(cases):
        for crit in case["success_criteria"]:
            mode = "must" if crit["rule"] == "tool_called_with" else "must_not"
            for m, tools, cond in spec:
                if m == mode and ablation.tools(crit) <= tools and cond(case):
                    out.append((i, crit["id"]))
                    break
    return out


def status(rule_id: str, tied: list) -> str:
    spec = SPEC[rule_id]
    if isinstance(spec, str) and spec.startswith("speech"):
        return "not_checkable"
    return "covered" if tied else "not_covered"


def compute() -> tuple[list[dict], list[dict]]:
    policy_lines = POLICY.read_text().splitlines()
    cases = ablation.suite()
    rules = load_rules()
    if [r["id"] for r in rules] != list(SPEC) or len({r["id"] for r in rules}) != len(rules):
        raise ValueError("rules.csv ids and SPEC keys differ or repeat")
    for r in rules:
        check_text(r, policy_lines)
        if r["kind"] not in ("must", "must_not", "order", "condition"):
            raise ValueError(f"{r['id']}: bad kind {r['kind']}")
        t = ties(r["id"], cases)
        r["status"], r["n_criteria"] = status(r["id"], t), len(t)
        r["scenarios"] = " ".join(str(s) for s in sorted({s for s, _ in t}))
        r["note"] = SPEC[r["id"]] if isinstance(SPEC[r["id"]], str) else WHY.get(r["id"], "")
        no_clause = isinstance(SPEC[r["id"]], str) and r["status"] != "not_checkable"
        if (r["status"] == "not_covered") != (r["id"] in WHY or no_clause):
            raise ValueError(f"{r['id']}: WHY note does not match status {r['status']}")
        r["_ties"] = t
    crits = [{"scenario": i, "id": c["id"], "tool": c["param"]["tool"], "rule": c["rule"],
              "rules": [r["id"] for r in rules if (i, c["id"]) in r["_ties"]]}
             for i, case in enumerate(cases) for c in case["success_criteria"]]
    return rules, crits


def report(rules: list[dict], crits: list[dict]) -> str:
    sections = list(dict.fromkeys(r["section"] for r in rules))
    tot = Counter(r["status"] for r in rules)
    checkable = tot["covered"] + tot["not_covered"]
    lines = ["# Policy rule coverage (telecom, 24-scenario suite)", "",
             "Generated by `coverage.py` from `rules.csv`, the control policy "
             f"(`{POLICY.relative_to(PROJECT)}`) and the first 24 cases of the suite. "
             "No model calls.", "",
             f"**{len(rules)} rules. Covered {tot['covered']}, not covered {tot['not_covered']}, "
             f"not checkable {tot['not_checkable']}. Of the {checkable} checkable rules, "
             f"{tot['covered']} ({100 * tot['covered'] / checkable:.0f}%) are checked by at least "
             f"one of the {len(crits)} criteria.**", "",
             "**Covered is not passed.** A covered rule has a criterion that would fail if the "
             "agent skipped the call; it says nothing about how often agents pass it.", "",
             "## Method", "",
             "- **Rule**: one directive telling the agent what to do, not do, or in which order "
             "or condition (`rules.csv`, text checked verbatim against the stated policy lines). "
             "Not counted: descriptions of the domain and of what tools return, \"X can be used "
             "to check\" hints (lines 232, 238, 285, 344), \"Refer to\" cross-references, and "
             "the data-limit options paragraph (313-319, whose actions are D04 and P06). A "
             "directive restated in a later section with the same condition and action is one "
             "rule with both line numbers (16;171, 106;118, 156;301, 253;360).",
             "- **Covered**: at least one criterion checks it. Tie rule (rule-ablation prereg, "
             "made strict): the rule directs a call to the criterion's tool under a condition "
             "the scenario's starting state meets, and every tool in the criterion's `any_of` is "
             "one the rule directs. `tool_called_with` ties to a must-call clause, "
             "`tool_not_called` to a must-not-call clause; \"before\" constraints are not tied "
             "separately.",
             "- **Not checkable**: only observable in what the agent says (or reasons). The "
             "grader reads tool calls only.",
             "- **Differences from the prereg table**: (1) line 324 is not covered: its only "
             "candidate criterion (scenario 2, set_network_mode_preference) starts at `4g_only`, "
             "which is neither 2G nor 3G; (2) the two `diagnosed_before_handoff` criteria "
             "(any_of check_sim_status, check_status_bar) tie to no rule, since a status-bar "
             "check is not a call the policy directs; the prereg's R5 counted them; (3) line 118 "
             "is merged with 106 (check the bill is overdue first), which no criterion grades.",
             "",
             "| section | rules | covered | not covered | not checkable |", "|---|---|---|---|---|"]
    for s in sections + ["Total"]:
        rs = rules if s == "Total" else [r for r in rules if r["section"] == s]
        c = Counter(r["status"] for r in rs)
        lines.append(f"| {s} | {len(rs)} | {c['covered']} | {c['not_covered']} | "
                     f"{c['not_checkable']} |")
    lines += ["", "## Rules no criterion checks", "", "| id | section | lines | rule | why |",
              "|---|---|---|---|---|"]
    for r in rules:
        if r["status"] == "not_covered":
            lines.append(f"| {r['id']} | {r['section']} | {r['lines']} | "
                         f"{r['text'].split(' | ')[0][:110]} | {r['note']} |")
    lines += ["", "## Not checkable by a tool-call grader", ""]
    lines += [f"- {r['id']} ({r['section']}, line {r['lines']}): {r['note'][8:]}"
              for r in rules if r["status"] == "not_checkable"]
    lines += ["", "## Covered rules and their criteria", ""]
    lines += [f"- {r['id']} ({r['section']}, line {r['lines']}): {r['n_criteria']} criteria, "
              f"scenarios {r['scenarios']}" for r in rules if r["status"] == "covered"]
    orphans = [c for c in crits if not c["rules"]]
    lines += ["", f"## Criteria tied to no rule ({len(orphans)} of {len(crits)})", ""]
    lines += [f"- scenario {c['scenario']}: {c['id']} ({c['tool']})" for c in orphans]
    return "\n".join(lines) + "\n"


def main() -> None:
    rules, crits = compute()
    with open(HERE / "coverage.csv", "w", newline="") as f:
        cols = ["id", "section", "lines", "kind", "status", "n_criteria", "scenarios", "note"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rules)
    (HERE / "coverage.md").write_text(report(rules, crits))
    print(json.dumps(Counter(r["status"] for r in rules)))


if __name__ == "__main__":
    main()
