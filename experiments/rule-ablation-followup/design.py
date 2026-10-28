"""Pre-registered constants of the follow-up, shared by the selection, narration, run and scorer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
MAIN = PROJECT / "experiments" / "rule-ablation"
COVERAGE = PROJECT / "experiments" / "rule-coverage" / "coverage.py"

RULES = ("R3_payment", "R5_escalate_unfixable")
ARMS = ("control", *RULES)
PER_RULE = 10                 # most scenarios drawn per rule; fewer if the pool has fewer
SEED = 20261028
REPEATS = 5
CAP_USD = 45.0
EST_PER_CONV = 0.1224         # main run: $54.3454 over 444 conversations (amendment 2 estimate)
# amendment 3: per conversation by scenario kind, main run, arms control/R3/R5 on its
# payment scenarios (1, 13, 16) and escalation scenarios (3, 5, 12, 17, 23)
EST_BY_KIND = {"repair": 0.1417, "escalation": 0.0694}
WEBCHAT = "You are contacting support by web chat from another device."
NARRATOR_MODEL = "gpt-4.1"   # amendment 1: the default claude-opus-5 (meta_user.py:310) fails auth via both .env routes

SELECTION = HERE / "selection.json"
SUITE = HERE / "suite_followup.json"
CERTIFICATE = HERE / "certificate.json"
LEDGER = HERE / "spend-ledger.jsonl"
LEDGER_PREFIX = (7857, "ec8bae3695b243029f11637794d13185fc6678825503123d204b780920ecb38c")  # amendment 3: rows before the run
SUITE_SHA: str | None = "8885530081d4516dd5a311146a64dc3366c947479f7be1f9913445ddd0937e47"  # amendment 3 (14 scenarios)


R5_STRICT = ("C02", "C04", "C09")   # rule-coverage ids for policy lines 245, 247, 259 (the R5 span)


def tied(rule: str, crit: dict, case: dict) -> bool:
    """R3: the main run's rules.tied. R5: strict (rule-coverage, 4d756f2): every tool that can
    pass the criterion is one the R5 text directs, under a condition the scenario's starting
    state meets; diagnosed_before_handoff (passes on check_status_bar) is not tied."""
    rules, cov = _modules()
    if rule != "R5_escalate_unfixable":
        return rules.tied(rule, crit, case)
    mode = "must" if crit["rule"] == "tool_called_with" else "must_not"
    return any(m == mode and rules.tools(crit) <= tools and cond(case)
               for rid in R5_STRICT for m, tools, cond in cov.SPEC[rid])


def _modules():
    import importlib.util
    import sys
    if str(MAIN) not in sys.path:
        sys.path.insert(0, str(MAIN))
    import rules
    if "rule_coverage" not in sys.modules:
        spec = importlib.util.spec_from_file_location("rule_coverage", COVERAGE)
        sys.modules["rule_coverage"] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sys.modules["rule_coverage"])
    return rules, sys.modules["rule_coverage"]


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _lower(x):
    if isinstance(x, dict):
        return {k: _lower(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_lower(v) for v in x]
    return x.lower() if isinstance(x, str) else x


def functional_key(case: dict) -> str:
    """Same kind, same world and same graded criteria, ignoring letter case and the
    criteria's description text: two such cases are one test."""
    crits = [{k: v for k, v in c.items() if k != "description"}
             for c in case["success_criteria"]]
    body = {"kind": case.get("kind") or "repair", "criteria": crits,
            "world": (case.get("world_setup") or {}).get("entities")}
    return hashlib.sha256(json.dumps(_lower(body), sort_keys=True, default=str)
                          .encode()).hexdigest()
