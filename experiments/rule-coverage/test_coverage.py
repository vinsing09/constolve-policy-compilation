"""Run: PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q -p no:cacheprovider test_coverage.py"""
import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import coverage as cov  # noqa: E402

ABLATION_TO_RULES = {"R1_airplane": {"C01"}, "R2_apn": {"C05", "C06"}, "R3_payment": {"B03", "B05"},
                     "R4_suspension": {"S01"}, "R5_escalate_unfixable": {"C02", "C04", "C09"},
                     "R6_transfer_scope": {"G06", "G08"}}


@pytest.fixture(scope="module")
def result():
    return cov.compute()


def test_totals_partition(result):
    rules, crits = result
    assert len(crits) == 93
    counts = {s: sum(r["status"] == s for r in rules) for s in ("covered", "not_covered", "not_checkable")}
    assert sum(counts.values()) == len(rules) == 51
    assert counts == {"covered": 20, "not_covered": 20, "not_checkable": 11}


def test_ties_match_the_ablation_mapping(result):
    rules, _ = result
    frozen = json.loads((cov.PROJECT / "experiments/rule-ablation/ties.json").read_text())
    by_id = {r["id"]: set(r["_ties"]) for r in rules}
    for ab, ids in ABLATION_TO_RULES.items():
        mine = set().union(*(by_id[i] for i in ids))
        theirs = {tuple(x) for x in frozen[ab]}
        if ab == "R5_escalate_unfixable":
            assert theirs - mine == {(5, "diagnosed_before_handoff"), (23, "diagnosed_before_handoff")}
            assert mine <= theirs
        else:
            assert mine == theirs, ab


def test_wrong_line_number_is_refused(result):
    rules, _ = result
    lines = cov.POLICY.read_text().splitlines()
    bad = dict(next(r for r in rules if r["id"] == "C01"), lines="238")
    with pytest.raises(ValueError):
        cov.check_text(bad, lines)
    cov.check_text(next(r for r in rules if r["id"] == "B04"), lines)


def test_any_of_criterion_with_an_undirected_tool_does_not_tie():
    cases = cov.ablation.suite()
    assert (3, "s1_get_data_usage") not in cov.ties("R01", cases)
    assert (3, "s1_get_data_usage") not in cov.ties("C07", cases)
    assert not any(c == "diagnosed_before_handoff" for _, c in cov.ties("C02", cases))


def test_condition_drives_coverage_on_real_case_shape():
    cases = cov.ablation.suite()
    assert cov.ties("M05", cases) == []
    mutated = copy.deepcopy(cases)
    mutated[2]["world_setup"]["entities"]["phone"]["network_mode_preference"] = "3g_only"
    assert cov.ties("M05", mutated) == [(2, "s3_set_network_mode_preference")]
    mutated[23]["success_criteria"].append({"id": "no_resume", "rule": "tool_not_called",
                                            "param": {"tool": "resume_line"}})
    assert cov.ties("S02", mutated) == [(23, "no_resume")]


def test_speech_rules_never_count_as_covered(result):
    rules, _ = result
    speech = {k for k, v in cov.SPEC.items() if isinstance(v, str) and v.startswith("speech")}
    assert {r["id"] for r in rules if r["status"] == "not_checkable"} == speech
