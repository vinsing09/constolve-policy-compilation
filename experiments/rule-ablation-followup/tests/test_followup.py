"""Follow-up tests on the real suite, the real selection and a fake-model run on a code-fixed copy.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -p no:cacheprovider -q \
        experiments/rule-ablation-followup/tests
"""
import copy as cp
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import design as D  # noqa: E402
import run_followup as rf  # noqa: E402
import run_manifest as rm  # noqa: E402
import score_followup as sf  # noqa: E402
import reduce_r5  # noqa: E402
import select_scenarios as ss  # noqa: E402
import rules  # noqa: E402
from runlog import tree_hash  # noqa: E402

SUITE24 = rules.suite()
SEL = json.loads(D.SELECTION.read_text())
FU = json.loads(D.SUITE.read_text())
FU16 = json.loads((HERE / "suite_followup_16.json").read_text())
INDEX = json.loads((HERE / "pool_index.json").read_text())
R3, R5 = D.RULES


# ---------- duplicates and selection ----------

def test_functional_key_ignores_letter_case_and_description_only():
    c = SUITE24[13]
    same = cp.deepcopy(c)
    same["world_setup"]["entities"]["bills"][0]["status"] = "overdue"
    same["success_criteria"][0]["description"] = "anything"
    assert D.functional_key(same) == D.functional_key(c)
    other = cp.deepcopy(c)
    other["world_setup"]["entities"]["lines"][0]["roaming_enabled"] = False
    assert D.functional_key(other) != D.functional_key(c)
    other = cp.deepcopy(c)
    other["success_criteria"][0]["param"]["before"] = []
    assert D.functional_key(other) != D.functional_key(c)


def test_suite_has_exactly_five_identical_pairs():
    keys = [D.functional_key(c) for c in SUITE24]
    pairs = {(i, j) for i in range(24) for j in range(i + 1, 24) if keys[i] == keys[j]}
    assert pairs == {(0, 15), (3, 17), (6, 9), (7, 14), (8, 11)}
    exact = [ss.exact_key(c) for c in SUITE24]
    assert all(exact[i] == exact[j] for i, j in pairs)
    assert INDEX["stats"]["suite_functional_distinct"] == 19


def test_selection_recomputes_from_the_pool_index():
    picked, record = ss.select(INDEX["pool"], SUITE24)
    assert {r: [p["walk_index"] for p in picked[r]] for r in D.RULES} == \
        {r: SEL["record"][r]["walk_indices"] for r in D.RULES}
    assert record == SEL["record"]
    assert record[R3]["candidates"] == record[R3]["picked"] == 6
    assert record[R5]["picked"] == 10


def test_selected_are_new_distinct_tagged_and_unchanged_by_narration():
    suite_keys = {D.functional_key(c) for c in SUITE24}
    keys = [D.functional_key(c) for c in FU16]
    assert len(set(keys)) == 16 and not set(keys) & suite_keys
    for x, c in zip(SEL["cases"], FU16):
        assert [k["id"] for k in c["success_criteria"] if D.tied(x["rule"], k, c)] == x["tied"] != []
        for k in ("world_setup", "success_criteria", "reference_solution", "kind", "symptoms"):
            assert c[k] == x["case"][k]
    assert [x["rule"] for x in SEL["cases"]] == [R3] * 6 + [R5] * 10


def test_reduction_keeps_r3_and_a_seeded_8_of_r5_in_order():
    assert reduce_r5.reduce(FU16, SEL) == FU
    walk = {x["functional_key"]: (x["rule"], x["walk_index"]) for x in SEL["cases"]}
    got = [walk[D.functional_key(c)] for c in FU]
    assert [r for r, _ in got] == [R3] * 6 + [R5] * 8
    assert [w for _, w in got] == [2924, 2928, 2929, 2936, 2937, 2946, 97, 372, 702, 1245,
                                   1255, 1320, 1396, 2680]


def test_strict_r5_never_ties_a_criterion_passable_by_a_status_bar_check():
    cov = D._modules()[1]
    for cases in (SUITE24, FU16, FU):
        mine = {(i, k["id"]) for i, c in enumerate(cases) for k in c["success_criteria"]
                if D.tied(R5, k, c)}
        assert mine == {t for rid in D.R5_STRICT for t in cov.ties(rid, cases)}
        assert not any(cid == "diagnosed_before_handoff" for _, cid in mine)
    assert len(sf.ties()[R5]) == 8 and len(sf.ties()[R3]) == 12


def test_every_brief_ends_with_the_web_chat_line_once():
    for c in FU16:
        info = c["known_facts"]["known_info"]
        assert info.endswith(" " + D.WEBCHAT) and info.count(D.WEBCHAT) == 1
        assert c["scenario"]["instructions"]["known_info"] == info and c["goal"].strip()


def test_suite_and_certificate_are_the_registered_ones():
    assert D.sha_file(D.SUITE) == D.SUITE_SHA
    cert = json.loads(D.CERTIFICATE.read_text())
    assert cert["selection_sha256"] == D.sha_file(D.SELECTION) and cert["dropped"] == []
    assert [r["walk_index"] for r in cert["cases"]] == [x["walk_index"] for x in SEL["cases"]]
    assert len(rf.suite_cases(None)) == 14
    assert D.sha_file(reduce_r5.FULL) == reduce_r5.FULL_SHA


# ---------- scoring ----------

def test_pooled_counts_main_duplicates_once_on_real_main_rows():
    _, main_rows, _ = sf.sa.load(sf.MAIN_RUN)
    out = sf.pooled(main_rows, [], {r: set() for r in D.RULES})
    assert out[R5]["tied"]["n_scenarios"] == 4          # 3, 5, 12, 17, 23 with 3 == 17
    assert out[R3]["tied"]["n_scenarios"] == 3


def test_identical_arms_drop_nothing():
    rows = [{"arm": a, "rep": 0, "scen": i, "cid": k["id"], "ok": (i + n) % 2 == 0}
            for a in D.ARMS for i, c in enumerate(FU) for n, k in enumerate(c["success_criteria"])]
    for r, v in sf.score(rows, sf.ties()).items():
        assert v["tied"]["drop"] == 0 and not v["tied"]["caught"] and v["tied"]["p"] == 1.0


# ---------- driver, on a copy of code-fixed with fake model calls ----------

@pytest.fixture(scope="module")
def copy(tmp_path_factory):
    dst = tmp_path_factory.mktemp("c") / "code-fixed"
    shutil.copytree(D.PROJECT / "code-fixed", dst, symlinks=True)
    yield dst
    shutil.rmtree(dst)          # the copy holds .env


def run(copy, results, name, arms="control,R5_escalate_unfixable", cases="7", test=True, **env):
    clean = {k: v for k, v in os.environ.items()
             if not (k in rm.EXACT or k.startswith(rm.PREFIXES)) and k != "RULE_ABLATION_TEST"}
    script = HERE / "tests" / "fake_followup_run.py" if test else HERE / "run_followup.py"
    p = subprocess.run(
        [sys.executable, "-B", str(script), "--run", name, "--cases", cases, "--repeats", "1",
         "--arms", arms, "--copy", str(copy), "--results", str(results)],
        env={**clean, "PYTHONDONTWRITEBYTECODE": "1",
             "FAKE_LEDGER": str(results / "ledger.jsonl"), **env},
        capture_output=True, text=True, timeout=600)
    m = results / name / "manifest.json"
    return p, (json.loads(m.read_text()) if m.exists() else None)


@pytest.fixture(scope="module")
def good(copy, tmp_path_factory):
    results = tmp_path_factory.mktemp("results")
    real_ledger = D.sha_file(D.LEDGER)
    before = tree_hash(copy)
    p, m = run(copy, results, "g")
    return results, p, m, before, real_ledger


def test_good_run_is_clean_and_each_arm_gets_its_policy(good, copy):
    results, p, m, before, real_ledger = good
    assert p.returncode == 0, p.stderr[-3000:]
    s = m["summary"]
    assert s["problems"] == [] and s["conversations"] == {"clean": 14}
    assert m["case_ids"][6].startswith("6:") and m["suite_sha256"] == D.SUITE_SHA
    meta = json.loads((D.MAIN / "variants.json").read_text())
    assert {a: v["policy_sha256"] for a, v in m["arms"].items()} == \
        {a: meta[a]["sha256"] for a in ("control", R5)}
    assert len(list((results / "g" / "traces").glob("*.json"))) == 14
    assert tree_hash(copy) == before and not (copy / "logs" / "bench").exists()
    assert rm.seal_path(results / "g").read_text() == rm.seal(results / "g")
    ledger = [json.loads(x) for x in (results / "ledger.jsonl").read_text().splitlines()]
    assert ledger[0]["kind"] == lg.FREEZE and sum(x["kind"] == lg.FREEZE for x in ledger) == 1
    assert ledger and abs(sum(x["usd"] for x in ledger) - m["guard_spent_usd"]) < 1e-3
    assert D.sha_file(D.LEDGER) == real_ledger                  # the real ledger untouched


def test_traces_carry_the_follow_up_world_and_brief(good):
    d = json.loads(next((good[0] / "g" / "traces").glob("ours_control_0_6_*.json")).read_text())
    assert [k["id"] for k in d["criteria"]] == [k["id"] for k in FU[6]["success_criteria"]]


def test_good_run_scores_without_registration(good):
    m, rows, n = sf.load(good[0] / "g", is_registered=False)
    assert n == 7 and {r["arm"] for r in rows} == {"control", R5}
    with pytest.raises(sf.Refused):
        sf.check_registered(good[0] / "g", m)


def test_wrong_suite_hash_is_refused(copy, tmp_path):
    p, m = run(copy, tmp_path, "w", FAKE_SUITE_SHA="0" * 64)
    assert p.returncode != 0 and "not the registered file" in p.stderr and m is None


def test_estimate_over_cap_is_refused_before_any_call(copy, tmp_path):
    p, m = run(copy, tmp_path, "c", FAKE_CAP_USD="1.0")
    assert p.returncode == 2 and "exceeds" in p.stderr and m is None
    assert not (tmp_path / "ledger.jsonl").exists()


def test_paid_mode_cannot_choose_copy_results_or_design(copy, tmp_path):
    # tripwire: an invalid key in the shell (preflight refuses OPENAI_*; calls would fail auth)
    p, _ = run(copy, tmp_path, "x", test=False, OPENAI_API_KEY="sk-tripwire-invalid")
    assert p.returncode == 2 and "unrecognized arguments" in p.stderr
    assert not (HERE / "results" / "x").exists()


# ---------- audit r1: ledger integrity and complete runs ----------

import ledger as lg  # noqa: E402

REAL = D.LEDGER.read_text().splitlines(keepends=True)


def ledger_with(tmp_path, lines):
    p = tmp_path / "l.jsonl"
    p.write_text("".join(lines))
    return p


def row(kind, usd, conv="c"):
    return json.dumps({"run": "main", "conv": conv, "usd": usd, "kind": kind}) + "\n"


def test_registered_ledger_prefix_sums_to_the_spend_so_far():
    assert abs(lg.prior_spend(D.LEDGER, D.LEDGER_PREFIX) - 19.650498) < 1e-6
    assert D.LEDGER_PREFIX[0] <= len(REAL)


@pytest.mark.parametrize("bad", [
    "truncated", "negative_row", "settle_without_reserve", "settle_beyond_reserve",
    "probe_after_prefix", "edited_prefix", "settle_a_prefix_reservation",
    "settle_across_a_freeze", "freeze_with_money"])
def test_tampered_ledger_is_refused(tmp_path, bad):
    lines = list(REAL[:D.LEDGER_PREFIX[0]])
    if bad == "truncated":
        lines = lines[:-1]
    elif bad == "negative_row":
        lines.append(row("reserve_worst_case", -5.0))
    elif bad == "settle_without_reserve":
        lines.append(row("settle_call", -1.0))
    elif bad == "settle_beyond_reserve":
        lines += [row("reserve_worst_case", 0.3), row("settle_call", -0.5)]
    elif bad == "settle_a_prefix_reservation":
        last = json.loads(lines[-1])                    # an open reservation of the killed run
        assert last["kind"] == "reserve_worst_case"
        lines.append(json.dumps({"run": last["run"], "conv": last["conv"], "usd": -last["usd"],
                                 "kind": "settle_call"}) + "\n")
    elif bad == "settle_across_a_freeze":              # a run crashed; a later one starts
        lines += [row(lg.FREEZE, 0.0, None), row("reserve_worst_case", 0.3, "a"),
                  row(lg.FREEZE, 0.0, None), row("settle_call", -0.3, "a")]
    elif bad == "freeze_with_money":
        lines.append(row(lg.FREEZE, -1.0, None))
    elif bad == "probe_after_prefix":
        lines.append(row("manual_probe_worst_case", -0.1))
    else:
        lines[0] = lines[0].replace('"usd": ', '"usd": -')
    with pytest.raises(lg.BudgetStop):
        lg.prior_spend(ledger_with(tmp_path, lines), D.LEDGER_PREFIX)


def test_honest_appends_are_accepted(tmp_path):
    lines = list(REAL[:D.LEDGER_PREFIX[0]]) + [row(lg.FREEZE, 0.0, None),
        row("reserve_worst_case", 0.36, "a"), row("reserve_worst_case", 0.36, "b"),
        row("settle_call", -0.33, "b"), row("settle_call", -0.35, "a"),
        row("reserve_worst_case", 0.36, "a")]                     # a crash: still charged
    got = lg.prior_spend(ledger_with(tmp_path, lines), D.LEDGER_PREFIX)
    assert abs(got - (19.650498 + 0.03 + 0.01 + 0.36)) < 1e-6


@pytest.mark.parametrize("bad", ["negative_row", "truncated"])
def test_paid_runner_refuses_a_tampered_ledger(monkeypatch, tmp_path, bad):
    """In-process paid mode: tripwires make it impossible to reach the copy or any model,
    even if the code under test is broken (errors.md 4: this test once ran for real)."""
    import litellm

    def tripwire(*a, **k):
        raise AssertionError("the paid path went past the ledger check")
    lines = list(REAL[:D.LEDGER_PREFIX[0]])
    lines = lines + [row("reserve_worst_case", -40.0)] if bad == "negative_row" else lines[:-200]
    monkeypatch.setattr(rf, "LEDGER", ledger_with(tmp_path, lines))
    monkeypatch.setattr(rf.ra, "dirty", lambda paths: False)
    for target, name in ((rf.rm, "lock"), (rf.rm, "preflight"), (rf.rm, "setup_checksum"),
                         (litellm, "acompletion"), (litellm, "completion")):
        monkeypatch.setattr(target, name, tripwire)
    with pytest.raises(SystemExit, match="REFUSED"):
        rf.main(["--run", "tripwire-test"])
    assert not (HERE / "results" / "tripwire-test").exists()


def test_shortened_case_list_is_refused():
    ids = [f"{i}:{c.get('source_task_id', '')}"[:120] for i, c in enumerate(FU)]
    sf.check_ids(ids, 14, True)
    sf.check_ids(ids[:7], 7, False)
    for args in ((ids[:1], 14, True), (ids[:7], 7, True), (ids[:1], 14, False), (ids[1:2], 1, False)):
        with pytest.raises(sf.Refused):
            sf.check_ids(*args)


def test_summary_that_disagrees_with_the_files_is_refused(good, tmp_path):
    dst = tmp_path / "results" / "g"
    shutil.copytree(good[0] / "g", dst)
    m = json.loads((dst / "manifest.json").read_text())
    m["summary"]["valid_blocks"] += 1
    (dst / "manifest.json").write_text(json.dumps(m, indent=1))
    rm.seal_path(dst).parent.mkdir()
    rm.seal_path(dst).write_text(rm.seal(dst))
    with pytest.raises(sf.Refused, match="recomputed"):
        sf.load(dst, is_registered=False)


def test_a_scenario_without_its_certificate_is_refused(monkeypatch, tmp_path):
    cert = json.loads(D.CERTIFICATE.read_text())
    cert["cases"][-1]["certified"] = False                       # walk 2680, in the suite
    p = tmp_path / "certificate.json"
    p.write_text(json.dumps(cert))
    monkeypatch.setattr(D, "CERTIFICATE", p)
    with pytest.raises(SystemExit, match="certificate"):
        rf.suite_cases(None)
