"""Tests for rules.py, run_ablation.py and score_ablation.py.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest -q -p no:cacheprovider \
        experiments/rule-ablation/tests

Driver tests use a full copy of code-fixed and litellm's mock_response (no API calls).
Scorer tests use the real full-fixed-2 traces (production shape: 24 scenarios, real
criteria and trace files).
"""
import glob
import itertools
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
PROJECT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PROJECT / "scripts"))
import rules  # noqa: E402
import run_manifest as rm  # noqa: E402
import score_ablation as sa  # noqa: E402
from runlog import tree_hash  # noqa: E402

CONTROL = (HERE / "policies" / "control.md").read_text()
META = json.loads((HERE / "variants.json").read_text())
FF2 = PROJECT / "results" / "full-fixed-2"


def test_control_file_is_the_run_policy():
    assert rules.sha(CONTROL) == rules.CONTROL_SHA == META["control"]["sha256"]


@pytest.mark.parametrize("rule", list(rules.RULES))
def test_variant_is_control_minus_exactly_its_spans(rule):
    text = (HERE / "policies" / f"{rule}.md").read_text()
    assert rules.sha(text) == META[rule]["sha256"]
    spans = rules.RULES[rule]["spans"]
    assert all(CONTROL.count(s) == 1 for s in spans)
    assert len(CONTROL) - len(text) == sum(map(len, spans))
    rebuilt = CONTROL
    for s in spans:
        rebuilt = rebuilt.replace(s, "", 1)
    assert rebuilt == text
    assert all(s not in text for s in spans)


def test_remove_refuses_a_span_that_is_not_unique():
    with pytest.raises(ValueError):
        rules.remove(CONTROL.replace("Use `check_sim_status()`", "X"), "R5_escalate_unfixable")


def test_frozen_ties_match_the_rule_definitions():
    assert json.loads((HERE / "ties.json").read_text()) == rules.ties()


def test_tie_counts_and_boundaries_on_the_real_suite():
    t = {r: {tuple(x) for x in v} for r, v in rules.ties().items()}
    scen = {r: sorted({i for i, _ in v}) for r, v in t.items()}
    assert {r: len(v) for r, v in t.items()} == {
        "R1_airplane": 11, "R2_apn": 20, "R3_payment": 6, "R4_suspension": 9,
        "R5_escalate_unfixable": 10, "R6_transfer_scope": 24}
    assert (18, "s1_reboot_device") not in t["R2_apn"]          # no APN reset in case 18
    assert (6, "s1_reboot_device") in t["R2_apn"]
    assert not any(c.endswith("get_data_usage") for _, c in t["R5_escalate_unfixable"])
    assert scen["R5_escalate_unfixable"] == [3, 5, 12, 17, 23]
    assert sum(c == "no_premature_escalation" for _, c in t["R6_transfer_scope"]) == 19
    shared = t["R5_escalate_unfixable"] & t["R6_transfer_scope"]
    assert {c for _, c in shared} <= {"s1_transfer_to_human_agents",
                                      "s2_transfer_to_human_agents",
                                      "s3_transfer_to_human_agents"} and len(shared) == 5


def brute(d):
    d = [x for x in d if abs(x) > 1e-12]
    obs = sum(d)
    sums = [sum(s * x for s, x in zip(signs, d))
            for signs in itertools.product((1, -1), repeat=len(d))]
    return sum(s >= obs - 1e-9 for s in sums) / len(sums) if d else 1.0


def test_signflip_exact_matches_brute_force():
    rng = random.Random(1)
    for _ in range(30):
        d = [rng.choice([0, 1 / 3, 2 / 3, 1, -1 / 3, 0.5, -0.25]) for _ in range(rng.randint(1, 9))]
        assert sa.signflip_p(d) == pytest.approx(brute(d))


def test_signflip_floor_depends_on_scenario_count():
    assert sa.signflip_p([1, 1, 1]) == 1 / 8          # 3 scenarios can never reach p < 0.05
    assert sa.signflip_p([1, 1, 1, 1]) == 1 / 16
    assert sa.signflip_p([1] * 5) == 1 / 32
    assert sa.signflip_p([0, 0]) == 1.0
    assert sa.signflip_p([0.2] * 24) < 1e-4                    # Monte Carlo branch
    assert sa.signflip_p([-1] * 6) == 1.0


def ff2_rows():
    """full-fixed-2: with_policy as control, stub as every variant."""
    rows = []
    for f in glob.glob(str(FF2 / "traces" / "*.json")):
        d = json.loads(Path(f).read_text())
        i = int(d["case"].split(":")[0])
        arms = ["control"] if d["arm"] == "with_policy" else list(rules.RULES)
        rows += [{"arm": a, "rep": d["repeat"], "scen": i, "cid": c["id"], "ok": c["ok"]}
                 for a in arms for c in d["criteria"]]
    return rows


def test_scores_on_production_traces_match_hand_counts():
    out = sa.score(ff2_rows(), json.loads((HERE / "ties.json").read_text()))
    r1 = out["R1_airplane"]["tied"]
    assert (r1["n_control"], r1["n_scenarios"]) == (55, 11)
    assert r1["control_pct"] == pytest.approx(100 * 52 / 55)
    assert r1["variant_pct"] == pytest.approx(100 * 24 / 55)
    assert r1["caught"]
    r3 = out["R3_payment"]["tied"]
    assert (r3["control_pct"], r3["variant_pct"], r3["p"]) == (100, 0, 1 / 8)
    assert not r3["caught"]                                     # 100-point drop, 3 scenarios
    r4 = out["R4_suspension"]["tied"]
    assert r4["drop"] < 0 and not r4["caught"]
    untied = out["R1_airplane"]["untied"]
    assert untied["n_scenarios"] == 24 and untied["drop"] > 20
    assert set(out["pair"]) == set(rules.PAIR)


def test_identical_arms_drop_nothing_and_are_not_caught():
    rows = [r for r in ff2_rows() if r["arm"] == "control"]
    rows += [{**r, "arm": "R2_apn"} for r in rows]
    out = sa.score(rows, json.loads((HERE / "ties.json").read_text()))
    assert out["R2_apn"]["tied"]["drop"] == 0 and out["R2_apn"]["tied"]["p"] == 1.0
    assert not out["R2_apn"]["untied"]["caught"]


# ---------- driver, on a copy of code-fixed with fake model calls ----------

@pytest.fixture(scope="module")
def copy(tmp_path_factory):
    dst = tmp_path_factory.mktemp("c") / "code-fixed"
    shutil.copytree(PROJECT / "code-fixed", dst, symlinks=True)
    yield dst
    shutil.rmtree(dst)          # the copy holds .env


def run(copy, results, name, arms="control,R1_airplane", **env):
    clean = {k: v for k, v in os.environ.items()
             if not (k in rm.EXACT or k.startswith(rm.PREFIXES))}
    p = subprocess.run(
        [sys.executable, "-B", str(HERE / "tests" / "fake_ablation_run.py"), "--run", name,
         "--cases", "1", "--repeats", "1", "--arms", arms, "--copy", str(copy),
         "--results", str(results)],
        env={**clean, "PYTHONDONTWRITEBYTECODE": "1", "FAKE_LEDGER": str(results / "ledger.jsonl"), **env},
        capture_output=True, text=True, timeout=600)
    m = results / name / "manifest.json"
    return p, (json.loads(m.read_text()) if m.exists() else None)


@pytest.fixture(scope="module")
def good(copy, tmp_path_factory):
    results = tmp_path_factory.mktemp("results")
    before = tree_hash(copy)
    p, m = run(copy, results, "g")
    return results, p, m, before


def test_good_run_counts_and_gives_each_arm_its_own_prompt(good, copy):
    results, p, m, before = good
    assert p.returncode == 0, p.stderr[-3000:]
    s = m["summary"]
    assert s["problems"] == [] and s["conversations"] == {"clean": 2}
    assert s["valid_blocks"] == s["blocks"] == 1
    assert {a: v["policy_sha256"] for a, v in m["arms"].items()} == {
        "control": rules.CONTROL_SHA, "R1_airplane": META["R1_airplane"]["sha256"]}
    agent_calls = [json.loads(x) for x in (results / "g" / "calls.jsonl").read_text().splitlines()]
    assert {c["conv"].split("|")[0] for c in agent_calls if c["caller"] == "agent"} == \
        {"control", "R1_airplane"}
    assert len(list((results / "g" / "traces").glob("*.json"))) == 2
    assert tree_hash(copy) == before and not (copy / "logs" / "bench").exists()
    assert rm.seal_path(results / "g").read_text() == rm.seal(results / "g")


def test_good_run_scores_without_commit(good):
    results = good[0]
    m, rows, n = sa.load(results / "g", registered=False)
    assert n == 1 and {r["arm"] for r in rows} == {"control", "R1_airplane"}


def test_tampered_run_is_refused(good, tmp_path):
    src = good[0] / "g"
    dst = tmp_path / "results" / "g"
    shutil.copytree(src, dst)
    shutil.copy(rm.seal_path(src), tmp_path / "seals.tmp")
    (tmp_path / "seals").mkdir()
    shutil.copy(tmp_path / "seals.tmp", rm.seal_path(dst))
    f = next((dst / "traces").glob("*.json"))
    d = json.loads(f.read_text())
    d["criteria"][0]["ok"] = not d["criteria"][0]["ok"]
    ok = [c["ok"] for c in d["criteria"]]
    d["solved"], d["actions_done"] = all(ok), sum(ok)       # internally consistent edit
    f.write_text(json.dumps(d))
    with pytest.raises(sa.Refused, match="seal"):
        sa.load(dst, registered=False)


def test_budget_too_small_starts_nothing(copy, tmp_path):
    p, m = run(copy, tmp_path, "b", FAKE_CAP_USD="0.4", FAKE_EST="0")
    assert p.returncode == 1
    assert m["summary"]["conversations"] == {"skipped": 2}
    assert "no valid blocks" in m["summary"]["problems"]
    assert m["summary"]["model_calls"] == 0


def test_call_without_room_for_its_worst_case_is_refused(copy, tmp_path):
    p, m = run(copy, tmp_path, "h", arms="control", FAKE_CAP_USD="0.1", FAKE_EST="0",
               FAKE_NEXT="0", FAKE_INFLIGHT="0", FAKE_WORST="0")
    assert m["summary"]["conversations"] == {"budget_stopped": 1}
    assert m["summary"]["blocks_lost_to_failures"] == 0
    assert m["summary"]["model_calls"] == 0 and m["guard_spent_usd"] == 0
    assert json.loads((tmp_path / "h" / "stopped.json").read_text()) == \
        list(m["summary"]["status"])


def test_ledger_counts_earlier_runs_in_the_ledger_folder(copy, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"run": "old", "conv": None, "usd": 59.99, "kind": "call"}) + "\n")
    for root in ("a", "b"):                       # any results root, one ledger
        (tmp_path / root).mkdir()
        p, m = run(copy, tmp_path / root, "l", FAKE_EST="0.01", FAKE_LEDGER=str(ledger))
        assert p.returncode == 2 and "exceeds" in p.stderr


def test_spend_is_written_to_one_ledger_across_runs_and_roots(copy, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    for root in ("a", "b"):
        (tmp_path / root).mkdir()
        p, m = run(copy, tmp_path / root, "r", arms="control", FAKE_LEDGER=str(ledger))
        assert p.returncode == 0, p.stderr[-2000:]
    rows = [json.loads(x) for x in ledger.read_text().splitlines()]
    assert {r["run"] for r in rows} == {"r"} and len(rows) == 4 * m["summary"]["model_calls"]
    assert sum(r["usd"] for r in rows) == pytest.approx(2 * m["summary"]["cost_usd"], abs=1e-3)


def test_failed_call_stays_charged_its_worst_case(copy, tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    p, m = run(copy, tmp_path, "f", arms="control", FAKE_FAIL="1", FAKE_LEDGER=str(ledger))
    rows = [json.loads(x) for x in ledger.read_text().splitlines()]
    failed = {r["conv"] for r in rows if r["kind"] == "settle_failed_call_worst_case"}
    assert failed
    for conv in failed:                                         # 32,768 output tokens alone
        assert sum(r["usd"] for r in rows if r["conv"] == conv and
                   r["kind"] != "settle_call") > 0.26
    assert m["summary"]["conversations"] == {"error": 1}


def test_concurrent_calls_never_pass_the_cap(monkeypatch, tmp_path):
    import asyncio
    import types
    import run_ablation as ra
    from runlog import CONV
    monkeypatch.setattr(ra, "LEDGER", tmp_path / "ledger.jsonl")
    kw = {"messages": [{"role": "user", "content": "x" * 1000}], "tools": None}
    worst = ra.worst_cost(kw, 2e-6, 8e-6)

    async def acompletion(**_):
        await asyncio.sleep(0.01)
        return object()
    fake = types.SimpleNamespace(acompletion=acompletion,
                                 model_cost={rm.MODEL: {"input_cost_per_token": 2e-6,
                                                        "output_cost_per_token": 8e-6}},
                                 completion_cost=lambda **_: worst)   # every call at its bound
    budget = ra.Budget(3 * worst + 1e-9)
    ra.install_guard(fake, budget, {})

    async def one(i):
        CONV.set(f"c|0|{i}")
        try:
            await fake.acompletion(**kw)
            return 1
        except ra.BudgetStop:
            return 0
    async def main():
        return await asyncio.gather(*[one(i) for i in range(8)])
    done = asyncio.run(main())
    assert sum(done) == 3 and budget.spent <= budget.hard and abs(budget.reserved) < 1e-12


def test_estimate_over_the_cap_is_refused_before_any_run(copy, tmp_path):
    p, m = run(copy, tmp_path, "e", FAKE_EST="40")
    assert p.returncode == 2 and m is None and "exceeds" in p.stderr


def test_unknown_or_repeated_arm_is_refused(copy, tmp_path):
    for arms in ("control,R9", "control,control"):
        p, m = run(copy, tmp_path, "u", arms=arms)
        assert p.returncode != 0 and m is None


def registered_manifest(good):
    m = json.loads((good[0] / "g" / "manifest.json").read_text())
    copy = rm.PROJECT / "code-fixed"
    m.update(arms={a: {} for a in rules.ARMS}, cases=24, repeats=3, copy="code-fixed",
             copy_checksum=rm.stored("checksum", copy), copy_tree_hash=rm.stored("tree", copy),
             command=[sys.executable, str(HERE / "run_ablation.py"), "--run", "g"],
             project_commit=subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT,
                                           capture_output=True, text=True).stdout.strip())
    return m


@pytest.mark.parametrize("change", [
    {"arms": {a: {} for a in rules.ARMS[:-1]}}, {"cases": 23}, {"repeats": 2}, {"cap": 20},
    {"copy": "code-original"}, {"copy_tree_hash": "0" * 64}, {"project_commit": "f" * 40},
    {"command": [sys.executable, str(HERE / "run_ablation.py"), "--run", "g", "--copy", "x"]},
    {"command": [sys.executable, str(HERE / "x.py"), "--run", "g"]},
    {"temperatures": {"agent": 0.2, "customer": 0.7}}])
def test_unregistered_runs_are_refused(good, change):
    m = {**registered_manifest(good), **change}
    with pytest.raises(sa.Refused):
        sa.check_registered(good[0] / "g", m)


@pytest.mark.parametrize("field,kind", [("copy_checksum", "checksum"), ("copy_tree_hash", "tree")])
def test_forged_stored_hashes_do_not_pass(good, monkeypatch, field, kind):
    real = rm.stored
    m = {**registered_manifest(good), field: "a" * 64}
    monkeypatch.setattr(rm, "stored", lambda k, c: "a" * 64 if k == kind else real(k, c))
    with pytest.raises(sa.Refused):
        sa.check_registered(good[0] / "g", m)


def test_scorer_only_change_since_the_run_is_refused(good, monkeypatch):
    def git_ok(*args):
        return not (args[0] == "diff" and "experiments/rule-ablation/score_ablation.py" in args)
    monkeypatch.setattr(sa.sc, "git_ok", git_ok)
    with pytest.raises(sa.Refused, match="pinned"):
        sa.check_registered(good[0] / "g", registered_manifest(good))


def test_uncommitted_pinned_change_is_refused_at_scoring(good, monkeypatch):
    monkeypatch.setattr(sa.ra, "dirty", lambda paths: True)
    with pytest.raises(sa.Refused, match="uncommitted"):
        sa.load(good[0] / "g", registered=True)


def test_paid_runs_cannot_choose_copy_or_results(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "RULE_ABLATION_TEST"}
    for opt in ("--copy", "--results"):
        p = subprocess.run([sys.executable, "-B", str(HERE / "run_ablation.py"), "--run", "x",
                            opt, str(tmp_path)], env={**env, "PYTHONDONTWRITEBYTECODE": "1"},
                           capture_output=True, text=True, timeout=120)
        assert p.returncode == 2 and "unrecognized arguments" in p.stderr


def test_second_paid_run_waits_for_the_ledger_lock(tmp_path, monkeypatch):
    import run_ablation as ra
    monkeypatch.setattr(ra, "LEDGER", tmp_path / "ledger.jsonl")
    first = ra.ledger_lock()
    assert first is not None and ra.ledger_lock() is None
    first.close()
    assert ra.ledger_lock() is not None


def test_scorer_and_engine_hash_files_are_pinned():
    for f in ("experiments/rule-ablation/score_ablation.py", "scripts/score.py",
              "experiments/rule-ablation/prereg.md", "scripts/checksum-fixed.txt",
              "scripts/tree-fixed.txt"):
        assert f in sa.ra.PINNED


def test_registered_manifest_passes_when_pinned_files_are_committed(good):
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *sa.ra.PINNED], cwd=PROJECT,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        pytest.skip("pinned files have uncommitted changes")
    sa.check_registered(good[0] / "g", registered_manifest(good))


def test_smoke_shaped_run_is_refused_as_a_registered_run(good):
    with pytest.raises(Exception):
        sa.load(good[0] / "g", registered=True)


def test_duplicate_trace_is_refused(good, tmp_path):
    dst = tmp_path / "results" / "g"
    shutil.copytree(good[0] / "g", dst)
    f = next((dst / "traces").glob("*.json"))
    shutil.copy(f, dst / "traces" / ("zz_" + f.name))
    (tmp_path / "seals").mkdir()
    rm.seal_path(dst).write_text(rm.seal(dst))
    with pytest.raises(sa.Refused, match="duplicate|recomputed"):
        sa.load(dst, registered=False)
    m = json.loads((dst / "manifest.json").read_text())
    want = {a: v["policy_sha256"] for a, v in m["arms"].items()}
    again = sa.ra.check(dst, list(m["arms"]), 1, m["case_ids"], {}, want, set())
    assert any(x.startswith("duplicate trace") for x in again["problems"])
