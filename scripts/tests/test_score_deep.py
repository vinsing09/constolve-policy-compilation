"""Tests for scripts/score_deep.py on runs made by the real τ³ harness (fake model only,
mode "test": 2 tasks × 2 trials), plus the real August per-task file.

    .venv/bin/python -B -m pytest -q -p no:cacheprovider scripts/tests/test_score_deep.py
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
PROJECT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "tests"))
sys.dont_write_bytecode = True
import run_deep as rd  # noqa: E402
import run_manifest as rm  # noqa: E402
import score_deep as sd  # noqa: E402

sd._real_score = sd.score
from test_run_deep import fake_run  # noqa: E402

REWARDS = {"v0": {(0, 0): 1.0, (0, 1): 0.0, (1, 0): 0.0, (1, 1): 0.0},
           "v2_deep": {(0, 0): 1.0, (0, 1): 1.0, (1, 0): 1.0, (1, 1): 0.0}}


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("pair")
    for v in sd.VERSIONS:
        p, _, _ = fake_run(tmp, f"t-{v}", version=v)
        assert p.returncode == 0, p.stderr[-2000:]
    return tmp


def runs(root):
    return {v: root / "results" / f"t-{v}" for v in sd.VERSIONS}


def reseal(run):
    rm.seal_path(run).write_text(rm.seal(run))


def set_rewards(root, rewards=REWARDS):
    """(task position, trial) -> reward, written into τ³'s own results.json, then resealed."""
    ids = rd.base_ids()[:2]
    for v, run in runs(root).items():
        f = run / "tau3" / "results.json"
        d = json.loads(f.read_text())
        for x in d["simulations"]:
            x["reward_info"]["reward"] = rewards[v][(ids.index(x["task_id"]), x["trial"])]
        f.write_text(json.dumps(d))
        reseal(run)


@pytest.fixture
def scored_pair(pair, tmp_path):
    root = tmp_path / "w"
    shutil.copytree(pair, root)
    set_rewards(root)
    return root


def test_per_trial_rates_gap_and_pass_k_by_hand(scored_pair):
    s = sd.score(runs(scored_pair), mode="test")["scores"]["all"]
    assert s["v0"]["pass_rate"]["per_trial"] == [50.0, 0.0]
    assert s["v2_deep"]["pass_rate"]["per_trial"] == [100.0, 50.0]
    assert s["gap"]["per_trial"] == [50.0, 50.0] and s["gap"]["half_range"] == 0.0
    assert s["v0"]["pass_rate"]["mean"] == 25.0 and s["v0"]["pass_rate"]["half_range"] == 25.0
    # task 0: v0 1/2, v2 2/2; task 1: v0 0/2, v2 1/2
    assert s["v0"]["pass_hat_k"] == {1: 25.0, 2: 0.0}
    assert s["v2_deep"]["pass_hat_k"] == {1: 75.0, 2: 50.0}


def test_matches_taus_own_metrics(scored_pair):
    sys.path.insert(0, str(rd.TAU3 / "src"))
    from tau2.data_model.simulation import Results
    from tau2.metrics.agent_metrics import compute_metrics
    s = sd.score(runs(scored_pair), mode="test")["scores"]["all"]
    for v, run in runs(scored_pair).items():
        m = compute_metrics(Results.load(run / "tau3" / "results.json"))
        assert m.pass_hat_ks == pytest.approx({k: x / 100 for k, x in s[v]["pass_hat_k"].items()})
        assert 100 * m.avg_reward == pytest.approx(s[v]["pass_rate"]["mean"])


def test_august_rates_reproduce_the_deep_policy_trace():
    a = sd.august(PROJECT / "deep_policy", sd.subsets(PROJECT / "deep_policy"))
    r = lambda xs: [round(x, 1) for x in xs]  # noqa: E731
    assert r(a["v0"]["all"]) == [29.8, 32.5, 26.3] and r(a["v2_deep"]["all"]) == [57.9, 54.4]
    assert r(a["v0"]["unseen_51"]) == [66.7, 72.5, 58.8]
    assert r(a["v2_deep"]["unseen_51"]) == [80.4, 76.5]
    assert r(a["v0"]["seen_63"]) == [0.0, 0.0, 0.0] and r(a["v2_deep"]["seen_63"]) == [39.7, 36.5]


def _edit_manifest(run, edit):
    f = run / "manifest.json"
    m = json.loads(f.read_text())
    edit(m)
    f.write_text(json.dumps(m))
    reseal(run)


@pytest.mark.parametrize("mutation,why", [
    ("unsealed", "does not match its seal"), ("problems", "is not a clean test run"),
    ("wrong_version", "is not a clean test run"), ("swapped", "is not a clean test run"),
    ("summary_drift", "no longer give the manifest's summary"),
    ("config_differs", "configs differ"), ("overlap", "not back to back"),
    ("long_gap", "not back to back"), ("reversed_v0", "not back to back"),
    ("reversed_v2", "not back to back"),
    ("keys_differ", "do not share the exact")])
def test_anything_but_the_clean_pair_is_refused(scored_pair, mutation, why):
    r = runs(scored_pair)
    if mutation == "unsealed":
        (r["v0"] / "stdout.log").write_text("changed")
    elif mutation == "problems":
        _edit_manifest(r["v0"], lambda m: m["summary"]["problems"].append("x"))
    elif mutation == "wrong_version":
        _edit_manifest(r["v0"], lambda m: m.update(version="v2_deep"))
    elif mutation == "swapped":
        r = {"v0": r["v2_deep"], "v2_deep": r["v0"]}
    elif mutation == "summary_drift":
        _edit_manifest(r["v0"], lambda m: m["summary"].update(cost_usd=99.0))
    elif mutation == "config_differs":
        _edit_manifest(r["v0"], lambda m: m["config"].update(max_concurrency=4))
    elif mutation == "overlap":
        _edit_manifest(r["v2_deep"], lambda m: m.update(start=json.loads(
            (r["v0"] / "manifest.json").read_text())["start"]))
    elif mutation == "reversed_v0":
        _edit_manifest(r["v0"], lambda m: m.update(start="2099-01-01T00:00:00+0530"))
    elif mutation == "reversed_v2":
        _edit_manifest(r["v2_deep"], lambda m: m.update(end="2000-01-01T00:00:00+0530"))
    elif mutation == "long_gap":
        _edit_manifest(r["v0"], lambda m: m.update(start="2000-01-01T00:00:00+0530",
                                                   end="2000-01-01T01:00:00+0530"))
    else:
        f = r["v2_deep"] / "tau3" / "results.json"
        d = json.loads(f.read_text())
        d["simulations"][0]["seed"] = 1
        f.write_text(json.dumps(d))
        reseal(r["v2_deep"])
        with pytest.raises(sd.Refused):
            sd.score(r, mode="test")
        return
    with pytest.raises(sd.Refused, match=why):
        sd.score(r, mode="test")


def test_back_to_back_across_midnight_is_accepted(scored_pair):
    r = runs(scored_pair)
    _edit_manifest(r["v0"], lambda m: m.update(start="2026-10-20T22:54:53+0530",
                                               end="2026-10-21T00:40:00+0530"))
    _edit_manifest(r["v2_deep"], lambda m: m.update(start="2026-10-21T00:40:05+0530",
                                                    end="2026-10-21T02:20:00+0530"))
    sd.score(r, mode="test")


@pytest.mark.parametrize("gap,ok", [("06:44:39", True), ("07:56:04", True), ("07:56:05", False)])
def test_gap_limit_is_eight_hours(scored_pair, gap, ok):
    """The real pair: v0 ends 23:56:04, v2_deep rerun starts 06:44:39 (6 h 48 min)."""
    r = runs(scored_pair)
    _edit_manifest(r["v0"], lambda m: m.update(start="2026-10-20T22:54:53+0530",
                                               end="2026-10-20T23:56:04+0530"))
    _edit_manifest(r["v2_deep"], lambda m: m.update(start=f"2026-10-21T{gap}+0530",
                                                    end="2026-10-21T09:00:00+0530"))
    if ok:
        sd.score(r, mode="test")
    else:
        with pytest.raises(sd.Refused, match="not back to back"):
            sd.score(r, mode="test")


def test_pairing_check_on_its_own(scored_pair):
    """keys_differ above is caught earlier by the run check; prove check_pair catches it too."""
    r = runs(scored_pair)
    loaded = {v: sd.load(r[v], v, "test") for v in sd.VERSIONS}
    ms, sims = {v: x[0] for v, x in loaded.items()}, {v: x[1] for v, x in loaded.items()}
    sd.check_pair(ms, sims)
    sims["v2_deep"] = [{**sims["v2_deep"][0], "seed": 1}, *sims["v2_deep"][1:]]
    with pytest.raises(sd.Refused, match="do not share the exact"):
        sd.check_pair(ms, sims)


def test_cli_refuses_uncommitted_seals_and_leaves_no_score(scored_pair, capsys):
    r = runs(scored_pair)
    out = r["v0"].parent / "deep-score.json"
    out.write_text("{}")
    assert sd.main([str(r["v0"]), str(r["v2_deep"])]) == 1
    assert "not committed" in capsys.readouterr().err and not out.exists()


@pytest.mark.parametrize("reward,ok", [(1.0, True), (1 - 1e-7, True), (1 + 1e-7, True),
                                       (0.999, False), (0.5, False), (0.0, False), (None, False)])
def test_success_is_taus_definition(reward, ok):
    sys.path.insert(0, str(rd.TAU3 / "src"))
    from tau2.metrics.agent_metrics import is_successful
    assert sd.success(reward) is ok
    if reward is not None:
        assert is_successful(reward) is ok


def test_cli_scores_a_committed_clean_pair(scored_pair, monkeypatch, capsys):
    """score.committed raises on failure and returns None on success."""
    monkeypatch.setattr(sd, "committed", lambda seal: None)
    monkeypatch.setattr(sd, "score", lambda runs, **k: sd.__dict__["_real_score"](runs, mode="test"))
    r = runs(scored_pair)
    assert sd.main([str(r["v0"]), str(r["v2_deep"])]) == 0, capsys.readouterr().err
    out = json.loads((r["v0"].parent / "deep-score.json").read_text())
    assert out["scores"]["all"]["gap"]["per_trial"] == [50.0, 50.0]


def test_subsets_only_count_their_own_tasks(scored_pair):
    """A 1-task subset of the 2 tasks: task 0 only (v0 1/2, v2 2/2)."""
    sims = {v: sd.load(runs(scored_pair)[v], v, "test")[1] for v in sd.VERSIONS}
    one = {rd.base_ids()[0]}
    assert sd.rates(sims["v0"], one, 2) == [100.0, 0.0]
    assert sd.rates(sims["v2_deep"], one, 2) == [100.0, 100.0]
    assert sd.pass_k(sims["v0"], one, 2) == {1: 50.0, 2: 0.0}
    assert sd.pass_k(sims["v2_deep"], one, 2) == {1: 100.0, 2: 100.0}
