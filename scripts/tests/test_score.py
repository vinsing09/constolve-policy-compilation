"""Tests for scripts/score.py on real saved files: the five August runs behind +30.9
and this project's wrapper-sealed full original-code run (results/full-original-2).

    .venv/bin/python -B -m pytest -q -p no:cacheprovider scripts/tests/test_score.py
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.dont_write_bytecode = True
SCRIPTS = Path(__file__).resolve().parents[1]
PROJECT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(PROJECT / "code-original" / "scripts"))
import analyse_run  # noqa: E402
import run_manifest as rm  # noqa: E402
import runlog  # noqa: E402
import score as sc  # noqa: E402

AUGUST = Path.home() / "Documents/my_projects/agentops-backend/.worktrees/v2-serverB/logs/bench_runs"
RUNS = {"run23_ours": 29.5, "run25_run23suite": 31.6, "ours_r3": 29.5, "ours_r4": 34.3,
        "ours_r5": 29.7}
REAL = PROJECT / "results" / "full-original-2"


def raw(folder):
    return [json.loads(f.read_text()) for f in sorted(folder.glob("*.json"))]


def case_ids(folder):
    return sorted({d["case"] for d in raw(folder)}, key=lambda c: int(c.split(":")[0]))


def real_copy(tmp_path):
    """The real sealed run, copied to <tmp>/results/full-original with its seal."""
    run = tmp_path / "results" / REAL.name
    shutil.copytree(REAL, run, ignore=shutil.ignore_patterns("score.json"))
    reseal(run)
    return run


def reseal(run):
    rm.seal_path(run).parent.mkdir(exist_ok=True)
    rm.seal_path(run).write_text(rm.seal(run))


@pytest.mark.parametrize("name,gap", RUNS.items())
def test_august_runs_match_analyse_run_exactly(name, gap):
    folder = AUGUST / name
    s = sc.score(sc.load(folder, case_ids(folder), 5), 5)
    rows = raw(folder)
    for arm, short in (("with_policy", "with"), ("stub", "stub")):
        want = analyse_run.scores([d for d in rows if analyse_run.arm_of(d) == short])
        got = s["pooled"][arm]
        assert got["n"] == want["n"] == 120
        for k in sc.METRICS:
            assert got[k] == pytest.approx(want[k], abs=1e-9), (arm, k)
    assert round(s["pooled"]["gap"]["action"], 1) == gap
    assert s["repeat_gaps"]["action"]["mean"] == pytest.approx(s["pooled"]["gap"]["action"])


def test_real_run_is_the_committed_seal_and_matches_analyse_run(tmp_path):
    assert rm.seal(REAL) == rm.seal_path(REAL).read_text()
    sc.committed(rm.seal_path(REAL))
    run = real_copy(tmp_path)
    s = sc.score_run(run)
    rows = raw(run / "traces")
    assert s["conversations"] == len(rows) == 240
    for arm, short in (("with_policy", "with"), ("stub", "stub")):
        want = analyse_run.scores([d for d in rows if analyse_run.arm_of(d) == short])
        for k in sc.METRICS:
            assert s["pooled"][arm][k] == pytest.approx(want[k], abs=1e-9), (arm, k)


def test_per_repeat_gaps_mean_and_half_range_match_analyse_run_per_repeat(tmp_path):
    run = real_copy(tmp_path)
    s = sc.score_run(run)
    rows = raw(run / "traces")
    for k in sc.METRICS:
        want = []
        for r in range(5):
            w, b = (analyse_run.scores([d for d in rows if d["repeat"] == r
                                        and analyse_run.arm_of(d) == a])[k]
                    for a in ("with", "stub"))
            want.append(w - b)
        got = s["repeat_gaps"][k]
        assert got["per_repeat"] == pytest.approx(want, abs=1e-9), k
        assert got["mean"] == pytest.approx(sum(want) / 5, abs=1e-9), k
        assert got["half_range"] == pytest.approx((max(want) - min(want)) / 2, abs=1e-9), k


def test_one_flipped_criterion_moves_only_its_repeat():
    folder = AUGUST / "ours_r3"
    rows = sc.load(folder, case_ids(folder), 5)
    before = sc.score(rows, 5)
    d = next(d for d in rows if d["arm"] == "with_policy" and d["repeat"] == 2
             and any(m and not ok for ok, m in zip(d["ok"], d["must"])))
    i = next(i for i, (ok, m) in enumerate(zip(d["ok"], d["must"])) if m and not ok)
    d["ok"][i] = True
    after = sc.score(rows, 5)
    moved = [a - b for a, b in zip(after["repeat_gaps"]["action"]["per_repeat"],
                                   before["repeat_gaps"]["action"]["per_repeat"])]
    assert moved == pytest.approx([0, 0, 100 / 74, 0, 0])
    assert after["pooled"]["gap"]["action"] - before["pooled"]["gap"]["action"] == \
        pytest.approx(100 / 370)


def _json(f, edit):
    d = json.loads(f.read_text())
    edit(d)
    f.write_text(json.dumps(d))


def _edit(run, mutation):
    t = run / "traces"
    files = sorted(t.glob("*.json"))
    one = lambda edit: _json(files[0], edit)  # noqa: E731
    man = lambda edit: _json(run / "manifest.json", edit)  # noqa: E731
    rpt = lambda edit: _json(run / "report.json", edit)  # noqa: E731
    if mutation == "missing":
        files[0].unlink()
    elif mutation == "duplicate":
        shutil.copy(files[0], t / "copy.json")
    elif mutation == "foreign_case":
        one(lambda d: d.update(case="99:[graph:x]y"))
    elif mutation == "foreign_arm":
        one(lambda d: d.update(arm="without"))
    elif mutation == "string_repeat":
        one(lambda d: d.update(repeat=str(d["repeat"])))
    elif mutation == "not_json":
        files[0].write_text("{")
    elif mutation == "ok_is_int":
        one(lambda d: d["criteria"][0].update(ok=1))
    elif mutation == "criterion_dropped":
        one(lambda d: d["criteria"].pop())
    elif mutation == "no_criteria":
        one(lambda d: d.update(criteria=[]))
    elif mutation == "ok_flipped_only":
        one(lambda d: d["criteria"][0].update(ok=not d["criteria"][0]["ok"]))
    elif mutation == "trace_solved_wrong":
        one(lambda d: d.update(solved=not d["solved"]))
    elif mutation == "stored_rule":
        one(lambda d: d["criteria"][0].update(rule="tool_not_called"))
    elif mutation == "history_customer_error":
        one(lambda d: d["history"].append({"role": "user",
                                           "content": runlog.SIM_ERROR + " x)"}))
    elif mutation == "report_row_k":
        rpt(lambda d: d["runs"][0].update(k=d["runs"][0]["k"] - 1))
    elif mutation == "report_row_missing":
        rpt(lambda d: d["runs"].pop())
    elif mutation == "report_row_error":
        rpt(lambda d: d["runs"][0].update(error="TimeoutError: x"))
    elif mutation == "report_repeats":
        rpt(lambda d: d.update(repeats=4))
    elif mutation == "case_text":
        old = json.loads(files[0].read_text())["case"]
        new = old.split(":", 1)[0] + ":renamed"
        for f in [*files, run / "manifest.json", run / "report.json"]:
            f.write_text(f.read_text().replace(json.dumps(old), json.dumps(new)))
    elif mutation == "manifest_problems":
        man(lambda d: d["summary"].update(problems=["failed_calls: 1"]))
    elif mutation == "manifest_no_summary":
        man(lambda d: d.pop("summary"))
    elif mutation == "manifest_bench_exit":
        man(lambda d: d.update(bench_exit=1))
    elif mutation == "manifest_copy_changed":
        man(lambda d: d["summary"].update(copy_unchanged=False))
    elif mutation == "manifest_temperature":
        man(lambda d: d["temperatures"].update(customer=1.0))
    elif mutation == "run_id":
        man(lambda d: d.update(run="full-fixed-2"))
    elif mutation == "copy_name":
        man(lambda d: d.update(copy="code-fixed"))
    elif mutation == "copy_checksum":
        man(lambda d: d.update(copy_checksum="0" * 64))
    elif mutation == "copy_tree":
        man(lambda d: d.update(copy_tree_hash="0" * 64))
    elif mutation == "bench_command":
        man(lambda d: d.update(bench_command=[*d["bench_command"][:-1], "100"]))
    elif mutation == "no_start":
        man(lambda d: d.pop("start"))
    elif mutation == "end_before_start":
        man(lambda d: d.update(end="2026-10-20T00:00:00+05:30"))
    elif mutation == "command_copy":
        man(lambda d: d["command"].__setitem__(3, "code-fixed"))
    elif mutation == "command_run":
        man(lambda d: d["command"].__setitem__(5, "other-run"))
    elif mutation == "command_bench":
        man(lambda d: d["command"].__setitem__(-1, "100"))
    elif mutation == "command_script":
        man(lambda d: d["command"].__setitem__(1, "/tmp/other.py"))
    elif mutation == "command_python":
        man(lambda d: d["command"].__setitem__(0, "/tmp/evil-python"))
    elif mutation == "command_wrapper_elsewhere":
        man(lambda d: d["command"].__setitem__(1, "/tmp/scripts/run_manifest.py"))
    elif mutation == "command_copy_elsewhere":
        man(lambda d: d["command"].__setitem__(3, "/tmp/code-original"))
    elif mutation == "cwd":
        man(lambda d: d.update(cwd=str(PROJECT / "code-fixed")))
    elif mutation == "no_packages":
        man(lambda d: d.update(packages={}))
    elif mutation == "commit_unknown":
        man(lambda d: d.update(project_commit="0" * 40))
    elif mutation == "commit_old_wrapper":
        man(lambda d: d.update(project_commit="3beeaef1b61d2ef40e5581496e7d0c390bc28330"))
    elif mutation in ("cases", "repeats", "cap", "concurrency"):
        man(lambda d: d.update({mutation: d[mutation] - 1}))
    elif mutation == "manifest_cost":
        man(lambda d: d["summary"].update(cost_usd=1.0))
    elif mutation == "calls_deleted":
        (run / "calls.jsonl").unlink()
    elif mutation == "calls_returned_model":
        f = run / "calls.jsonl"
        rows = runlog.read_jsonl(f)
        rows[0]["returned_model"] = "gpt-4o-2024-08-06"
        f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif mutation == "coordinated_edit":
        d = json.loads(files[0].read_text())
        c = d["criteria"][0]
        c["ok"] = not c["ok"]
        d["actions_done"] += 1 if c["ok"] else -1
        d["solved"] = all(x["ok"] for x in d["criteria"])
        files[0].write_text(json.dumps(d))
        rpt(lambda r: next(x for x in r["runs"] if (x["id"], x["label"], x["rep"]) ==
                           (d["case"], d["arm"], d["repeat"])).update(
            k=d["actions_done"], solved=d["solved"]))
    elif mutation == "no_seal":
        rm.seal_path(run).unlink()


@pytest.mark.parametrize("mutation,reseal_after,why", [
    ("missing", True, "1 conversations missing"), ("duplicate", True, "appears twice"),
    ("foreign_case", True, "not in this run"), ("foreign_arm", True, "not in this run"),
    ("string_repeat", True, "not in this run"), ("not_json", True, "not JSON"),
    ("ok_is_int", True, "not true/false"), ("criterion_dropped", True, "differ from the suite"),
    ("no_criteria", True, "no criteria"),
    ("ok_flipped_only", True, "disagree with its criteria"),
    ("trace_solved_wrong", True, "disagree with its criteria"),
    ("stored_rule", True, "not exactly {id, ok, why}"),
    ("history_customer_error", True, "saved files no longer give"),
    ("report_row_k", True, "report.json rows disagree"),
    ("report_row_missing", True, "report.json rows disagree"),
    ("report_row_error", True, "report.json rows disagree"),
    ("report_repeats", True, "suite/cases/repeats/cap disagree"),
    ("case_text", True, "not the suite's first cases"),
    ("manifest_problems", True, "does not count"),
    ("manifest_no_summary", True, "does not count"),
    ("manifest_bench_exit", True, "manifest fields wrong"),
    ("manifest_copy_changed", True, "manifest fields wrong"),
    ("manifest_temperature", True, "manifest fields wrong"),
    ("manifest_cost", True, "saved files no longer give"),
    ("run_id", True, "manifest identity wrong"), ("copy_name", True, "manifest identity wrong"),
    ("copy_checksum", True, "manifest identity wrong"),
    ("copy_tree", True, "manifest identity wrong"),
    ("bench_command", True, "manifest identity wrong"),
    ("no_start", True, "provenance wrong: ['times']"),
    ("end_before_start", True, "provenance wrong: ['times']"),
    ("command_copy", True, "provenance wrong: ['command']"),
    ("command_run", True, "provenance wrong: ['command']"),
    ("command_bench", True, "provenance wrong: ['command']"),
    ("command_script", True, "provenance wrong: ['command']"),
    ("command_python", True, "provenance wrong: ['command']"),
    ("command_wrapper_elsewhere", True, "provenance wrong: ['command']"),
    ("command_copy_elsewhere", True, "provenance wrong: ['command']"),
    ("cwd", True, "provenance wrong: ['cwd']"),
    ("no_packages", True, "provenance wrong: ['versions']"),
    ("commit_unknown", True, "provenance wrong: ['wrapper']"),
    ("commit_old_wrapper", True, "provenance wrong: ['wrapper']"),
    ("cases", True, "manifest fields wrong"), ("repeats", True, "manifest fields wrong"),
    ("cap", True, "manifest fields wrong"), ("concurrency", True, "manifest fields wrong"),
    ("calls_deleted", True, "saved files"),
    ("calls_returned_model", True, "saved files no longer give"),
    ("coordinated_edit", False, "differ from the seal"),
    ("calls_deleted", False, "differ from the seal"),
    ("no_seal", False, "no seal")])
def test_anything_but_the_exact_sealed_clean_run_is_refused(tmp_path, mutation,
                                                            reseal_after, why):
    run = real_copy(tmp_path)
    (run / "score.json").write_text(json.dumps(sc.score_run(run)))
    _edit(run, mutation)
    if reseal_after:
        reseal(run)
    with pytest.raises(sc.Refused) as exc:
        sc.score_run(run)
    assert why in str(exc.value)
    assert not (run / "score.json").exists()


def test_a_file_changed_while_scoring_is_refused(tmp_path, monkeypatch):
    run = real_copy(tmp_path)
    load = sc.load

    def load_then_change(*a):
        rows = load(*a)
        f = next((run / "traces").glob("*.json"))
        f.write_text(f.read_text() + " ")
        return rows
    monkeypatch.setattr(sc, "load", load_then_change)
    with pytest.raises(sc.Refused, match="differ from the seal"):
        sc.score_run(run)


def test_a_file_changed_after_scoring_removes_the_score(tmp_path, monkeypatch, capsys):
    run = real_copy(tmp_path)
    score_run = sc.score_run

    def score_then_change(*a, **k):
        s = score_run(*a, **k)
        (run / "report.json").write_text((run / "report.json").read_text() + " ")
        return s
    monkeypatch.setattr(sc, "committed", lambda seal: None)
    monkeypatch.setattr(sc, "score_run", score_then_change)
    assert sc.main([str(run)]) == 1
    assert "changed while scoring" in capsys.readouterr().err
    assert not (run / "score.json").exists()


def test_cli_scores_a_committed_clean_run(tmp_path, monkeypatch):
    run = real_copy(tmp_path)
    monkeypatch.setattr(sc, "committed", lambda seal: None)
    assert sc.main([str(run)]) == 0
    assert json.loads((run / "score.json").read_text())["conversations"] == 240


def test_cli_refuses_a_seal_that_is_not_committed(tmp_path, capsys):
    run = real_copy(tmp_path)
    (run / "score.json").write_text("{}")
    assert sc.main([str(run)]) == 1
    assert "not committed" in capsys.readouterr().err
    assert not (run / "score.json").exists()


def test_suite_rule_that_disagrees_with_august_inference_is_refused(tmp_path, monkeypatch):
    suite = json.loads(sc.SUITE.read_text())
    suite[3]["success_criteria"][0]["rule"] = "tool_not_called"
    fake = tmp_path / "suite.json"
    fake.write_text(json.dumps(suite))
    monkeypatch.setattr(sc, "SUITE", fake)
    with pytest.raises(sc.Refused, match="case 3"):
        sc.suite_cases(24)
