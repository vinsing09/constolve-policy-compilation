"""Tests for scripts/run_deep.py: τ³'s real telecom harness, real base tasks, real grader;
only the model is faked (scripts/tests/fake_deep_run.py).

    .venv/bin/python -B -m pytest -q -p no:cacheprovider scripts/tests/test_run_deep.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
PROJECT = SCRIPTS.parent
sys.path.insert(0, str(SCRIPTS))
sys.dont_write_bytecode = True
import run_deep as rd  # noqa: E402
import run_manifest as rm  # noqa: E402
import runlog  # noqa: E402


def clean_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("OPENAI_", "LITELLM_", "TAU2_", "AZURE_"))}
    return {**env, **extra}


def fake_run(tmp, run, *args, version="v0", **env):
    counts = tmp / f"{run}.counts.json"
    p = subprocess.run(
        [sys.executable, "-B", str(SCRIPTS / "tests" / "fake_deep_run.py"), "--version",
         version, "--run", run, "--mode", "test", "--results",
         str(tmp / "results"), *args],
        env=clean_env(FAKE_COUNTS=str(counts), **env), capture_output=True, text=True,
        timeout=600)
    m = tmp / "results" / run / "manifest.json"
    return p, (json.loads(counts.read_text()) if counts.exists() else None), \
        (json.loads(m.read_text()) if m.exists() else None)


@pytest.fixture(scope="module")
def both(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("deep")
    out = {}
    for v in ("v0", "v2_deep"):
        p, counts, m = fake_run(tmp, f"good-{v}", version=v)
        assert p.returncode == 0, p.stderr[-3000:]
        out[v] = (tmp / "results" / f"good-{v}", counts, m)
    return out


def sims(run_dir):
    return json.loads((run_dir / "tau3" / "results.json").read_text())["simulations"]


@pytest.mark.parametrize("v", ["v0", "v2_deep"])
def test_good_run_ran_the_intended_policy_and_logged_every_call(both, v):
    run_dir, counts, m = both[v]
    s = m["summary"]
    assert s["problems"] == [] and s["copy_unchanged"]
    assert s["simulations"] == s["expected"] == 4
    assert s["calls_by_caller"] == counts and s["untagged_calls"] == 0
    name, sha = rd.POLICIES[v]
    text = (PROJECT / "deep_policy" / name).read_bytes()
    assert hashlib.sha256(text).hexdigest() == sha == m["policy_sha256"]
    assert {x["policy"] for x in sims(run_dir)} == {text.decode()}
    agent_calls = [c for c in runlog.read_jsonl(run_dir / "calls.jsonl") if c["caller"] == "agent"]
    assert agent_calls and all(c["temperature"] == 0.0 and c["num_retries"] == 0
                               for c in agent_calls)
    assert m["config"]["llm_agent"] == m["config"]["llm_user"] == "gpt-4.1"
    assert m["config"]["llm_args_user"] == {"temperature": 0.0, "num_retries": 0}
    assert m["config"]["max_retries"] == 0 and m["config"]["max_steps"] == 200
    assert rm.seal_path(run_dir).read_text() == rm.seal(run_dir)


def test_both_versions_share_the_exact_task_trial_seed_keys(both):
    keys = {v: sorted((x["task_id"], x["trial"], x["seed"]) for x in sims(r))
            for v, (r, _, _) in both.items()}
    assert keys["v0"] == keys["v2_deep"]
    assert keys["v0"] == sorted(map(tuple, both["v0"][2]["expected_keys"]))
    assert {k[0] for k in keys["v0"]} == set(rd.base_ids()[:2])


@pytest.mark.parametrize("v", ["v0", "v2_deep"])
def test_the_policy_file_is_what_the_agent_prompt_carries(tmp_path, v):
    prompts = tmp_path / "prompts.json"
    p, counts, m = fake_run(tmp_path, f"prompt-{v}", version=v, FAKE_PROMPTS=str(prompts))
    assert p.returncode == 0, p.stderr[-2000:]
    assert json.loads(prompts.read_text()) == [rd.POLICIES[v][1]]


def test_v0_file_is_taus_own_telecom_policy():
    """So for v0 a missing patch would run the same text; the patch only matters for v2."""
    sys.path.insert(0, str(rd.TAU3 / "src"))
    from tau2.domains.telecom.environment import get_environment
    built_in = get_environment().policy
    assert built_in.encode() == (PROJECT / "deep_policy" / rd.POLICIES["v0"][0]).read_bytes()
    assert len(built_in) == 23318


def test_a_missing_policy_patch_is_caught_on_the_deep_version(tmp_path):
    p, counts, m = fake_run(tmp_path, "nopatch", version="v2_deep", FAKE_MUTATE="no_patch")
    assert p.returncode == 1 and "wrong_policy: 4" in m["summary"]["problems"]


@pytest.mark.parametrize("mutate,problem", [
    ("fail_call", "infrastructure_errors: 1"),
    ("wrong_model", "returned models"), ("undated", "returned models"),
    ("untagged", "untagged_calls"),
    ("no_usage", "calls_missing_usage_or_cost")])
def test_broken_runs_do_not_count(tmp_path, mutate, problem):
    p, counts, m = fake_run(tmp_path, mutate, FAKE_MUTATE=mutate)
    assert p.returncode == 1, p.stderr[-3000:]
    assert any(problem in x for x in m["summary"]["problems"]), m["summary"]["problems"]


def test_a_simulation_missing_from_the_results_is_caught(both, tmp_path):
    src, _, m = both["v0"]
    run = tmp_path / "r"
    shutil.copytree(src, run)
    f = run / "tau3" / "results.json"
    d = json.loads(f.read_text())
    d["simulations"].pop()
    f.write_text(json.dumps(d))
    s = rd.check(run, rd.base_ids()[:2], 2, m["policy_sha256"])
    assert "simulations are not exactly the expected (task, trial, seed) set" in s["problems"]


def _ids(f):
    return f.read_text().split()


@pytest.mark.parametrize("hazard,why", [
    ("policy_byte", "does not hash"), ("other_policy_byte", "does not hash"),
    ("other_policy_missing", "does not hash"), ("run_exists", "already exists"),
    ("shell_tau2", "shell sets TAU2_DATA_DIR"), ("subset_foreign", "not 51 distinct"),
    ("subset_duplicate", "not 51 distinct"), ("subset_empty", "not 62 distinct"),
    ("subset_overlap", "subsets do not split"), ("august_truncated", "3 + 2 August")])
def test_preflight_refuses(tmp_path, hazard, why):
    pol = tmp_path / "deep_policy"
    shutil.copytree(PROJECT / "deep_policy", pol)
    env, args, run = {}, ["--policy-dir", str(pol)], "x"
    un, fed, seen = (pol / f"{s}_task_ids.txt" for s in rd.SUBSETS)
    if hazard in ("policy_byte", "other_policy_byte"):
        f = pol / rd.POLICIES["v0" if hazard == "policy_byte" else "v2_deep"][0]
        f.write_bytes(f.read_bytes() + b"\n")
    elif hazard == "other_policy_missing":
        (pol / rd.POLICIES["v2_deep"][0]).unlink()
    elif hazard == "run_exists":
        (tmp_path / "results" / "x").mkdir(parents=True)
    elif hazard == "shell_tau2":
        env["TAU2_DATA_DIR"] = str(tmp_path)
    elif hazard == "subset_foreign":
        un.write_text("\n".join(_ids(un)[:-1] + ["not_a_task"]) + "\n")
    elif hazard == "subset_duplicate":
        un.write_text("\n".join(_ids(un)[:-1] + _ids(un)[:1]) + "\n")
    elif hazard == "subset_empty":
        fed.write_text("")
    elif hazard == "subset_overlap":
        moved = _ids(seen)[0]
        un.write_text("\n".join(_ids(un)[:-1] + [moved]) + "\n")
        seen.write_text("\n".join(_ids(seen)[1:] + [_ids(un)[-1]]) + "\n")
    elif hazard == "august_truncated":
        f = pol / "august_results_per_task.json"
        d = json.loads(f.read_text())
        d["v0"] = d["v0"][:1]
        f.write_text(json.dumps(d))
    before = runlog.tree_hash(rd.COPY)
    p, counts, m = fake_run(tmp_path, run, *args, **env)
    assert p.returncode == 2 and why in p.stderr, p.stderr[-1500:]
    assert counts == {"agent": 0, "user": 0} and runlog.tree_hash(rd.COPY) == before


def test_expected_seeds_are_taus_own():
    """runner/batch.py:511-512 with seed 300: what both versions will get."""
    import random
    random.seed(300)
    seeds = [random.randint(0, 1000000) for _ in range(5)]
    keys = rd.expected_keys(["a"], 5)
    assert [k[2] for k in keys] == seeds and [k[1] for k in keys] == list(range(5))


@pytest.mark.parametrize("v", ["v0", "v2_deep"])
def test_every_agent_prompt_carried_the_intended_policy(both, v):
    run_dir, _, m = both[v]
    agent = [c for c in runlog.read_jsonl(run_dir / "calls.jsonl") if c["caller"] == "agent"]
    assert {c["prompt_policy_sha256"] for c in agent} == {rd.POLICIES[v][1]}
    s = m["summary"]
    assert s["agent_prompts_wrong_policy"] == s["calls_with_retries_on"] == 0
    assert s["sims_calls_not_matching_messages"] == 0 and s["sdk_retries"] == 0
    assert set(m["inputs_sha256"]) == set(rd.INPUTS)


@pytest.mark.parametrize("edit,problem", [
    ("retries_on", "calls_with_retries_on: 1"), ("sdk_retry", "sdk_retries: 1"),
    ("prompt_policy", "agent_prompts_wrong_policy: 1"),
    ("call_dropped", "sims_calls_not_matching_messages: 1"),
    ("other_caller", "neither the agent nor the customer"),
    ("swap_sims", "sims_calls_not_matching_messages: 2"),
    ("raw_replaced", "sims_calls_not_matching_messages: 1"),
    ("tokens", "sims_calls_not_matching_messages: 1"),
    ("no_agent_calls", "sims_without_agent_call: 1"),
    ("temperature", "temperature 0"), ("seed", "seeds_ok=False")])
def test_one_bad_call_row_makes_the_run_not_count(both, tmp_path, edit, problem):
    src, _, m = both["v2_deep"]
    run = tmp_path / "r"
    shutil.copytree(src, run)
    f = run / "calls.jsonl"
    rows = runlog.read_jsonl(f)
    i = next(n for n, c in enumerate(rows) if c["caller"] == "agent")
    if edit == "call_dropped":
        rows.pop(i)
    elif edit == "swap_sims":
        j = next(n for n, c in enumerate(rows) if c["caller"] == "agent" and
                 c["sim"][1] == rows[i]["sim"][1] and c["sim"][0] != rows[i]["sim"][0])
        rows[i]["sim"], rows[j]["sim"] = rows[j]["sim"], rows[i]["sim"]
    elif edit == "raw_replaced":
        res = run / "tau3" / "results.json"
        d = json.loads(res.read_text())
        msgs = [m for m in d["simulations"][0]["messages"] if m.get("raw_data")
                and m["role"] == "assistant"]
        other = next(m for x in d["simulations"][1:] for m in x["messages"]
                     if m.get("raw_data") and m["role"] == "assistant")
        msgs[0]["raw_data"] = other["raw_data"]
        res.write_text(json.dumps(d))
    elif edit == "tokens":
        rows[i]["prompt_tokens"] += 1
    elif edit == "no_agent_calls":
        sim = rows[i]["sim"]
        rows = [c for c in rows if not (c["sim"] == sim and c["caller"] == "agent")]
        res = run / "tau3" / "results.json"
        d = json.loads(res.read_text())
        x = next(x for x in d["simulations"] if [x["task_id"], x["seed"]] == sim)
        for msg in x["messages"]:
            if msg["role"] == "assistant":
                msg["raw_data"] = None
        res.write_text(json.dumps(d))
    else:
        rows[i].update({"retries_on": {"num_retries": 3}, "sdk_retry": {"sdk_retries": 1},
                        "prompt_policy": {"prompt_policy_sha256": rd.POLICIES["v0"][1]},
                        "other_caller": {"caller": "other"},
                        "temperature": {"temperature": 0.7},
                        "seed": {"seed": 1}}[edit])
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    s = rd.check(run, rd.base_ids()[:2], 2, m["policy_sha256"])
    assert any(problem in x for x in s["problems"]), s["problems"]


def test_provenance_record_still_matches_upstream():
    import tau3_provenance
    assert tau3_provenance.main(["--check"]) == 0
    prov = json.loads(rd.PROVENANCE.read_text())
    assert list(prov["differ"]) == ["src/tau2/utils/llm_utils.py"]
    assert prov["not_in_upstream"] == ["VENDOR_SHA"]


@pytest.mark.parametrize("field,value", [
    ("runtime_subset_complete", False), ("mode_differ", {"x": {}}),
    ("differ", {"src/tau2/utils/llm_utils.py": {}, "y": {}}), ("upstream_commit", "0" * 40),
    ("not_in_upstream", ["VENDOR_SHA", "z"]), ("patch_sha256", "0" * 64)])
def test_the_provenance_record_must_say_upstream_plus_one_patch(field, value):
    import tau3_provenance as tp
    prov, patch = json.loads(rd.PROVENANCE.read_text()), tp.PATCH.read_bytes()
    assert rd.record_ok(prov, patch)
    assert not rd.record_ok({**prov, field: value}, patch)


def test_a_stale_or_uncommitted_provenance_record_is_refused(tmp_path, monkeypatch):
    fake = tmp_path / "tau3-provenance.json"
    d = json.loads(rd.PROVENANCE.read_text())
    d["vendor_tree_hash"] = "0" * 64
    fake.write_text(json.dumps(d))
    monkeypatch.setattr(rd, "PROVENANCE", fake)
    assert rd.provenance_problems() == [
        "scripts/tau3-provenance.json or tau3_vendor.patch is not committed or does not say "
        "upstream + one patch with a complete runtime subset"]
    monkeypatch.setattr(rd, "committed", lambda f: True)
    assert rd.provenance_problems() == [
        "vendored τ³ no longer matches the committed provenance record"]
    monkeypatch.setattr(rd, "PROVENANCE", PROJECT / "scripts" / "tau3-provenance.json")
    assert rd.provenance_problems() == []


@pytest.mark.parametrize("results,run,why", [
    (rd.COPY / "results-here", "x", "inside code-original"),
    (rd.COPY, "x", "inside code-original"),
    (PROJECT / "results", "../code-original/x", "not a single safe name"),
    (PROJECT / "results", "/tmp/x", "not a single safe name")])
def test_output_inside_the_copy_or_an_unsafe_name_is_refused(results, run, why):
    """Checked in-process on preflight only, so a broken check cannot write into the copy."""
    a = type("A", (), {"run": run, "policy_dir": PROJECT / "deep_policy"})()
    bad = rd.preflight(a, (results / run).resolve(), {})
    assert any(why in b for b in bad), bad


def test_a_seal_that_cannot_be_written_makes_the_run_not_count(tmp_path):
    p, counts, m = fake_run(tmp_path, "nosel", FAKE_MUTATE="seal_fail")
    assert p.returncode == 1
    assert any(x.startswith("seal not written: OSError: disk full")
               for x in m["summary"]["problems"]), m["summary"]["problems"]
    assert not (tmp_path / "seals" / "nosel.sha256").exists()


def test_a_seals_path_that_is_a_file_is_refused_before_the_run(tmp_path):
    (tmp_path / "seals").write_text("not a folder")
    a = type("A", (), {"run": "x", "policy_dir": PROJECT / "deep_policy"})()
    bad = rd.preflight(a, tmp_path / "results" / "x", {})
    assert any("exists and is not a folder" in b for b in bad), bad


def test_the_provenance_record_says_the_runtime_subset_is_complete():
    prov = json.loads(rd.PROVENANCE.read_text())
    assert prov["runtime_subset_complete"] is True and prov["mode_differ"] == {}
    assert prov["upstream_paths"] == 1424 and prov["vendor_paths"] == 1233
    assert len(prov["omitted_from_vendor"]) == 192
    assert not [x for x in prov["omitted_from_vendor"] if x.startswith(("src/", "data/tau2/domains/"))]


def test_an_interrupted_seal_leaves_a_run_that_does_not_count(tmp_path):
    p, counts, m = fake_run(tmp_path, "intr", FAKE_MUTATE="seal_interrupt")
    assert p.returncode != 0
    assert any(x.startswith("seal not written: KeyboardInterrupt")
               for x in m["summary"]["problems"]), m["summary"]["problems"]
    assert not (tmp_path / "seals" / "intr.sha256").exists()


def test_preflight_runs_the_provenance_check(tmp_path, monkeypatch):
    monkeypatch.setattr(rd, "provenance_problems", lambda: ["provenance says no"])
    a = type("A", (), {"run": "x", "policy_dir": PROJECT / "deep_policy"})()
    assert rd.preflight(a, tmp_path / "results" / "x", {}) == ["provenance says no"]


def test_the_provenance_check_requires_the_record_fields(monkeypatch):
    monkeypatch.setattr(rd, "record_ok", lambda prov, patch: False)
    assert rd.provenance_problems() == [
        "scripts/tau3-provenance.json or tau3_vendor.patch is not committed or does not say "
        "upstream + one patch with a complete runtime subset"]
