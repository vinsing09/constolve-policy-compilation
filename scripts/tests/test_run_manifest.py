"""Tests for scripts/run_manifest.py and runlog.py on a full copy of code-original.

    .venv/bin/python -m pytest -q -p no:cacheprovider scripts/tests
"""
import asyncio
import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import run_manifest as rm  # noqa: E402
import runlog  # noqa: E402

PROJECT = SCRIPTS.parent
BENCH = ["--suite", "ours", "--domain", "telecom", "--cases", "1", "--repeats", "1",
         "--max-turns", "3"]
B = rm.bench_args(BENCH)


@pytest.fixture(scope="module")
def copy(tmp_path_factory):
    dst = tmp_path_factory.mktemp("c") / "code-original"
    shutil.copytree(PROJECT / "code-original", dst, symlinks=True)
    yield dst
    shutil.rmtree(dst)          # the copy holds .env


def clean_env():
    return {k: v for k, v in os.environ.items()
            if not (k in rm.EXACT or k.startswith(rm.PREFIXES))}


def start(copy, results, run, bench=BENCH, **env):
    return subprocess.Popen(
        [sys.executable, str(SCRIPTS / "tests" / "fake_llm_run.py"), "--copy", str(copy),
         "--run", run, "--results", str(results), "--", *bench],
        env={**clean_env(), "FAKE_COUNTS": str(results / f"{run}.counts.json"), **env},
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def finish(proc, results, run):
    out, err = proc.communicate(timeout=300)
    p = types.SimpleNamespace(returncode=proc.returncode, stdout=out, stderr=err)
    counts, m = results / f"{run}.counts.json", results / run / "manifest.json"
    return p, (json.loads(counts.read_text()) if counts.exists() else None), \
        (json.loads(m.read_text()) if m.exists() else None)


def fake_run(copy, results, run, bench=BENCH, **env):
    return finish(start(copy, results, run, bench, **env), results, run)


@pytest.fixture(scope="module")
def good_run(copy, tmp_path_factory):
    results = tmp_path_factory.mktemp("results")
    p, counts, m = fake_run(copy, results, "r1")
    assert p.returncode == 0, p.stderr[-3000:]
    return results / "r1", counts, m


def test_clean_copy_passes_preflight_in_any_locale(copy, tmp_path, monkeypatch):
    monkeypatch.setenv("LC_ALL", "C.UTF-8")
    assert rm.preflight(copy, tmp_path / "new", B, clean_env()) == []


@pytest.mark.parametrize("hazard,why", [
    ("trace", "logs/bench exists"), ("report", "exists:"),
    ("shell_model", "shell sets USER_SIM_MODEL"), ("shell_retries", "DEFAULT_MAX_RETRIES"),
    ("edited", "does not match checksum"), ("env_file", "does not match tree"),
    ("empty_dir", "does not match tree"), ("symlink", "does not match tree"),
    ("run_dir", "already exists"), ("tau3", "only --suite ours")])
def test_preflight_refuses(copy, tmp_path, hazard, why):
    b, environ, run_dir = B, clean_env(), tmp_path / "new"
    traces, report = rm.outputs(copy, b)
    undo = []
    if hazard == "trace":
        traces.mkdir()
        undo.append(traces.rmdir)
    elif hazard in ("report", "edited"):
        f = report if hazard == "report" else copy / "services" / "extra.py"
        f.write_text("{}")
        undo.append(f.unlink)
    elif hazard == "env_file":
        env = copy / ".env"
        old = env.read_bytes()
        env.write_bytes(old + b"\nDEFAULT_MAX_RETRIES=0\n")
        undo.append(lambda: env.write_bytes(old))
    elif hazard == "empty_dir":
        d = copy / "services" / "newdir"
        d.mkdir()
        undo.append(d.rmdir)
    elif hazard == "symlink":
        link = copy / "third_party" / "tau-bench"
        old = os.readlink(link)
        link.unlink()
        link.symlink_to("/elsewhere")
        undo.append(lambda: (link.unlink(), link.symlink_to(old)))
    elif hazard.startswith("shell"):
        environ["USER_SIM_MODEL" if hazard == "shell_model" else "DEFAULT_MAX_RETRIES"] = "x"
    elif hazard == "tau3":
        b = rm.bench_args(["--suite", "tau3"])
    else:
        run_dir.mkdir()
    try:
        assert any(why in x for x in rm.preflight(copy, run_dir, b, environ))
    finally:
        for u in undo:
            u()
    assert rm.preflight(copy, tmp_path / "new2", B, clean_env()) == []


def test_good_run_logs_every_call_per_conversation(good_run):
    run_dir, counts, m = good_run
    s = m["summary"]
    assert s["problems"] == [] and s["copy_unchanged"]
    assert s["trace_files"] == s["report_rows"] == 2
    assert m["models_resolved"] == {"agent": "gpt-4.1", "customer": "gpt-4.1",
                                    "customer_temperature": 0.7}
    calls = runlog.read_jsonl(run_dir / "calls.jsonl")
    assert s["calls_by_caller"] == counts
    assert counts["agent"] == 2 * 3 * 2      # conversations x capped turns x (tool, reply)
    assert {c["temperature"] for c in calls if c["caller"] == "agent"} == {0}
    assert {c["temperature"] for c in calls if c["caller"] == "customer"} == {0.7}
    assert {c["conv"].split("|")[0] for c in calls} == {"with_policy", "stub"}
    assert all(c["returned_model"] == "gpt-4.1" for c in calls)
    assert all(c["prompt_tokens"] > 0 and c["completion_tokens"] > 0 and c["cost_usd"] > 0
               for c in calls)
    assert s["calls_missing_usage_or_cost"] == 0
    assert s["customer_reply_shapes"] == {"ok": counts["customer"]}
    assert s["customer_response_formats"] == ["null"]
    assert s["cost_usd"] == round(sum(c["cost_usd"] for c in calls), 4) > 0
    d = json.loads(next((run_dir / "traces").iterdir()).read_text())
    assert d["history"][0]["content"] == "Hi, my phone has no service."
    assert "get_customer_by_phone" in d["calls"] and len(d["criteria"]) == 6


def test_good_run_is_sealed_and_scores(good_run):
    run_dir = good_run[0]
    seal = rm.seal_path(run_dir)
    assert seal == run_dir.parent.parent / "seals" / "r1.sha256"
    assert seal.read_text() == rm.seal(run_dir)
    names = {line.split("  ./", 1)[1] for line in seal.read_text().splitlines()}
    assert {"manifest.json", "report.json", "calls.jsonl", "stdout.log"} <= names
    assert sum(n.startswith("traces/") for n in names) == 2
    import score
    s = score.score_run(run_dir, job1=False)
    assert s["conversations"] == 2 and set(s["pooled"]) == {"with_policy", "stub", "gap"}
    with pytest.raises(score.Refused, match="manifest fields wrong"):
        score.score_run(run_dir)


def test_good_run_leaves_copy_as_it_was(good_run, copy):
    assert not rm.outputs(copy, B)[0].exists() and not rm.outputs(copy, B)[1].exists()
    assert runlog.tree_hash(copy) == rm.stored("tree", copy)


def test_same_run_name_is_refused(good_run, copy):
    p, _, _ = fake_run(copy, good_run[0].parent, "r1")
    assert p.returncode == 2 and "already exists" in p.stderr


def test_second_run_on_a_locked_copy_is_refused(copy, tmp_path):
    held = rm.lock(copy)
    try:
        p, counts, m = fake_run(copy, tmp_path, "r2")
    finally:
        held.close()
    assert p.returncode == 2 and "holds the lock" in p.stderr and m is None


@pytest.mark.parametrize("env,problem", [
    ({"FAKE_CUSTOMER_BAD_FROM": "3"}, "customer_errors"),
    ({"FAKE_CUSTOMER_BAD_FROM": "1"}, "customer_errors"),
    ({"FAKE_AGENT_RAISE": "1"}, "report_errors"),
    ({"FAKE_RETURNED": "gpt-4.1-mini"}, "returned models"),
    ({"FAKE_NO_USAGE": "1"}, "calls_missing_usage_or_cost"),
    ({"FAKE_COST_RAISE": "1"}, "calls_missing_usage_or_cost"),
])
def test_failures_make_the_run_not_count(copy, tmp_path, env, problem):
    p, counts, m = fake_run(copy, tmp_path, "bad", **env)
    s = m["summary"]
    assert p.returncode == 1 and any(problem in x for x in s["problems"]), s
    assert s["copy_unchanged"]


def test_later_turn_customer_failure_is_invisible_in_transcript_but_counted(copy, tmp_path):
    _, _, m = fake_run(copy, tmp_path, "late", FAKE_CUSTOMER_BAD_FROM="3")
    s = m["summary"]
    assert s["customer_errors"] >= 1
    assert s["customer_errors_visible_in_transcript"] < s["conversations_with_customer_error"]


def stage(src, root, files=None):
    """A good run's outputs put back where bench.py writes them, for collect()."""
    fake_copy, run_dir = root / "copy", root / "run"
    traces, report = rm.outputs(fake_copy, B)
    traces.mkdir(parents=True)
    report.parent.mkdir(parents=True)
    run_dir.mkdir()
    shutil.copy(src / "report.json", report)
    shutil.copy(src / "calls.jsonl", run_dir / "calls.jsonl")
    for f in files if files is not None else sorted((src / "traces").iterdir()):
        shutil.copy(f, traces / f.name)
    return fake_copy, run_dir, report


def good_case(src):
    return [json.loads(next((src / "traces").iterdir()).read_text())["case"]]


def test_restaged_good_run_has_no_problems(good_run, tmp_path):
    fake_copy, run_dir, _ = stage(good_run[0], tmp_path)
    assert rm.collect(fake_copy, run_dir, B, good_case(good_run[0]))["problems"] == []


def test_wrong_or_partial_conversation_sets_are_caught(good_run, tmp_path):
    src = good_run[0]
    files = sorted((src / "traces").iterdir())
    for case, ids in (("missing", None), ("duplicate", None), ("foreign", ["99:other"])):
        fake_copy, run_dir, _ = stage(src, tmp_path / case,
                                      files[:1] if case == "missing" else files)
        if case == "duplicate":
            shutil.copy(files[0], rm.outputs(fake_copy, B)[0] / ("again_" + files[0].name))
        s = rm.collect(fake_copy, run_dir, B, ids or good_case(src))
        assert "trace files are not exactly the expected conversations" in s["problems"]


def _calls(run_dir, edit):
    f = run_dir / "calls.jsonl"
    rows = [r for r in map(edit, runlog.read_jsonl(f)) if r is not None]
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.mark.parametrize("mutation,problem", [
    ("report_foreign", "report rows are not exactly"),
    ("report_duplicate", "report rows are not exactly"),
    ("requested_model", "requested models"),
    ("other_caller", "neither the agent nor the customer"),
    ("no_customer_calls", "customer calls missing for 1 "),
    ("no_agent_calls", "agent calls missing for 1 "),
])
def test_call_and_report_mismatches_are_caught(good_run, tmp_path, mutation, problem):
    fake_copy, run_dir, report = stage(good_run[0], tmp_path)
    first = runlog.read_jsonl(run_dir / "calls.jsonl")[0]["conv"]
    if mutation.startswith("report"):
        d = json.loads(report.read_text())
        if mutation == "report_foreign":
            d["runs"][0]["id"] = "99:other"
        else:
            d["runs"][1] = d["runs"][0]
        report.write_text(json.dumps(d))
    elif mutation == "requested_model":
        _calls(run_dir, lambda r: {**r, "requested_model": "gpt-4o"} if r["conv"] == first else r)
    elif mutation == "other_caller":
        _calls(run_dir, lambda r: {**r, "caller": "other"})
    else:
        who = mutation.split("_")[1]
        _calls(run_dir, lambda r: None if r["conv"] == first and r["caller"] == who else r)
    s = rm.collect(fake_copy, run_dir, B, good_case(good_run[0]))
    assert any(problem in x for x in s["problems"]), s["problems"]


@pytest.mark.parametrize("field,bad", [
    ("prompt_tokens", 0), ("prompt_tokens", None), ("prompt_tokens", 12.0),
    ("completion_tokens", 0), ("completion_tokens", None),
    ("cost_usd", 0), ("cost_usd", 0.0), ("cost_usd", None), ("cost_usd", "0.01")])
def test_one_bad_usage_or_cost_field_alone_makes_the_run_not_count(good_run, tmp_path,
                                                                  field, bad):
    fake_copy, run_dir, _ = stage(good_run[0], tmp_path)
    first = runlog.read_jsonl(run_dir / "calls.jsonl")[0]
    assert all(rm.usable(r) for r in runlog.read_jsonl(run_dir / "calls.jsonl"))
    _calls(run_dir, lambda r: {**r, field: bad} if r == first else r)
    s = rm.collect(fake_copy, run_dir, B, good_case(good_run[0]))
    assert s["calls_missing_usage_or_cost"] == 1
    assert s["problems"] == ["calls_missing_usage_or_cost: 1"], s["problems"]


def test_retried_timeout_still_makes_the_run_not_count(copy, tmp_path):
    """journey_runner retries a timeout once; the conversation then looks normal."""
    p, counts, m = fake_run(copy, tmp_path, "to", FAKE_AGENT_TIMEOUT_ONCE="1")
    s = m["summary"]
    calls = runlog.read_jsonl(tmp_path / "to" / "calls.jsonl")
    assert p.returncode == 1 and s["problems"] == ["failed_calls: 1"], s
    assert s["report_errors"] == 0 and s["conversations_with_no_criteria"] == 0
    assert len(calls) == counts["agent"] + counts["customer"]
    [bad] = [c for c in calls if not c["ok"]]
    assert bad["caller"] == "agent" and bad["error"].startswith("Timeout")


def test_more_cases_than_the_suite_has_is_refused_before_any_call(copy, tmp_path):
    bench = [x if x != "1" else "25" for x in BENCH[:6]] + BENCH[6:]
    p, counts, m = fake_run(copy, tmp_path, "c25", bench)
    assert p.returncode != 0 and counts == {"agent": 0, "customer": 0}
    assert "cases: asked 25, ran 24" in m["summary"]["problems"]
    assert len(m["case_ids"]) == 24 and m["cases"] == 25


def test_setup_failure_outside_bench_try_makes_the_run_not_count(copy, tmp_path):
    p, _, m = fake_run(copy, tmp_path, "setup", FAKE_MUTATE="setup_raise")
    assert p.returncode != 0 and "bench exit None" in m["summary"]["problems"]
    assert m["summary"]["copy_unchanged"]


def test_two_runs_started_together_on_one_copy(copy, tmp_path):
    procs = [start(copy, tmp_path, r, FAKE_DELAY="0.5") for r in ("a", "b")]
    done = [finish(pr, tmp_path, r) for pr, r in zip(procs, ("a", "b"))]
    codes = sorted(pr.returncode for pr, _, _ in done)
    assert codes == [0, 2], [pr.stderr[-500:] for pr, _, _ in done]
    refused = next(pr for pr, _, _ in done if pr.returncode == 2)
    assert "holds the lock" in refused.stderr


@pytest.mark.parametrize("mutation", [None, "one_round", "no_results", "blank_payload",
                                      "no_actor"])
def test_customer_action_is_observed_and_graded_as_the_customer(copy, tmp_path, mutation):
    """Case 0's s0 criterion needs the customer to toggle airplane mode itself."""
    result = "Airplane mode is now OFF."          # generated/env_telecom.py:1517
    env = {"FAKE_CUSTOMER_ACT": "toggle_airplane_mode", "FAKE_CUSTOMER_EXPECT": result}
    if mutation:
        env["FAKE_MUTATE"] = mutation
    p, counts, m = fake_run(copy, tmp_path, "act", **env)
    traces = [json.loads(f.read_text()) for f in (tmp_path / "act" / "traces").iterdir()]
    acts = [tc for d in traces for t in d["history"] if t.get("role") == "user"
            for tc in t.get("tool_calls") or []]
    graded = [{c["id"]: c["ok"] for c in d["criteria"]}["s0_toggle_airplane_mode"]
              for d in traces]
    assert p.returncode == 0 and m["summary"]["problems"] == []
    assert len(acts) == 2 and {tc["tool_name"] for tc in acts} == {"toggle_airplane_mode"}
    # each conversation's opening customer turn ends with the reply built from the result
    shown_result = counts["saw_result"] >= 2 and \
        [d["history"][0]["content"] for d in traces] == [f"My phone now says: {result}"] * 2
    attributed = all(tc.get("actor") == "user" for tc in acts) and all(graded)
    assert (shown_result, attributed) == {
        None: (True, True), "one_round": (False, True),
        "no_results": (False, True), "blank_payload": (False, True),
        "no_actor": (True, False)}[mutation]


def test_show_turns_reads_a_real_run_directory(good_run):
    run_dir = str(good_run[0])
    out = subprocess.run([sys.executable, str(SCRIPTS / "show_turns.py"), run_dir, run_dir,
                          "--turns", "2", "--width", "100"], capture_output=True, text=True, check=True).stdout
    assert out.count("CUSTOMER: Hi, my phone has no service.") == 2
    assert "[did: get_customer_by_phone (failed)]" in out
    n = len(runlog.read_jsonl(good_run[0] / "calls.jsonl"))
    assert f"r1: {n} model calls, $" in out


class _Flaky(http.server.BaseHTTPRequestHandler):
    hits = 0

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        type(self).hits += 1
        if self.hits == 1:
            self.send_response(429)
            self.send_header("retry-after-ms", "10")
            body = b'{"error": {"message": "slow down"}}'
        else:
            self.send_response(200)
            body = json.dumps({
                "id": "x", "object": "chat.completion", "created": 1,
                "model": "gpt-4.1-2025-04-14",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": "hi"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1,
                          "total_tokens": 6}}).encode()
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_sdk_retry_and_provider_model_are_recorded(tmp_path):
    """A real OpenAI-shaped HTTP exchange: 429, then success with a dated model name."""
    import litellm
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Flaky)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    class Mod:
        pass
    bench, us = Mod(), Mod()
    bench.one = None
    us.UserSim = type("U", (), {"reply": None})
    saved = litellm.acompletion
    try:
        runlog.Recorder(tmp_path / "c.jsonl", tmp_path / "e.jsonl").install(bench, us, litellm)
        asyncio.run(litellm.acompletion(
            model="gpt-4.1", messages=[{"role": "user", "content": "x"}], temperature=0,
            api_base=f"http://127.0.0.1:{srv.server_port}/v1", api_key="test"))
    finally:
        litellm.acompletion = saved
        srv.shutdown()
    [row] = runlog.read_jsonl(tmp_path / "c.jsonl")
    assert (row["ok"], row["sdk_retries"], row["returned_model"]) == \
        (True, 1, "gpt-4.1-2025-04-14")
    assert runlog.RETURNED_OK.match(row["returned_model"])


@pytest.fixture(scope="module")
def copy_fixed(tmp_path_factory):
    dst = tmp_path_factory.mktemp("cf") / "code-fixed"
    shutil.copytree(PROJECT / "code-fixed", dst, symlinks=True)
    yield dst
    shutil.rmtree(dst)          # the copy holds .env


@pytest.mark.parametrize("which,want", [("original", None),
                                        ("fixed", {"type": "json_object"})])
def test_only_the_fixed_customer_asks_for_json_mode(copy, copy_fixed, tmp_path, which, want):
    c = copy if which == "original" else copy_fixed
    formats = tmp_path / "formats.jsonl"
    p, counts, m = fake_run(c, tmp_path, "jm", FAKE_FORMATS=str(formats))
    assert p.returncode == 0 and m["summary"]["problems"] == [], p.stderr[-2000:]
    rows = [json.loads(x) for x in formats.read_text().splitlines()]
    assert len(rows) == counts["agent"] + counts["customer"]
    assert {json.dumps(f) for w, f in rows if w == "customer"} == {json.dumps(want)}
    assert {json.dumps(f) for w, f in rows if w == "agent"} == {"null"}
    assert m["summary"]["customer_response_formats"] == [json.dumps(want)]
    assert m["summary"]["customer_reply_shapes"] == {"ok": counts["customer"]}


def test_fixed_copy_is_the_original_plus_the_two_patches(tmp_path):
    out = subprocess.run(["diff", "-rq", "code-original", "code-fixed"], cwd=PROJECT,
                         capture_output=True, text=True).stdout
    assert [x for x in out.splitlines() if "No such file" not in x] == \
        ["Files code-original/services/user_sim.py and code-fixed/services/user_sim.py differ"]
    (tmp_path / "services").mkdir()
    shutil.copy(PROJECT / "code-original/services/user_sim.py", tmp_path / "services")
    git = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True,  # noqa: E731
                                    stdin=subprocess.DEVNULL, capture_output=True)
    git("init", "-q")
    git("apply", "--include=services/user_sim.py", str(SCRIPTS / "role_fix.patch"))
    git("apply", str(SCRIPTS / "json_mode.patch"))
    assert (tmp_path / "services/user_sim.py").read_bytes() == \
        (PROJECT / "code-fixed/services/user_sim.py").read_bytes()


@pytest.mark.parametrize("reply,tools,shape", [
    ('{"message": "Hi", "done": false, "actions": []}', True, "ok"),
    ('{"message": "Hi", "done": false}', False, "ok"),
    ('{"message": "Hi", "done": false, "actions": [{"tool": "toggle_wifi", "params": {}}]}',
     True, "ok"),
    ('{"message": "Hi", "done": false}', True, "actions_missing"),
    ('```json\n{"message": "Hi", "done": false, "actions": []}\n```', True, "fenced"),
    ('{"message": "Hi", "done": false, "actions": [], "mood": "ok"}', True, "extra_fields"),
    ('{"message": "Hi", "done": false, "actions": [{"tool": "toggle_wifi"}]}', True,
     "bad_action"),
    ('{"message": "Hi", "done": false, "actions": [{"tool": "x", "params": "on"}]}', True,
     "bad_action"),
    ('{"message": "Hi", "done": false, "actions": ["toggle_wifi"]}', True, "bad_action"),
    ('{"message": "Hi", "done": false, "actions": {}}', True, "actions_not_list"),
    ("Sure, I turned it off.", True, "not_json"), ('["Hi"]', True, "not_object"),
    ('{"done": false, "actions": []}', True, "message_not_string"),
    ('{"message": "Hi", "done": "false", "actions": []}', True, "done_not_bool"),
    ('{"message": "Hi", "actions": []}', True, "done_not_bool"),
    (None, True, "not_json"),
    ('{"message": "Hi", "done": false, "actions": [{"tool": "x", "params": {"v": NaN}}]}',
     True, "not_json"),
    ('{"message": "Hi", "done": false, "actions": [{"tool": "x", "params": {"v": Infinity}}]}',
     True, "not_json"),
    ('{"message": "Hi", "done": false, "actions": [{"tool": "x", "params": {"v": -Infinity}}]}',
     True, "not_json"),
    ('{"message": "", "done": false, "actions": []}', True, "message_empty"),
    ('{"message": "  ", "done": false, "actions": []}', True, "message_empty")])
def test_reply_shape(reply, tools, shape):
    assert runlog.reply_shape(reply, tools) == shape


def test_off_shape_customer_replies_are_counted_not_hidden(copy, tmp_path):
    """'done': "false" is read as done=True by user_sim.py:269; no customer error, so it
    is counted here instead."""
    p, counts, m = fake_run(copy, tmp_path, "shape",
                            FAKE_CUSTOMER='{"message": "Hi, my phone has no service.", '
                                          '"done": "false", "actions": []}')
    s = m["summary"]
    assert s["customer_reply_shapes"] == {"done_not_bool": counts["customer"]}
    assert s["customer_errors"] == 0 and p.returncode == 0 and s["problems"] == []
    rows = [c for c in runlog.read_jsonl(tmp_path / "shape" / "calls.jsonl")
            if c["caller"] == "customer"]
    assert {c["reply_text"] for c in rows} == {'{"message": "Hi, my phone has no '
                                                'service.", "done": "false", "actions": []}'}


def test_a_customer_with_phone_tools_that_omits_actions_is_counted(copy, tmp_path):
    p, counts, m = fake_run(copy, tmp_path, "noact",
                            FAKE_CUSTOMER='{"message": "Hi, my phone has no service.", '
                                          '"done": false}')
    assert m["summary"]["customer_reply_shapes"] == {"actions_missing": counts["customer"]}
