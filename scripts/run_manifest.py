#!/usr/bin/env python3
"""Run one frozen code copy's scripts/bench.py unchanged, with a manifest and a call log.

    .venv/bin/python scripts/run_manifest.py --copy code-fixed --run smoke-fixed -- \
        --suite ours --domain telecom --cases 1 --repeats 1 --max-turns 30

Writes results/<run>/: manifest.json (before the run, completed after), calls.jsonl
(one line per model call), customer_errors.jsonl, stdout.log, traces/, report.json,
and last seals/<run>.sha256 (sha256sum format) beside results/, to be committed.
A run counts only if manifest.summary.problems is empty.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
from runlog import RETURNED_OK, SIM_ERROR, Recorder, conv_key, read_jsonl, tree_hash  # noqa: E402

MODEL = "gpt-4.1"
ARMS = ("with_policy", "stub")
EXACT = ("DEFAULT_MODEL", "USER_SIM_MODEL", "DEFAULT_MAX_RETRIES", "REQUEST_TIMEOUT")
PREFIXES = ("JOURNEY_", "LITELLM_", "OPENAI_", "AZURE_", "TAU2_")


def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def setup_checksum(copy: Path) -> str:
    """The method behind scripts/checksum-*.txt (19 Oct setup), locale pinned."""
    out = subprocess.run(
        "find . -type f ! -name .env | sort | xargs -d '\\n' sha256sum | sha256sum",
        shell=True, cwd=copy, capture_output=True, text=True, check=True,
        env={**os.environ, "LC_ALL": "en_US.UTF-8"})
    return out.stdout.split()[0]


def stored(kind: str, copy: Path) -> str | None:
    f = HERE / f"{kind}-{copy.name.removeprefix('code-')}.txt"
    return f.read_text().split()[0] if f.exists() else None


def dotenv(copy: Path, key: str) -> str | None:
    for line in (copy / ".env").read_text().splitlines():
        k, sep, v = line.partition("=")
        if sep and not line.strip().startswith("#") and k.strip() == key:
            return v.strip()
    return None


def bench_args(argv: list[str]) -> argparse.Namespace:
    """bench.py's own arguments and defaults (bench.py:146-153), to record them."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--domain", default="telecom")
    ap.add_argument("--cases", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=8)
    return ap.parse_args(argv)


def outputs(copy: Path, b: argparse.Namespace) -> tuple[Path, Path]:
    return copy / "logs" / "bench", copy / "reports" / f"bench_{b.suite}_{b.domain}.json"


def preflight(copy: Path, run_dir: Path, b: argparse.Namespace, environ) -> list[str]:
    bad = []
    if (b.suite, b.domain) != ("ours", "telecom"):
        bad.append("only --suite ours --domain telecom is validated")
    if run_dir.exists():
        bad.append(f"{run_dir} already exists")
    traces, report = outputs(copy, b)
    if traces.exists():
        bad.append(f"{traces} exists: an earlier run's files would mix in")
    if report.exists():
        bad.append(f"{report} exists: an earlier run's report would be overwritten")
    for kind, fn in (("checksum", setup_checksum), ("tree", tree_hash)):
        want = stored(kind, copy)
        if want is None or fn(copy) != want:
            bad.append(f"{copy.name} does not match {kind}-*.txt")
    for k in environ:
        if k in EXACT or k.startswith(PREFIXES):
            bad.append(f"shell sets {k}; it would override .env or change litellm")
    if dotenv(copy, "DEFAULT_MODEL") != MODEL:
        bad.append(f".env DEFAULT_MODEL is not {MODEL}")
    return bad


def lock(copy: Path):
    """One run per code copy at a time, held from preflight to the last check."""
    f = (PROJECT / "results" / f".lock-{copy.name}").open("a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    return f


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            if not st.closed:
                st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            if not st.closed:
                st.flush()


def usable(call: dict) -> bool:
    """A paid gpt-4.1 call always reports tokens both ways and costs something."""
    return all(isinstance(call.get(k), int) and call[k] > 0
               for k in ("prompt_tokens", "completion_tokens")) and \
        isinstance(call.get("cost_usd"), (int, float)) and call["cost_usd"] > 0


def collect(copy: Path, run_dir: Path, b, case_ids: list[str]) -> dict:
    """Move the run's files out of the copy and check them against what was asked for."""
    traces, report = outputs(copy, b)
    dest = run_dir / "traces"
    dest.mkdir()
    for f in sorted(traces.glob("*.json")) if traces.exists() else []:
        shutil.move(str(f), dest / f.name)
    if report.exists():
        shutil.move(str(report), run_dir / "report.json")
    return check(run_dir, b.repeats, case_ids)


def check(run_dir: Path, repeats: int, case_ids: list[str]) -> dict:
    """The run's saved files checked against what was asked for; reads only run_dir."""
    dest = run_dir / "traces"
    expected = {conv_key(a, r, c) for a in ARMS for r in range(repeats) for c in case_ids}

    keys, in_transcript, no_criteria = [], 0, 0
    for f in sorted(dest.glob("*.json")):
        d = json.loads(f.read_text())
        keys.append(conv_key(d["arm"], d["repeat"], d["case"]))
        in_transcript += any(t.get("role") == "user" and
                             str(t.get("content") or "").startswith(SIM_ERROR)
                             for t in d.get("history") or [])
        no_criteria += not d.get("criteria")
    rows = (json.loads((run_dir / "report.json").read_text())["runs"]
            if (run_dir / "report.json").exists() else [])
    rkeys = [conv_key(r["label"], r["rep"], r["id"]) for r in rows]
    calls = read_jsonl(run_dir / "calls.jsonl")
    cust_err = read_jsonl(run_dir / "customer_errors.jsonl")
    ok = [c for c in calls if c["ok"]]
    s = {"expected": len(expected), "trace_files": len(keys), "report_rows": len(rows),
         "report_errors": sum(1 for r in rows if r.get("error")),
         "conversations_with_no_criteria": no_criteria,
         "customer_errors": len(cust_err),
         "conversations_with_customer_error": len({e["conv"] for e in cust_err}),
         "customer_errors_visible_in_transcript": in_transcript,
         "model_calls": len(calls), "failed_calls": len(calls) - len(ok),
         "sdk_retries": sum(c["sdk_retries"] for c in calls),
         "calls_by_caller": {k: sum(c["caller"] == k for c in calls)
                             for k in sorted({c["caller"] for c in calls})},
         "requested_models": sorted({str(c["requested_model"]) for c in calls}),
         "returned_models": sorted({str(c.get("returned_model")) for c in ok}),
         "calls_missing_usage_or_cost": sum(not usable(c) for c in ok),
         "customer_reply_shapes": dict(sorted(collections.Counter(
             str(c.get("reply_shape")) for c in ok if c["caller"] == "customer").items())),
         "customer_response_formats": sorted({json.dumps(c.get("response_format"))
                                              for c in calls if c["caller"] == "customer"}),
         "cost_usd": round(sum(c["cost_usd"] for c in ok if usable(c)), 4)}
    p = []
    if set(keys) != expected or len(keys) != len(expected):
        p.append("trace files are not exactly the expected conversations")
    if set(rkeys) != expected or len(rkeys) != len(expected):
        p.append("report rows are not exactly the expected conversations")
    for k in ("report_errors", "conversations_with_no_criteria", "customer_errors",
              "failed_calls", "calls_missing_usage_or_cost"):
        if s[k]:
            p.append(f"{k}: {s[k]}")
    if set(s["calls_by_caller"]) - {"agent", "customer"}:
        p.append("a model call came from neither the agent nor the customer")
    if s["requested_models"] != [MODEL]:
        p.append(f"requested models {s['requested_models']}")
    if not ok or not all(RETURNED_OK.match(str(c.get("returned_model"))) for c in ok):
        p.append(f"returned models {s['returned_models']}")
    for who in ("agent", "customer"):
        have = {c["conv"] for c in calls if c["caller"] == who}
        if have != expected:
            p.append(f"{who} calls missing for {len(expected - have)} conversations, "
                     f"from unknown conversations: {len(have - expected)}")
    s["problems"] = p
    return s


def seal(run_dir: Path) -> str:
    """sha256sum lines for every file in the run folder except score.json."""
    files = sorted((f for f in run_dir.rglob("*") if f.is_file() and f.name != "score.json"),
                   key=lambda f: os.fsencode(f.relative_to(run_dir).as_posix()))
    return "".join(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  ./"
                   f"{f.relative_to(run_dir).as_posix()}\n" for f in files)


def seal_path(run_dir: Path) -> Path:
    return run_dir.parent.parent / "seals" / f"{run_dir.name}.sha256"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", required=True, help="code-original or code-fixed")
    ap.add_argument("--run", required=True)
    ap.add_argument("--results", default=str(PROJECT / "results"))
    ap.add_argument("bench", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    rest = a.bench[1:] if a.bench[:1] == ["--"] else a.bench
    copy, run_dir = Path(a.copy).resolve(), Path(a.results).resolve() / a.run
    b = bench_args(rest)

    held = lock(copy)
    bad = ["another run holds the lock on this copy"] if held is None else \
        preflight(copy, run_dir, b, os.environ)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 2
    run_dir.mkdir(parents=True)
    tree_before = tree_hash(copy)
    manifest = {
        "run": a.run, "copy": copy.name, "copy_checksum": setup_checksum(copy),
        "copy_tree_hash": tree_before,
        "project_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT,
                                         capture_output=True, text=True).stdout.strip(),
        "command": [sys.executable, __file__, *argv],
        "bench_command": ["scripts/bench.py", *rest], "cwd": str(copy),
        "python": sys.version.split()[0],
        "packages": {p: version(p) for p in ("litellm", "openai", "httpx", "pydantic")},
        "models_requested": {"agent": MODEL, "customer": MODEL},
        "temperatures": {"agent": 0, "customer": 0.7},
        "cap": b.max_turns, "cases": b.cases, "repeats": b.repeats,
        "concurrency": b.concurrency, "suite": b.suite, "domain": b.domain,
        "start": now()}
    mpath = run_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=1))

    sys.dont_write_bytecode = True
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != HERE]
    os.chdir(copy)
    log = (run_dir / "stdout.log").open("w")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log), Tee(sys.__stderr__, log)
    rc, case_ids = None, []
    try:
        spec = importlib.util.spec_from_file_location("bench", copy / "scripts" / "bench.py")
        bench = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bench)
        import litellm
        import services.user_sim as us
        if Path(us.__file__).resolve() != copy / "services" / "user_sim.py":
            raise SystemExit(f"loaded {us.__file__}, not this copy's")
        sim = us.UserSim()
        manifest["models_resolved"] = {"agent": os.environ.get("DEFAULT_MODEL"),
                                       "customer": sim.model,
                                       "customer_temperature": sim.temperature}
        mpath.write_text(json.dumps(manifest, indent=1))
        if manifest["models_resolved"] != {"agent": MODEL, "customer": MODEL,
                                           "customer_temperature": 0.7}:
            raise SystemExit(f"models resolved wrong: {manifest['models_resolved']}")
        case_ids = [c["id"] for c in bench.our_cases(b.domain, b.cases)]
        manifest["case_ids"] = case_ids
        mpath.write_text(json.dumps(manifest, indent=1))
        if len(case_ids) != b.cases:
            raise SystemExit(f"asked for {b.cases} cases, the suite gives {len(case_ids)}")
        Recorder(run_dir / "calls.jsonl", run_dir / "customer_errors.jsonl").install(
            bench, us, litellm)
        sys.argv = ["scripts/bench.py", *rest]
        rc = asyncio.run(bench.main())
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        log.close()
        s = collect(copy, run_dir, b, case_ids)
        traces = outputs(copy, b)[0]
        if traces.exists() and not any(traces.iterdir()):
            traces.rmdir()
        s["copy_unchanged"] = tree_hash(copy) == tree_before
        if len(case_ids) != b.cases:
            s["problems"].append(f"cases: asked {b.cases}, ran {len(case_ids)}")
        if rc != 0:
            s["problems"].append(f"bench exit {rc}")
        if not s["copy_unchanged"]:
            s["problems"].append("code copy changed during the run")
        manifest.update(end=now(), bench_exit=rc, summary=s)
        mpath.write_text(json.dumps(manifest, indent=1))
        seal_path(run_dir).parent.mkdir(exist_ok=True)
        seal_path(run_dir).write_text(seal(run_dir))
        held.close()
    print(json.dumps(manifest["summary"], indent=1))
    return 0 if not s["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
