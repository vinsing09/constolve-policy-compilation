#!/usr/bin/env python3
"""Recompute a run's scores from its saved traces (docs/step6-trace.md).

    .venv/bin/python scripts/score.py results/full-original-2 results/full-fixed-2

Same three numbers as analyse_run.py:114-140 (solved fully, criteria passed, actions
completed), per arm and gap, pooled over the run and per repeat. Refuses anything that
is not exactly the run its manifest says it is. Writes <run>/score.json.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_manifest as rm  # noqa: E402

SUITE = HERE.parent / "code-original" / "generated" / "graph_suite_telecom.json"
ARMS = ("with_policy", "stub")
METRICS = ("fully", "partly", "action")


class Refused(Exception):
    pass


def august_rule(cid: str) -> str:
    """analyse_run.py:66-72, the rule it infers from a criterion id."""
    if "forb" in cid or cid.startswith("no_premature"):
        return "tool_not_called"
    if cid.startswith("order"):
        return "tool_called_before"
    return "tool_called_with"


def suite_cases(n: int) -> dict[str, list[tuple[str, str]]]:
    """Case id (services/suite_bench.py:45) -> [(criterion id, rule)], first n cases."""
    suite = json.loads(SUITE.read_text())
    if len(suite) < n:
        raise Refused(f"suite has {len(suite)} cases, run needs {n}")
    out = {}
    for i, case in enumerate(suite[:n]):
        rules = out[f"{i}:{case.get('source_task_id', '')}"[:120]] = \
            [(c["id"], c["rule"]) for c in case["success_criteria"]]
        for cid, rule in rules:
            if rule != august_rule(cid):
                raise Refused(f"case {i} {cid}: suite rule {rule}, analyse_run infers "
                              f"{august_rule(cid)}")
    return out


def load(traces: Path, case_ids: list[str], repeats: int) -> list[dict]:
    """Every trace, checked against the exact expected set and the suite."""
    rules = suite_cases(len(case_ids))
    if case_ids != list(rules):
        raise Refused("case ids are not the suite's first cases, in order")
    want = {(a, r, c) for a in ARMS for r in range(repeats) for c in case_ids}
    rows, seen = [], set()
    for f in sorted(traces.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except ValueError as exc:
            raise Refused(f"{f.name}: not JSON ({exc})") from None
        key = (d.get("arm"), d.get("repeat"), d.get("case"))
        if key not in want:
            raise Refused(f"{f.name}: {key} is not in this run")
        if key in seen:
            raise Refused(f"{f.name}: {key} appears twice")
        seen.add(key)
        crits = d.get("criteria")
        if not isinstance(crits, list) or not crits:
            raise Refused(f"{f.name}: no criteria")
        if any(not isinstance(c, dict) or set(c) != {"id", "ok", "why"} for c in crits):
            raise Refused(f"{f.name}: a criterion is not exactly {{id, ok, why}}")
        expect = rules[key[2]]
        if [c.get("id") for c in crits] != [cid for cid, _ in expect]:
            raise Refused(f"{f.name}: criterion ids differ from the suite")
        if not all(isinstance(c.get("ok"), bool) for c in crits):
            raise Refused(f"{f.name}: a criterion's ok is not true/false")
        ok = [c["ok"] for c in crits]
        if (d.get("solved"), d.get("actions_done"), d.get("actions_required")) != \
                (all(ok), sum(ok), len(ok)):
            raise Refused(f"{f.name}: solved/actions_done/actions_required disagree "
                          "with its criteria")
        rows.append({"arm": key[0], "repeat": key[1], "case": key[2],
                     "ok": ok,
                     "must": [rule == "tool_called_with" for _, rule in expect]})
    if seen != want:
        raise Refused(f"{len(want - seen)} conversations missing, e.g. "
                      f"{sorted(want - seen, key=str)[:2]}")
    return rows


def metrics(rows: list[dict]) -> dict:
    act = [ok for d in rows for ok, must in zip(d["ok"], d["must"]) if must]
    return {"n": len(rows),
            "fully": 100 * statistics.mean(all(d["ok"]) for d in rows),
            "partly": 100 * statistics.mean(sum(d["ok"]) / len(d["ok"]) for d in rows),
            "action": 100 * sum(act) / len(act)}


def compare(rows: list[dict]) -> dict:
    w = metrics([d for d in rows if d["arm"] == "with_policy"])
    s = metrics([d for d in rows if d["arm"] == "stub"])
    return {"with_policy": w, "stub": s, "gap": {k: w[k] - s[k] for k in METRICS}}


def score(rows: list[dict], repeats: int) -> dict:
    per = [compare([d for d in rows if d["repeat"] == r]) for r in range(repeats)]
    spread = {}
    for k in METRICS:
        g = [p["gap"][k] for p in per]
        spread[k] = {"per_repeat": g, "mean": statistics.mean(g),
                     "half_range": (max(g) - min(g)) / 2}
    return {"pooled": compare(rows), "per_repeat": per, "repeat_gaps": spread}


def check_report(report: Path, rows: list[dict], m: dict) -> None:
    """bench's own report (bench.py:140-142,212-217) must agree with manifest and traces."""
    d = json.loads(report.read_text())
    if (d.get("suite"), d.get("cases"), d.get("repeats"), d.get("cap")) != \
            (m["suite"], m["cases"], m["repeats"], m["cap"]):
        raise Refused("report.json suite/cases/repeats/cap disagree with the manifest")
    got = sorted((r.get("label"), r.get("rep"), r.get("id"), r.get("error"), r.get("solved"),
                  r.get("k"), r.get("n")) for r in d["runs"])
    want = sorted((d["arm"], d["repeat"], d["case"], None, all(d["ok"]), sum(d["ok"]),
                   len(d["ok"])) for d in rows)
    if got != want:
        raise Refused("report.json rows disagree with the traces")


WANT = {"suite": "ours", "domain": "telecom",
        "models_requested": {"agent": rm.MODEL, "customer": rm.MODEL},
        "temperatures": {"agent": 0, "customer": 0.7},
        "models_resolved": {"agent": rm.MODEL, "customer": rm.MODEL,
                            "customer_temperature": 0.7},
        "bench_exit": 0}


JOB1 = {"cases": 24, "repeats": 5, "cap": 30, "concurrency": 8}   # PLAN.md:12-14
JOB1_COMMAND = ["scripts/bench.py", "--suite", "ours", "--domain", "telecom", "--cases", "24",
                "--repeats", "5", "--max-turns", "30"]
COPIES = ("code-original", "code-fixed")


def check_manifest(run: Path, m: dict, job1: bool) -> None:
    s = m.get("summary") or {}
    if s.get("problems") != []:
        raise Refused(f"{run.name}: manifest problems {s.get('problems')!r}; "
                      "the run does not count")
    bad = [k for k, v in {**WANT, **(JOB1 if job1 else {})}.items() if m.get(k) != v]
    if bad or s.get("copy_unchanged") is not True or len(m.get("case_ids") or []) != m["cases"]:
        raise Refused(f"{run.name}: manifest fields wrong: {bad or 'copy/cases'}")
    copy = rm.PROJECT / str(m.get("copy"))
    who = [m.get("run") == run.name, m.get("copy") in COPIES,
           m.get("copy_checksum") == rm.stored("checksum", copy),
           m.get("copy_tree_hash") == rm.stored("tree", copy),
           not job1 or m.get("bench_command") == JOB1_COMMAND]
    if not all(who):
        raise Refused(f"{run.name}: manifest identity wrong (run, copy, checksum, tree, "
                      f"command: {who})")


WRAPPER = ["scripts/run_manifest.py", "scripts/runlog.py"]


def git_ok(*args: str) -> bool:
    return subprocess.run(["git", "-C", str(rm.PROJECT), *args], capture_output=True,
                          stdin=subprocess.DEVNULL).returncode == 0


def check_provenance(run: Path, m: dict) -> None:
    """The run's recorded command, place, times and wrapper version (run_manifest.py:241-252).
    python/packages are recorded for the report; the project has one venv, so they are
    required to be present, not pinned."""
    try:
        start, end = (datetime.datetime.fromisoformat(m[k]) for k in ("start", "end"))
    except (KeyError, TypeError, ValueError):
        start = end = None
    cmd = m.get("command") or []
    ok = {
        "times": start is not None and end >= start,
        "command": len(cmd) == 7 + len(JOB1_COMMAND) - 1 and cmd[:3] + cmd[4:] == [
            str(rm.PROJECT / ".venv" / "bin" / "python"), str(rm.HERE / "run_manifest.py"),
            "--copy", "--run", m.get("run"), "--", *JOB1_COMMAND[1:]]
        and (rm.PROJECT / cmd[3]).resolve() == rm.PROJECT / str(m.get("copy")),
        "cwd": m.get("cwd") == str(rm.PROJECT / str(m.get("copy"))),
        "versions": bool(m.get("python")) and bool(m.get("packages")),
        "wrapper": bool(m.get("project_commit"))
        and git_ok("cat-file", "-e", f"{m['project_commit']}^{{commit}}")
        and git_ok("merge-base", "--is-ancestor", m["project_commit"], "HEAD")
        and git_ok("diff", "--quiet", m["project_commit"], "HEAD", "--", *WRAPPER)}
    if not all(ok.values()):
        raise Refused(f"{run.name}: manifest provenance wrong: "
                      f"{[k for k, v in ok.items() if not v]}")


def recheck(run: Path, m: dict) -> None:
    """The wrapper's own checks (run_manifest.check) rerun on the saved files."""
    try:
        again = rm.check(run, m["repeats"], m["case_ids"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise Refused(f"{run.name}: saved files unreadable by the wrapper's check "
                      f"({type(exc).__name__}: {exc})") from None
    if again["problems"] or any(m["summary"].get(k) != v for k, v in again.items()):
        raise Refused(f"{run.name}: the saved files no longer give the manifest's summary")


def check_seal(run: Path, seal: Path) -> None:
    if not seal.is_file():
        raise Refused(f"{run.name}: no seal at {seal}")
    if rm.seal(run) != seal.read_text():
        raise Refused(f"{run.name}: files differ from the seal {seal.name}")


def score_run(run: Path, seal: Path | None = None, job1: bool = True) -> dict:
    """job1=False only for tests of smaller runs in temporary folders (no Job 1 size,
    command or provenance checks); main() always uses job1=True."""
    seal = seal or rm.seal_path(run)
    (run / "score.json").unlink(missing_ok=True)
    check_seal(run, seal)
    m = json.loads((run / "manifest.json").read_text())
    check_manifest(run, m, job1)
    if job1:
        check_provenance(run, m)
    rows = load(run / "traces", m["case_ids"], m["repeats"])
    check_report(run / "report.json", rows, m)
    recheck(run, m)
    if len(rows) != 2 * m["cases"] * m["repeats"]:
        raise Refused(f"{run.name}: {len(rows)} traces, manifest implies "
                      f"{2 * m['cases'] * m['repeats']}")
    check_seal(run, seal)
    return {"run": m["run"], "copy": m["copy"], "copy_tree_hash": m["copy_tree_hash"],
            "conversations": len(rows), **score(rows, m["repeats"]),
            "seal_sha256": hashlib.sha256(rm.seal(run).encode()).hexdigest()}


def show(s: dict) -> str:
    names = {"fully": "solved fully", "partly": "criteria passed",
             "action": "actions completed"}
    p, g = s["pooled"], s["repeat_gaps"]
    lines = [f"{s['run']} ({s['copy']}, {s['conversations']} conversations)",
             f"  {'':18s} {'policy':>7s} {'stub':>7s} {'gap':>7s}   per-repeat gaps"]
    for k in METRICS:
        lines.append(f"  {names[k]:18s} {p['with_policy'][k]:7.1f} {p['stub'][k]:7.1f} "
                     f"{p['gap'][k]:+7.1f}   {[round(x, 1) for x in g[k]['per_repeat']]}"
                     f" mean {g[k]['mean']:+.1f} half-range {g[k]['half_range']:.1f}")
    return "\n".join(lines)


def committed(seal: Path) -> None:
    """A seal only counts once it is in git, unchanged since the commit."""
    git = ["git", "-C", str(seal.parent)]
    tracked = subprocess.run([*git, "ls-files", "--error-unmatch", seal.name],
                             capture_output=True).returncode == 0
    clean = subprocess.run([*git, "diff", "--quiet", "HEAD", "--", seal.name],
                           capture_output=True).returncode == 0
    if not (tracked and clean):
        raise Refused(f"seal {seal} is not committed, or changed since its commit")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    a = ap.parse_args(argv)
    for run in a.runs:
        (run / "score.json").unlink(missing_ok=True)
        try:
            committed(rm.seal_path(run))
            s = score_run(run)
        except Refused as exc:
            print(f"REFUSED {run}: {exc}", file=sys.stderr)
            return 1
        (run / "score.json").write_text(json.dumps(s, indent=1))
        try:
            check_seal(run, rm.seal_path(run))
        except Refused as exc:
            (run / "score.json").unlink()
            print(f"REFUSED {run}: {exc} (changed while scoring)", file=sys.stderr)
            return 1
        print(show(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
