#!/usr/bin/env python3
"""Job 2 scores from the two saved τ³ runs (docs/job2-step5-trace.md).

    .venv/bin/python -B scripts/score_deep.py results/deep-v0 results/deep-v2_deep-2

Pass rate per trial (τ³ reward = 1) on all 114 tasks and the unseen 51 / fed 62 / seen 63
subsets, mean and half-range over the 5 trials, the paired per-trial gap, pass^k, and the
August per-run rates beside them. Refuses anything that is not the two sealed, clean, paired,
full runs with v2_deep starting at most 8 h after v0 ends. Writes results/deep-score.json.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_deep as rd  # noqa: E402
import run_manifest as rm  # noqa: E402
from score import Refused, committed  # noqa: E402

VERSIONS = ("v0", "v2_deep")
MAX_GAP = datetime.timedelta(hours=8)


def success(reward) -> bool:
    """metrics/agent_metrics.py:12-16."""
    return reward is not None and 1 - 1e-6 <= reward <= 1 + 1e-6


def pass_hat_k(n: int, c: int, k: int) -> float:
    """metrics/agent_metrics.py:113-126."""
    return math.comb(c, k) / math.comb(n, k)


def subsets(policy_dir: Path) -> dict[str, set[str]]:
    out = {"all": set(rd.base_ids())}
    for s in rd.SUBSETS:
        out[s] = set((policy_dir / f"{s}_task_ids.txt").read_text().split())
    return out


def load(run: Path, version: str, mode: str) -> tuple[dict, list[dict]]:
    seal = rm.seal_path(run)
    if not seal.is_file() or rm.seal(run) != seal.read_text():
        raise Refused(f"{run.name}: folder does not match its seal")
    m = json.loads((run / "manifest.json").read_text())
    s = m.get("summary") or {}
    want = (mode, version, rd.POLICIES[version][1], [])
    if (m.get("mode"), m.get("version"), m.get("policy_sha256"), s.get("problems")) != want:
        raise Refused(f"{run.name}: manifest is not a clean {mode} run of {version}")
    n_tasks, trials = rd.MODES[mode]
    again = rd.check(run, rd.base_ids()[:n_tasks], trials, m["policy_sha256"])
    if again["problems"] or any(s.get(k) != v for k, v in again.items()):
        raise Refused(f"{run.name}: the saved files no longer give the manifest's summary")
    sims = json.loads((run / "tau3" / "results.json").read_text())["simulations"]
    return m, sims


def check_pair(ms: dict[str, dict], sims: dict[str, list[dict]]) -> None:
    keys = {v: sorted((x["task_id"], x["trial"], x["seed"]) for x in sims[v]) for v in VERSIONS}
    if keys["v0"] != keys["v2_deep"]:
        raise Refused("the two versions do not share the exact (task, trial, seed) keys")
    cfg = [{k: v for k, v in ms[v]["config"].items() if k != "save_to"} for v in VERSIONS]
    if cfg[0] != cfg[1]:
        raise Refused("the two runs' configs differ beyond save_to")
    t = {(v, k): datetime.datetime.fromisoformat(ms[v][k].replace("+0530", "+05:30"))
         for v in VERSIONS for k in ("start", "end")}
    gap = t["v2_deep", "start"] - t["v0", "end"]
    if not (t["v0", "start"] <= t["v0", "end"] and t["v2_deep", "start"] <= t["v2_deep", "end"]
            and datetime.timedelta(0) <= gap <= MAX_GAP):
        raise Refused(f"the runs are not back to back (v2_deep starts {gap} after v0 ends)")


def rates(sims: list[dict], ids: set[str], trials: int) -> list[float]:
    per = []
    for t in range(trials):
        rows = [success(x["reward_info"]["reward"]) for x in sims
                if x["trial"] == t and x["task_id"] in ids]
        per.append(100 * sum(rows) / len(rows))
    return per


def pass_k(sims: list[dict], ids: set[str], trials: int) -> dict[int, float]:
    c = {i: 0 for i in ids}
    for x in sims:
        if x["task_id"] in ids:
            c[x["task_id"]] += success(x["reward_info"]["reward"])
    return {k: 100 * statistics.mean(pass_hat_k(trials, n, k) for n in c.values())
            for k in range(1, trials + 1)}


def spread(v: list[float]) -> dict:
    return {"per_trial": v, "mean": statistics.mean(v), "half_range": (max(v) - min(v)) / 2}


def august(policy_dir: Path, subs: dict[str, set[str]]) -> dict:
    d = json.loads((policy_dir / "august_results_per_task.json").read_text())
    return {v: {s: [100 * sum(r[t] for t in ids) / len(ids) for r in d[v]]
                for s, ids in subs.items()} for v in VERSIONS}


def score(runs: dict[str, Path], mode: str = "full", policy_dir: Path = rd.PROJECT / "deep_policy") -> dict:
    loaded = {v: load(runs[v], v, mode) for v in VERSIONS}
    ms, sims = {v: x[0] for v, x in loaded.items()}, {v: x[1] for v, x in loaded.items()}
    check_pair(ms, sims)
    n_tasks, trials = rd.MODES[mode]
    subs = subsets(policy_dir)
    if mode != "full":
        subs = {s: ids & set(rd.base_ids()[:n_tasks]) for s, ids in subs.items()}
        subs = {s: ids for s, ids in subs.items() if ids}
    out = {"runs": {v: {"run": ms[v]["run"], "seal_sha256": rd.sha256(rm.seal(runs[v]).encode()),
                        "start": ms[v]["start"], "end": ms[v]["end"],
                        "cost_usd": ms[v]["summary"]["cost_usd"]} for v in VERSIONS},
           "subsets": {s: len(ids) for s, ids in subs.items()}, "scores": {}}
    for s, ids in subs.items():
        r = {v: rates(sims[v], ids, trials) for v in VERSIONS}
        out["scores"][s] = {
            **{v: {"pass_rate": spread(r[v]), "pass_hat_k": pass_k(sims[v], ids, trials)}
               for v in VERSIONS},
            "gap": spread([b - a for a, b in zip(r["v0"], r["v2_deep"])])}
    out["august"] = august(policy_dir, subsets(policy_dir))
    return out


def show(s: dict) -> str:
    lines = [f"{'subset':16s} {'v0 mean (per trial)':38s} {'v2 mean (per trial)':38s} gap mean ±half"]
    for name, x in s["scores"].items():
        f = lambda v: f"{x[v]['pass_rate']['mean']:5.1f} {[round(p, 1) for p in x[v]['pass_rate']['per_trial']]}"  # noqa: E731
        lines.append(f"{name:16s} {f('v0'):38s} {f('v2_deep'):38s} "
                     f"{x['gap']['mean']:+5.1f} ±{x['gap']['half_range']:.1f}")
    lines.append("pass^k (all): " + "  ".join(
        f"v0 {k}:{v:.1f}" for k, v in s["scores"]["all"]["v0"]["pass_hat_k"].items()))
    lines.append("              " + "  ".join(
        f"v2 {k}:{v:.1f}" for k, v in s["scores"]["all"]["v2_deep"]["pass_hat_k"].items()))
    for v in VERSIONS:
        lines.append(f"August {v} per run: " + "; ".join(
            f"{n} {[round(p, 1) for p in r]}" for n, r in s["august"][v].items()))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("v0", type=Path)
    ap.add_argument("v2_deep", type=Path)
    a = ap.parse_args(argv)
    out = a.v0.resolve().parent / "deep-score.json"
    out.unlink(missing_ok=True)
    runs = {"v0": a.v0.resolve(), "v2_deep": a.v2_deep.resolve()}
    try:
        for r in runs.values():
            committed(rm.seal_path(r))          # raises Refused unless committed and unchanged
        s = score(runs)
        out.write_text(json.dumps(s, indent=1))
        for v, r in runs.items():
            if rm.seal(r) != rm.seal_path(r).read_text():
                out.unlink()
                raise Refused(f"{r.name} changed while scoring")
    except Refused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    print(show(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
