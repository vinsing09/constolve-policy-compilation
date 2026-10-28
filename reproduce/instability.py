#!/usr/bin/env python3
"""Section 5.5 and Table 7: verdict instability over 834 benchmark task runs.

    python -B reproduce/instability.py

records/verdict-instability/verdicts.csv holds one row per task run (domain, task, repeat, run
id, pass/fail, whether the run recorded an error), exported from the run database of the
earlier internal system that ran tau3-bench's own tasks, environment and simulated customer.
pass^k is the unbiased estimate C(s, k) / C(3, k) averaged over tasks, where s is the number of
the 3 runs that passed. The per-task counts are also compared with the archived summaries in
records/verdict-instability/archived/. Exit 1 on any difference from the paper.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

REC = Path(__file__).resolve().parents[1] / "records/verdict-instability"
PAPER = {  # tasks, runs, flipped tasks, flip %, passes per repeat, pass^1 - pass^3 (pp)
    "airline": (50, 150, 17, 34.0, [23, 31, 27], 16.0),
    "telecom": (114, 342, 34, 29.8, None, 14.6),
    "retail": (114, 342, 35, 30.7, [78, 77, 77], 14.3),
}
PAPER_PCT = {"airline": [46, 62, 54], "retail": [68, 68, 68]}


def load() -> dict[str, dict[str, list[int]]]:
    per: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    errors = 0
    with (REC / "verdicts.csv").open() as f:
        for r in csv.DictReader(f):
            per[r["domain"]][r["task"]][int(r["repeat"]) - 1] = int(r["passed"])
            errors += int(r["error"])
    if errors:
        raise SystemExit(f"{errors} runs recorded an error")
    return per


def pass_hat(counts: list[int], k: int) -> float:
    return 100 * sum(comb(s, k) / comb(3, k) for s in counts) / len(counts)


def compute() -> dict:
    out = {}
    for dom, tasks in load().items():
        s = [sum(v) for v in tasks.values()]
        out[dom] = {"tasks": len(tasks), "runs": 3 * len(tasks),
                    "flipped": sum(0 < x < 3 for x in s),
                    "per_repeat": [sum(v[i] for v in tasks.values()) for i in range(3)],
                    "pass1": pass_hat(s, 1), "pass3": pass_hat(s, 3),
                    "per_task": {t: sum(v) for t, v in tasks.items()}}
        out[dom]["flip_pct"] = 100 * out[dom]["flipped"] / len(tasks)
        out[dom]["drop"] = out[dom]["pass1"] - out[dom]["pass3"]
    return out


def _archived(r: dict) -> list[str]:
    bad = []
    for dom in ("telecom", "retail"):
        if json.loads((REC / f"archived/baseline_{dom}.json").read_text())["per_task"] != \
                r[dom]["per_task"]:
            bad.append(f"{dom}: per-task counts differ from archived baseline_{dom}.json")
    agr = json.loads((REC / "archived/tau3_per_task_agreement.json").read_text())["per_task"]
    got = {t: sum(v.values()) for t, v in agr.items()}
    if {t.split("_", 1)[1]: n for t, n in got.items()} != r["airline"]["per_task"]:
        bad.append("airline: per-task counts differ from archived tau3_per_task_agreement.json")
    return bad


def check() -> list[str]:
    r = compute()
    bad = _archived(r)
    for dom, (n, runs, flips, pct, reps, drop) in PAPER.items():
        g = r[dom]
        got = (g["tasks"], g["runs"], g["flipped"], round(g["flip_pct"], 1), round(g["drop"], 1))
        if got != (n, runs, flips, pct, drop):
            bad.append(f"{dom}: got {got}, paper {(n, runs, flips, pct, drop)}")
        if reps and g["per_repeat"] != reps:
            bad.append(f"{dom}: passes per repeat {g['per_repeat']}, expected {reps}")
        if dom in PAPER_PCT and [round(100 * x / n) for x in g["per_repeat"]] != PAPER_PCT[dom]:
            bad.append(f"{dom}: per-repeat scores differ from the paper's {PAPER_PCT[dom]}")
    total = (sum(g["tasks"] for g in r.values()), sum(g["runs"] for g in r.values()))
    if total != (278, 834):
        bad.append(f"total {total}, paper (278, 834)")
    drops = [round(g["drop"], 1) for g in r.values()]
    if round(max(drops) - min(drops), 1) != 1.7:
        bad.append(f"spread of drops {max(drops) - min(drops):.1f}, paper 1.7")
    base = sorted(round(g["pass1"]) for g in r.values())
    if (base[0], base[-1]) != (30, 68):
        bad.append(f"pass^1 base rates {base}, paper from 30% to 68%")
    return bad


def main() -> int:
    r = compute()
    print("Table 7 (Verdict instability, three identical runs per task)")
    print(f"  {'domain':8s} tasks runs flipped        passes/run   pass^1 pass^3 drop")
    for dom in ("airline", "telecom", "retail"):
        g = r[dom]
        print(f"  {dom:8s} {g['tasks']:5d} {g['runs']:4d} {g['flipped']:3d} ({g['flip_pct']:4.1f}%) "
              f"{str(g['per_repeat']):14s} {g['pass1']:6.1f} {g['pass3']:6.1f} {g['drop']:4.1f}pp")
    print(f"  total {sum(g['tasks'] for g in r.values())} tasks, "
          f"{sum(g['runs'] for g in r.values())} runs; no run recorded an error")
    bad = check()
    for b in bad:
        print("MISMATCH", b)
    print("instability: all values match the paper and the archived summaries" if not bad
          else f"instability: {len(bad)} mismatch(es)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
