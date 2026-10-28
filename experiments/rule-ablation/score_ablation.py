#!/usr/bin/env python3
"""Score a rule-removal run against prereg.md (modelled on scripts/score.py).

    .venv/bin/python -B experiments/rule-ablation/score_ablation.py experiments/rule-ablation/results/main

Uses only (repeat, scenario) blocks where every arm's conversation is clean. Per removed
rule: pass rate of its tied criteria, control vs variant; one-sided paired sign-flip test
over scenarios with tied criteria; the same for untied criteria. Writes <run>/score.json.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import run_manifest as rm  # noqa: E402
import score as sc  # noqa: E402
from runlog import conv_key  # noqa: E402
sys.path.insert(0, str(HERE))
import run_ablation as ra  # noqa: E402
from rules import ARMS, COPY_CHECKSUM, COPY_TREE, PAIR, RULES, sha, suite, ties as rule_ties  # noqa: E402

DROP_PTS, ALPHA, EXACT_MAX, MC_N, SEED = 20.0, 0.05, 20, 200_000, 20261028


class Refused(Exception):
    pass


def signflip_p(d: list[float]) -> float:
    """One-sided P(sum of randomly signed d >= observed sum), exact up to EXACT_MAX
    nonzero values (counts the observed assignment), Monte Carlo (hits+1)/(N+1) above."""
    d = [x for x in d if abs(x) > 1e-12]
    obs = sum(d)
    if not d:
        return 1.0
    if len(d) <= EXACT_MAX:
        dist = {0.0: 1}
        for x in d:
            nxt = defaultdict(int)
            for s, n in dist.items():
                nxt[round(s + x, 9)] += n
                nxt[round(s - x, 9)] += n
            dist = nxt
        return sum(n for s, n in dist.items() if s >= round(obs, 9) - 1e-9) / 2 ** len(d)
    rng = random.Random(SEED)
    hits = sum(sum(x if rng.random() < .5 else -x for x in d) >= obs - 1e-12
               for _ in range(MC_N))
    return (hits + 1) / (MC_N + 1)


REGISTERED = {"arms": list(ARMS), "cases": 24, "repeats": 3, "cap": 30, "concurrency": 8,
              "copy": "code-fixed",
              "models_requested": {"agent": rm.MODEL, "customer": rm.MODEL},
              "temperatures": {"agent": 0, "customer": 0.7},
              "models_resolved": {"agent": rm.MODEL, "customer": rm.MODEL,
                                  "customer_temperature": 0.7}}


def check_registered(run: Path, m: dict) -> None:
    """The pre-registered design, on the frozen copy, by the committed pinned code."""
    got = {**m, "arms": list(m.get("arms") or {})}
    bad = [k for k, v in REGISTERED.items() if got.get(k) != v]
    cmd = m.get("command") or []
    if bad or m.get("run") != run.name or m["summary"].get("copy_unchanged") is not True \
            or m.get("copy_checksum") != COPY_CHECKSUM \
            or m.get("copy_tree_hash") != COPY_TREE \
            or not str(cmd[1:2]).endswith("experiments/rule-ablation/run_ablation.py']") \
            or {"--copy", "--results", "--arms"} & set(cmd):
        raise Refused(f"not the registered run: {bad or 'run/copy/command'}")
    if not (sc.git_ok("merge-base", "--is-ancestor", str(m.get("project_commit")), "HEAD")
            and sc.git_ok("diff", "--quiet", str(m["project_commit"]), "HEAD", "--",
                          *ra.PINNED)):
        raise Refused("project commit missing or pinned files changed since the run")


def load(run: Path, registered: bool = True) -> tuple[dict, list[dict], int]:
    """Clean conversations of valid blocks, each trace checked against the suite."""
    m = json.loads((run / "manifest.json").read_text())
    s = m.get("summary") or {}
    if s.get("problems") or not s.get("status"):
        raise Refused(f"run has problems or no summary: {s.get('problems')}")
    seal = rm.seal_path(run)
    if not seal.exists() or seal.read_text() != rm.seal(run):
        raise Refused("seal missing or the run folder changed since it was sealed")
    if registered:
        if ra.dirty(ra.PINNED):
            raise Refused("pinned files have uncommitted changes")
        sc.committed(seal)
        check_registered(run, m)
    prompts = {k: set(v) for k, v in json.loads((run / "prompts.json").read_text()).items()}
    stopped = set(json.loads((run / "stopped.json").read_text()))
    want_sha = {a: v["policy_sha256"] for a, v in m["arms"].items()}
    again = ra.check(run, list(m["arms"]), m["repeats"], m["case_ids"], prompts, want_sha,
                     stopped)
    if again["problems"] or again["status"] != s["status"]:
        raise Refused(f"recomputed run check disagrees: {again['problems'][:3]}")
    meta = json.loads((HERE / "variants.json").read_text())
    for arm, v in m["arms"].items():
        if v["policy_sha256"] != meta[arm]["sha256"] or \
                sha((HERE / meta[arm]["file"]).read_text()) != meta[arm]["sha256"]:
            raise Refused(f"{arm}: run policy hash is not the pre-registered one")
    cases = suite()
    ids = m["case_ids"]
    if ids != [f"{i}:{c.get('source_task_id', '')}"[:120] for i, c in enumerate(cases)][:len(ids)]:
        raise Refused("case ids are not the suite's first cases, in order")
    arms, status = list(m["arms"]), s["status"]
    valid = [(r, c) for r in range(m["repeats"]) for c in ids
             if all(status.get(conv_key(a, r, c)) == "clean" for a in arms)]
    traces = {}
    for f in (run / "traces").glob("*.json"):
        d = json.loads(f.read_text())
        if (d["arm"], d["repeat"], d["case"]) in traces:
            raise Refused(f"duplicate trace {f.name}")
        traces[(d["arm"], d["repeat"], d["case"])] = d
    rows = []
    for r, c in valid:
        i = ids.index(c)
        want = [k["id"] for k in cases[i]["success_criteria"]]
        for a in arms:
            d = traces.get((a, r, c))
            if d is None:
                raise Refused(f"no trace for clean conversation {a}|{r}|{c}")
            crits = d["criteria"]
            if [k["id"] for k in crits] != want or not all(isinstance(k["ok"], bool)
                                                         for k in crits):
                raise Refused(f"{a}|{r}|{c}: criteria differ from the suite")
            ok = [k["ok"] for k in crits]
            if (d["solved"], d["actions_done"], d["actions_required"]) != \
                    (all(ok), sum(ok), len(ok)):
                raise Refused(f"{a}|{r}|{c}: solved/actions disagree with criteria")
            rows += [{"arm": a, "rep": r, "scen": i, "cid": k["id"], "ok": k["ok"]}
                     for k in crits]
    return m, rows, len(valid)


def compare(rows: list[dict], arm: str, keys: set) -> dict:
    """control vs arm on the criteria in keys: pooled pass %, drop, sign-flip p."""
    per = defaultdict(lambda: {"control": [0, 0], arm: [0, 0]})
    for x in rows:
        if x["arm"] in ("control", arm) and (x["scen"], x["cid"]) in keys:
            t = per[x["scen"]][x["arm"]]
            t[0] += x["ok"]
            t[1] += 1
    per = {s: v for s, v in per.items() if v["control"][1] and v[arm][1]}
    if not per:
        return {"n_scenarios": 0}
    tot = {a: [sum(v[a][i] for v in per.values()) for i in (0, 1)] for a in ("control", arm)}
    c, v = (100 * tot[a][0] / tot[a][1] for a in ("control", arm))
    d = [x["control"][0] / x["control"][1] - x[arm][0] / x[arm][1] for x in per.values()]
    p = signflip_p(d)
    return {"n_scenarios": len(per), "n_control": tot["control"][1], "n_variant": tot[arm][1],
            "control_pct": c, "variant_pct": v, "drop": c - v, "p": p,
            "caught": c - v >= DROP_PTS and p < ALPHA}


def score(rows: list[dict], ties: dict) -> dict:
    every = {(x["scen"], x["cid"]) for x in rows}
    tied = {r: {tuple(t) for t in ties[r]} for r in RULES}
    out = {}
    for r in RULES:
        if not any(x["arm"] == r for x in rows):
            continue
        out[r] = {"tied": compare(rows, r, tied[r]),
                  "untied": compare(rows, r, every - tied[r])}
    a, b = PAIR
    if a in out and b in out:
        only = {a: tied[a] - tied[b], b: tied[b] - tied[a]}
        out["pair"] = {v: {"own_only": compare(rows, v, only[v]),
                           "partner_only": compare(rows, v, only[a if v == b else b])}
                       for v in PAIR}
        for v in PAIR:
            o = out["pair"][v]
            o["lands_on_right_rule"] = bool(o["own_only"].get("caught")) and \
                not o["partner_only"].get("caught")
    return out


def table(out: dict) -> str:
    lines = ["| removed rule | tied criteria | control pass % | variant pass % | drop | p | "
             "caught? | untied change |", "|---|---|---|---|---|---|---|---|"]
    for r in RULES:
        if r not in out:
            continue
        t, u = out[r]["tied"], out[r]["untied"]
        if not t["n_scenarios"] or not u["n_scenarios"]:
            lines.append(f"| {r} | no valid scenarios | | | | | | |")
            continue
        lines.append(f"| {r} | {t['n_control']} in {t['n_scenarios']} scenarios | "
                     f"{t['control_pct']:.1f} | {t['variant_pct']:.1f} | {t['drop']:+.1f} | "
                     f"{t['p']:.4f} | {'yes' if t['caught'] else 'no'} | "
                     f"{-u['drop']:+.1f} (p={u['p']:.3f}) |")
    return "\n".join(lines)


def main(argv=None) -> int:
    run = Path((argv or sys.argv[1:])[0]).resolve()
    ties = json.loads((HERE / "ties.json").read_text())
    if ties != rule_ties():
        raise Refused("ties.json differs from rules.tied on the suite")
    m, rows, n_valid = load(run)
    out = score(rows, ties)
    res = {"run": m["run"], "valid_blocks": n_valid, "blocks": m["summary"]["blocks"],
           "cost_usd": m["summary"]["cost_usd"], "rules": out,
           "scorer_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                                           capture_output=True, text=True).stdout.strip()}
    (run / "score.json").write_text(json.dumps(res, indent=1) + "\n")
    print(f"valid blocks {n_valid} of {m['summary']['blocks']}, cost ${m['summary']['cost_usd']}")
    print(table(out))
    if "pair" in out:
        for v, o in out["pair"].items():
            print(f"pair {v}: own-only drop {o['own_only'].get('drop', 0):+.1f} "
                  f"p={o['own_only'].get('p', 1):.4f}; partner-only drop "
                  f"{o['partner_only'].get('drop', 0):+.1f} p={o['partner_only'].get('p', 1):.4f}"
                  f"; lands on right rule: {o['lands_on_right_rule']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
