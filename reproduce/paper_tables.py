#!/usr/bin/env python3
"""Recompute every table and figure of the paper that rests on this repository, and check each
value against the number printed in the paper (paper-v7 preprint). No model calls.

    python -B reproduce/paper_tables.py

Run after reproduce/reanalysis.py has confirmed experiments/duplicate-reanalysis/results.json
(verify_all.sh does this); the 19-cluster statistics of the Discrimination section and of the
rule-removal tables are read from that file. Everything else is computed here from the suite,
the policy, the compiled environments, the stored traces and the experiments' score files, and
(Sections 5.1, 5.2, 5.5 and Table 7) from records/ via repair_coverage.py, determinism.py and
instability.py. Exit 1 on any mismatch.
"""
from __future__ import annotations

import csv
import json
import runpy
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path[:0] = [str(ROOT / "reproduce"), str(ROOT / "scripts"), str(ROOT / "experiments" / "rule-ablation"),
                str(ROOT / "experiments" / "rule-ablation-followup")]
import repair_coverage as COV  # noqa: E402
import design as D  # noqa: E402
import determinism as DET  # noqa: E402
import instability as INST  # noqa: E402
import rules  # noqa: E402
import score as sc  # noqa: E402

BAD: list[str] = []


def check(what: str, got, want, tol: float = 0.0) -> None:
    ok = abs(got - want) <= tol + 1e-9 if isinstance(want, (int, float)) and not isinstance(
        want, bool) else got == want
    if not ok:
        BAD.append(f"{what}: got {got!r}, paper {want!r}")


def table_domains() -> None:
    """Table 1 (tab:domains). Telecom column fully; airline/retail compiled counts only."""
    tel = ROOT / "code-fixed/third_party/tau3-bench"
    data = tel / "data/tau2/domains/telecom"
    base = set(json.loads((data / "split_tasks.json").read_text())["base"])
    tasks = [t for t in json.loads((data / "tasks.json").read_text()) if t["id"] in base]
    user = sum(any(a.get("requestor") == "user" for a in t["evaluation_criteria"].get("actions")
                   or []) for t in tasks)
    src = tel / "src/tau2/domains/telecom"
    schemas = sum((src / f).read_text().count("@is_tool") for f in ("tools.py", "user_tools.py"))
    policy = len((ROOT / "experiments/rule-ablation/policies/control.md").read_text())
    env = ROOT / "code-fixed/generated/env_telecom.py"
    mod = runpy.run_path(str(env))
    lines = len(env.read_text().splitlines())
    meta = {d: json.loads((ROOT / f"code-fixed/generated/env_{d}.meta.json").read_text())
            for d in ("telecom", "airline", "retail")}
    comp = {d: f"{m['n_compiled']}/{m['n_schemas']}" for d, m in meta.items()}
    print("Table 1 (Domains and compilation outcomes), telecom column and compiled tools")
    print(f"  telecom: tasks {len(tasks)}, tool schemas {schemas}, policy length {policy:,}, "
          f"tasks with customer actions {user}, tools compiled {comp['telecom']}, health "
          f"predicates {len(mod['PREDICATES'])}, derived observables {len(mod['OBSERVABLES'])}, "
          f"environment size {lines:,} lines")
    bench = sum(len(f.read_text().splitlines()) for f in src.glob("*.py"))
    print(f"  airline tools compiled {comp['airline']}; retail tools compiled {comp['retail']}")
    print(f"  Section 5.1: the benchmark's own telecom environment, {bench:,} lines")
    check("Section 5.1 benchmark environment lines", bench, 2851)
    for k, g, w in (("tasks", len(tasks), 114), ("schemas", schemas, 43), ("policy", policy, 23318),
                    ("customer actions", user, 94), ("predicates", len(mod["PREDICATES"]), 11),
                    ("observables", len(mod["OBSERVABLES"]), 5), ("lines", lines, 2337)):
        check(f"Table 1 telecom {k}", g, w)
    check("Table 1 compiled", comp, {"telecom": "43/43", "airline": "14/14", "retail": "16/16"})


def table_suite() -> None:
    """Table 2 (tab:suite)."""
    cases = rules.suite()
    crit = [c for s in cases for c in s["success_criteria"]]
    must = [c for c in crit if c["rule"] == "tool_called_with"]
    noarg = sum(not c["param"].get("arguments") for c in must)
    clusters = len({D.functional_key(c) for c in cases})
    rows = _rows()
    per_arm = Counter(d["arm"] for d in rows)
    actions = sum(m for d in rows if d["arm"] == "with_policy" for m in d["must"])
    got = {"scenarios": len(cases), "distinct tests": clusters, "criteria": len(crit),
           "requiring a call": len(must), "no argument values": noarg,
           "forbidding a call": len(crit) - len(must), "repeats": 5,
           "conversations per arm": per_arm["with_policy"], "graded actions per arm": actions}
    want = {"scenarios": 24, "distinct tests": 19, "criteria": 93, "requiring a call": 74,
            "no argument values": 48, "forbidding a call": 19, "repeats": 5,
            "conversations per arm": 120, "graded actions per arm": 370}
    print("\nTable 2 (The 24-scenario telecom suite)\n  " +
          ", ".join(f"{k} {v}" for k, v in got.items()))
    check("Table 2", got, want)
    check("Table 2 stub arm", per_arm["stub"], 120)


def table_rulecov() -> None:
    """Table 3 (tab:rulecov), from experiments/rule-coverage/coverage.csv."""
    by = defaultdict(Counter)
    with open(ROOT / "experiments/rule-coverage/coverage.csv") as f:
        for r in csv.DictReader(f):
            by[r["section"]][r["status"]] += 1
    tot = Counter()
    print("\nTable 3 (Policy rule coverage by policy section)")
    print(f"  {'section':40s} rules covered not_cov not_chk")
    for s, c in by.items():
        tot.update(c)
        print(f"  {s:40s} {sum(c.values()):5d} {c['covered']:7d} {c['not_covered']:7d} "
              f"{c['not_checkable']:7d}")
    print(f"  {'Total':40s} {sum(tot.values()):5d} {tot['covered']:7d} {tot['not_covered']:7d} "
          f"{tot['not_checkable']:7d}")
    check("Table 3 total", (sum(tot.values()), tot["covered"], tot["not_covered"],
                            tot["not_checkable"]), (51, 20, 20, 11))
    check("Table 3 per-section (rules, covered, not covered, not checkable)",
          sorted((sum(c.values()), c["covered"], c["not_covered"], c["not_checkable"])
                 for c in by.values()),
          sorted([(8, 2, 2, 4), (1, 0, 1, 0), (1, 0, 1, 0), (7, 2, 3, 2), (3, 2, 1, 0),
                  (4, 1, 1, 2), (6, 0, 3, 3), (2, 1, 1, 0), (9, 6, 3, 0), (5, 4, 1, 0),
                  (5, 2, 3, 0)]))


_ROWS: list[dict] = []


def _rows() -> list[dict]:
    if not _ROWS:
        run = ROOT / "results/full-fixed-2"
        m = json.loads((run / "manifest.json").read_text())
        _ROWS.extend(sc.load(run / "traces", m["case_ids"], m["repeats"]))
    return _ROWS


PERSCEN = [(18, 25, 11, 25), (16, 25, 9, 25), (5, 20, 0, 20), (11, 15, 8, 15), (15, 25, 5, 25),
           (12, 15, 7, 15), (10, 10, 7, 10), (5, 5, 3, 5), (20, 20, 12, 20), (8, 10, 4, 10),
           (5, 5, 5, 5), (16, 20, 12, 20), (5, 10, 2, 10), (15, 15, 5, 15), (5, 5, 2, 5),
           (20, 25, 13, 25), (15, 20, 7, 20), (9, 15, 5, 15), (9, 10, 7, 10), (4, 5, 5, 5),
           (20, 25, 14, 25), (15, 20, 9, 20), (15, 15, 8, 15), (0, 10, 2, 10)]
FIG_POINTS = [-20, -20, 0, 20, 23.33, 24, 25, 28, 28, 30, 30, 30, 33.33, 35, 40, 40, 46.67, 50,
              66.67]


def table_perscen_and_figure() -> None:
    """Table 4 (tab:perscen), Figure 2 (fig:perscen) and the Discrimination statistics."""
    per = defaultdict(lambda: [0, 0, 0, 0])
    for d in _rows():
        i = int(d["case"].split(":")[0])
        off = 0 if d["arm"] == "with_policy" else 2
        for ok, must in zip(d["ok"], d["must"]):
            if must:
                per[i][off] += ok
                per[i][off + 1] += 1
    got = [tuple(per[i]) for i in range(24)]
    print("\nTable 4 (Per-scenario results; # = suite index + 1)")
    for i, (a, b, c, e) in enumerate(got):
        print(f"  {i + 1:2d}  policy {a:2d}/{b:2d}  stub {c:2d}/{e:2d}  delta "
              f"{100 * (a / b - c / e):+6.1f}")
    pw = sum(x[0] for x in got), sum(x[1] for x in got)
    ps = sum(x[2] for x in got), sum(x[3] for x in got)
    print(f"  pooled policy {pw[0]}/{pw[1]} ({100 * pw[0] / pw[1]:.1f}%), stub {ps[0]}/{ps[1]} "
          f"({100 * ps[0] / ps[1]:.1f}%), separation {100 * (pw[0] / pw[1] - ps[0] / ps[1]):+.1f}")
    check("Table 4 rows", got, PERSCEN)
    check("Table 4 pooled", (pw, ps), ((273, 370), (162, 370)))
    sep = json.loads((ROOT / "experiments/duplicate-reanalysis/results.json").read_text())
    c19 = sep["separation"]["clustered_19"]
    pts = sorted(round(v, 2) for v in c19["per_unit_pp"].values())
    print("\nFigure 2 (Per-scenario separation on the 19 distinct scenarios), points sorted:")
    print("  " + ", ".join(f"{p:g}" for p in pts))
    print(f"Discrimination (Section 5.3): 19 clusters, bootstrap 95% CI "
          f"[{c19['ci95'][0]:.1f}, {c19['ci95'][1]:.1f}], ahead / behind / tied "
          f"{c19['better']} / {c19['worse']} / {c19['tied']}, sign-flip p (exact) "
          f"{c19['signflip_p_exact']:.5f}")
    check("Figure 2 points", pts, FIG_POINTS)
    check("CI", [round(x, 1) for x in c19["ci95"]], [24.7, 35.4])
    check("ahead/behind/tied", (c19["better"], c19["worse"], c19["tied"]), (16, 2, 1))
    check("sign-flip p < 0.001", c19["signflip_p_exact"] < 0.001, True)


KNOWN = {"R2_apn": (52, 7, 100.0, 65.4, 34.6, 0.008, True, -3.7, 0.249),
         "R6_transfer_scope": (63, 19, 69.8, 54.0, 15.9, 0.058, False, -0.6, 0.234),
         "R3_payment": (16, 3, 100.0, 75.0, 25.0, 0.250, False, -10.1, 0.008),
         "R5_escalate_unfixable": (26, 4, 61.5, 34.6, 26.9, 0.125, False, -1.8, 0.358),
         "R1_airplane": (28, 8, 85.7, 92.9, -7.1, 0.719, False, -1.4, 0.557),
         "R4_suspension": (24, 7, 100.0, 91.7, 8.3, 0.250, False, -3.2, 0.230)}
FOLLOW = {("primary", "R3_payment"): (60, 6, 96.7, 76.7, 20.0, 0.063, False),
          ("primary", "R5_escalate_unfixable"): (40, 8, 42.5, 37.5, 5.0, 0.422, False),
          ("pooled", "R3_payment"): (76, 9, 97.4, 76.3, 21.1, 0.016, True),
          ("pooled", "R5_escalate_unfixable"): (61, 12, 49.2, 34.4, 14.8, 0.088, False)}


def _row(t: dict) -> tuple:
    return (t["n_control"], t["n_scenarios"], round(t["control_pct"], 1),
            round(t["variant_pct"], 1), round(t["drop"], 1), t["p"], bool(t["caught"]))


def _same_row(what: str, got: tuple, want: tuple) -> None:
    for k, (g, w) in enumerate(zip(got, want)):
        check(f"{what} col {k}", g, w, tol=0.0005 if k == 5 or k == 8 else 0.05)


def table_known_and_followup() -> None:
    """Table 5 (tab:known) and Table 6 (tab:followup)."""
    r = json.loads((ROOT / "experiments/duplicate-reanalysis/results.json").read_text())
    main19 = r["ablation_main"]["clustered_19"]
    print("\nTable 5 (Catching removed rules; 63 valid blocks, 19 distinct scenarios)")
    print(f"  {'rule':22s} checks(scen) ctrl%  var%   drop   p      caught untied(p)")
    caught = 0
    for rule, want in KNOWN.items():
        t, u = main19[rule]["tied"], main19[rule]["untied"]
        got = _row(t) + (round(-u["drop"], 1), u["p"])
        caught += got[6]
        print(f"  {rule:22s} {got[0]:3d} ({got[1]:2d})    {got[2]:5.1f} {got[3]:5.1f} "
              f"{got[4]:+6.1f} {got[5]:.4f} {'yes' if got[6] else 'no ':3s}    "
              f"{got[7]:+.1f} ({got[8]:.3f})")
        _same_row(f"Table 5 {rule}", got, want)
    off = sum(1 for rl in KNOWN if -main19[rl]["untied"]["drop"] <= -20
              and main19[rl]["untied"]["p"] < 0.05)
    print(f"  caught {caught} of 6; off-target untied drops >= 20 with p < 0.05: {off} of 6")
    check("Table 5 caught", caught, 1)
    check("Table 5 off-target", off, 0)
    print("\nTable 6 (Pre-registered follow-up and pooled secondary analysis)")
    src = {"primary": r["followup_primary"], "pooled": r["pooled_secondary"]}
    for (part, rule), want in FOLLOW.items():
        got = _row(src[part][rule]["tied"])
        print(f"  {part:8s} {rule:22s} {got[0]:3d} ({got[1]:2d}) {got[2]:5.1f} {got[3]:5.1f} "
              f"{got[4]:+6.1f} {got[5]:.4f} {'meets' if got[6] else 'no'}")
        _same_row(f"Table 6 {part} {rule}", got, want)


def records() -> None:
    """Sections 5.1, 5.2 and 5.5 and Table 7 (tab:rel), from records/ (repair_coverage.py, determinism.py, instability.py)."""
    r = COV.compute()
    print(f"\nSection 5.2: coverage {r['covered']}/{r['denominator']}, repair-graded "
          f"{r['repair-graded covered']}/{r['repair-graded']}, fault types {r['complete']} complete + "
          f"{r['partial']} partial of {r['agent-repairable']}, {r['emitted']:,} emitted")
    BAD.extend(f"Section 5.2 {b}" for b in COV.check())
    print("Section 5.1: determinism " + ", ".join(
        f"{d} {x['tasks']}x{x['repeats']}" for d, (x, _) in DET.load().items()))
    BAD.extend(f"Section 5.1 {b}" for b in DET.check())
    t = INST.compute()
    print("Table 7 (Verdict instability): " + "; ".join(
        f"{d} {g['tasks']} tasks {g['runs']} runs, {g['flip_pct']:.1f}% flip, drop {g['drop']:.1f}"
        for d, g in t.items()))
    BAD.extend(f"Table 7 {b}" for b in INST.check())


def main() -> int:
    table_domains()
    table_suite()
    table_rulecov()
    table_perscen_and_figure()
    table_known_and_followup()
    records()
    print()
    for b in BAD:
        print("MISMATCH", b)
    print("paper tables: all values match the paper" if not BAD else
          f"paper tables: {len(BAD)} mismatch(es)")
    return 1 if BAD else 0


if __name__ == "__main__":
    sys.exit(main())
