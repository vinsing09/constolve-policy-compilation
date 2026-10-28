#!/usr/bin/env python3
"""Re-score the 24-scenario telecom results with functional duplicates merged (no model calls).

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/duplicate-reanalysis/reanalyse.py

Unit = functionally distinct scenario (design.functional_key: kind, world, graded criteria).
Both members of a duplicate pair are pooled into one cluster, their repeats summed. Writes
results.json next to this file.
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
for p in (PROJECT / "scripts", PROJECT / "experiments" / "rule-ablation",
          PROJECT / "experiments" / "rule-ablation-followup"):
    sys.path.insert(0, str(p))
import design as D  # noqa: E402
import rules  # noqa: E402
import score as sc  # noqa: E402
import score_ablation as sa  # noqa: E402
import score_followup as sf  # noqa: E402

SEED, BOOT_N = 20261027, 20000
FIELDS = ("kind", "world_setup", "symptoms", "reference_solution", "success_criteria",
          "acceptance", "direct_state", "known_facts", "caller", "wont_volunteer",
          "will_reveal_if_asked")


def clusters(cases: list[dict]) -> list[int]:
    """Scenario index -> cluster id (lowest index sharing its functional key)."""
    first: dict[str, int] = {}
    return [first.setdefault(D.functional_key(c), i) for i, c in enumerate(cases)]


def pair_diffs(cases: list[dict], cl: list[int]) -> dict:
    """For each merged pair: which top-level fields differ (FIELDS must not)."""
    out = {}
    for i, c in enumerate(cl):
        if c != i:
            a, b = cases[c], cases[i]
            diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            if set(diff) & set(FIELDS):
                raise AssertionError(f"pair {c}={i} differs in {set(diff) & set(FIELDS)}")
            ra = a["scenario"]["instructions"]["reason_for_call"]
            out[f"{c}={i}"] = {"fields_differing": diff,
                               "opening_line_same": ra == b["scenario"]["instructions"]
                               ["reason_for_call"]}
    return out


def exact_signflip(d: list[float]) -> float:
    d = [x for x in d if abs(x) > 1e-12]
    obs, dist = sum(d), {0.0: 1}
    for x in d:
        nxt = defaultdict(int)
        for s, n in dist.items():
            nxt[round(s + x, 9)] += n
            nxt[round(s - x, 9)] += n
        dist = nxt
    return sum(n for s, n in dist.items() if s >= round(obs, 9) - 1e-9) / 2 ** len(d)


def separation(cl: list[int] | None) -> dict:
    """cluster30.py's +30.0 statistics; cl=None keeps the 24 scenarios as units."""
    run = PROJECT / "results" / "full-fixed-2"
    m = json.loads((run / "manifest.json").read_text())
    want_ids = [f"{i}:{c.get('source_task_id', '')}"[:120] for i, c in enumerate(rules.suite())]
    if m["case_ids"] != want_ids:
        raise AssertionError("full-fixed-2 cases are not the 24-scenario suite")
    rows = sc.load(run / "traces", m["case_ids"], m["repeats"])
    assert len(rows) == 240
    gap0 = sc.compare(rows)["gap"]["action"]
    ids = m["case_ids"]
    unit = (lambda c: c) if cl is None else (lambda c: ids[cl[ids.index(c)]])
    w, s = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for d in rows:
        t = (w if d["arm"] == "with_policy" else s if d["arm"] == "stub" else None)
        if t is None:
            continue
        a = t[unit(d["case"])]
        for ok, must in zip(d["ok"], d["must"]):
            if must:
                a[0] += ok
                a[1] += 1
    units = sorted(w)

    def gap(cs):
        return 100 * (sum(w[c][0] for c in cs) / sum(w[c][1] for c in cs)
                      - sum(s[c][0] for c in cs) / sum(s[c][1] for c in cs))
    rng = random.Random(SEED)
    boot = sorted(gap([rng.choice(units) for _ in units]) for _ in range(BOOT_N))
    d = {c: w[c][0] / w[c][1] - s[c][0] / s[c][1] for c in units}
    obs = sum(d.values())
    hits = sum(sum(v * rng.choice((1, -1)) for v in d.values()) >= obs - 1e-12
               for _ in range(BOOT_N))
    pos, neg = sum(v > 0 for v in d.values()), sum(v < 0 for v in d.values())
    return {"units": len(units), "pooled_gap": round(gap0, 2), "gap_over_units": gap(units),
            "ci95": [boot[int(.025 * BOOT_N) - 1], boot[int(.975 * BOOT_N) - 1]],
            "signflip_p_mc": hits / BOOT_N, "signflip_p_exact": exact_signflip(list(d.values())),
            "per_unit_pp": {c.split(":")[0]: round(100 * v, 2) for c, v in d.items()},
            "better": pos, "worse": neg, "tied": len(units) - pos - neg}


def ablation(cl: list[int], rows: list[dict], ties: dict) -> dict:
    """score_ablation.score with scenarios mapped to clusters; adds the minimum possible p."""
    rows = [dict(x, scen=cl[x["scen"]]) for x in rows]
    ties = {r: sorted({(cl[i], c) for i, c in v}) for r, v in ties.items()}
    out = sa.score(rows, ties)
    for r in rules.RULES:
        t = out[r]["tied"]
        t["clusters_with_ties"] = len({i for i, _ in ties[r]})
        t["min_possible_p"] = 0.5 ** t["clusters_with_ties"]
        t["changed_clusters"] = _changed(rows, r, set(map(tuple, ties[r])))
    return out


def _changed(rows, arm, keys) -> int:
    per = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for x in rows:
        if x["arm"] in ("control", arm) and (x["scen"], x["cid"]) in keys:
            per[x["scen"]][x["arm"]][0] += x["ok"]
            per[x["scen"]][x["arm"]][1] += 1
    return sum(abs(v["control"][0] / v["control"][1] - v[arm][0] / v[arm][1]) > 1e-12
               for v in per.values())


def loose_key(case: dict) -> str:
    """functional_key without criterion ids or order (a stricter duplicate test)."""
    crits = sorted(json.dumps({k: v for k, v in c.items() if k not in ("id", "description")},
                              sort_keys=True).lower() for c in case["success_criteria"])
    return json.dumps([case.get("kind") or "repair",
                       json.dumps(case["world_setup"], sort_keys=True).lower(), crits])


def pooled_loose(main_rows, fu_rows, fu_tied, main_ties) -> dict:
    """sf.pooled with loose_key as the unit; criteria matched by tool, not id."""
    s0, s1 = rules.suite(), sf.suite()

    def crit_key(case, cid):
        c = next(x for x in case["success_criteria"] if x["id"] == cid)
        return json.dumps({k: v for k, v in c.items() if k not in ("id", "description")},
                          sort_keys=True).lower()
    rows = [dict(x, scen=loose_key(s0[x["scen"]]), cid=crit_key(s0[x["scen"]], x["cid"]))
            for x in main_rows] + \
        [dict(x, scen=loose_key(s1[x["scen"]]), cid=crit_key(s1[x["scen"]], x["cid"]))
         for x in fu_rows]
    tied = {r: {(loose_key(s0[i]), crit_key(s0[i], c)) for i, c in main_ties[r]} |
            {(loose_key(s1[i]), crit_key(s1[i], c)) for i, c in fu_tied[r]} for r in D.RULES}
    return sf.score(rows, tied)


def overlap(main_cases: list[dict], fu_cases: list[dict]) -> dict:
    k0 = {D.functional_key(c) for c in main_cases}
    k1 = [D.functional_key(c) for c in fu_cases]
    w0 = {json.dumps(c["world_setup"], sort_keys=True).lower() for c in main_cases}
    return {"followup_scenarios": len(fu_cases), "distinct_followup_keys": len(set(k1)),
            "functional_key_overlap": len(set(k1) & k0),
            "loose_key_overlap": [[i, j] for i, a in enumerate(main_cases)
                                  for j, b in enumerate(fu_cases) if loose_key(a) == loose_key(b)],
            "same_world_as_a_suite_scenario": sum(json.dumps(c["world_setup"], sort_keys=True)
                                                  .lower() in w0 for c in fu_cases)}


def main() -> int:
    cases = rules.suite()
    cl = clusters(cases)
    reg = json.loads((rules.HERE / "results" / "main" / "score.json").read_text())["rules"]
    _, main_rows, n_valid = sa.load(sf.MAIN_RUN)
    ties = json.loads((rules.HERE / "ties.json").read_text())
    ident = ablation(list(range(len(cases))), main_rows, ties)
    for r in rules.RULES:
        for k in ("tied", "untied"):
            assert abs(ident[r][k]["p"] - reg[r][k]["p"]) < 1e-12, (r, k)
            assert abs(ident[r][k]["drop"] - reg[r][k]["drop"]) < 1e-9, (r, k)
    _, fu_rows, _ = sf.load(PROJECT / "experiments/rule-ablation-followup/results/main")
    fu_tied = sf.ties()
    res = {"clusters": len(set(cl)), "pairs": pair_diffs(cases, cl),
           "separation": {"original_24": separation(None), "clustered_19": separation(cl)},
           "ablation_main": {"valid_blocks": n_valid, "original_24": ident,
                             "clustered_19": ablation(cl, main_rows, ties)},
           "followup_overlap": overlap(cases, sf.suite()),
           "followup_primary": sf.score(fu_rows, fu_tied),
           "pooled_secondary": sf.pooled(main_rows, fu_rows, fu_tied),
           "pooled_secondary_loose": pooled_loose(main_rows, fu_rows, fu_tied,
                                                  sf.ties(rules.suite()))}
    (HERE / "results.json").write_text(json.dumps(res, indent=1, default=list) + "\n")
    print(json.dumps({"clusters": res["clusters"], "pairs": res["pairs"]}, indent=1))
    for k, v in res["separation"].items():
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
    for k in ("original_24", "clustered_19"):
        print(k)
        print(sa.table(res["ablation_main"][k]))
        for r in rules.RULES:
            t = res["ablation_main"][k][r]["tied"]
            print(f"  {r}: clusters {t['clusters_with_ties']} changed {t['changed_clusters']}"
                  f" min p {t['min_possible_p']:.4f}")
        for v, o in res["ablation_main"][k]["pair"].items():
            print(f"  pair {v}: own {o['own_only']['drop']:+.1f} p={o['own_only']['p']:.4f}; "
                  f"partner {o['partner_only']['drop']:+.1f} p={o['partner_only']['p']:.4f}; "
                  f"right rule {o['lands_on_right_rule']}")
    print("overlap", res["followup_overlap"])
    print("follow-up primary\n" + sa.table(res["followup_primary"]))
    print("pooled secondary\n" + sa.table(res["pooled_secondary"]))
    print("pooled secondary, loose key\n" + sa.table(res["pooled_secondary_loose"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
