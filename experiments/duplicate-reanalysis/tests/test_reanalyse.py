"""Tests on the real suite and the real sealed runs (no model calls).

    OPENAI_API_KEY=sk-tripwire-invalid PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B -m pytest \
        -p no:cacheprovider -q experiments/duplicate-reanalysis/tests
"""
import copy
import json
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reanalyse as R  # noqa: E402

CASES = R.rules.suite()
CL = R.clusters(CASES)


def test_exactly_five_pairs_merge():
    merged = sorted((c, i) for i, c in enumerate(CL) if c != i)
    assert merged == [(0, 15), (3, 17), (6, 9), (7, 14), (8, 11)]
    assert len(set(CL)) == 19


def test_pairs_share_every_functional_field():
    for c, i in [(0, 15), (3, 17), (6, 9), (7, 14), (8, 11)]:
        for f in R.FIELDS:
            assert CASES[c].get(f) == CASES[i].get(f), (c, i, f)


def test_exact_signflip_matches_registered_scorer():
    rng = random.Random(1)
    for _ in range(30):
        d = [rng.choice([0, 1 / 3, 2 / 3, 1, -1 / 3, -1 / 2]) for _ in range(rng.randint(1, 14))]
        assert abs(R.exact_signflip(d) - R.sa.signflip_p(d)) < 1e-12


def test_separation_unclustered_reproduces_cluster30():
    s = R.separation(None)
    assert s["units"] == 24 and (s["better"], s["worse"], s["tied"]) == (21, 2, 1)
    assert [round(x, 1) for x in s["ci95"]] == [24.6, 35.3]


def test_clustering_keeps_every_graded_criterion():
    _, rows, _ = R.sa.load(R.sf.MAIN_RUN)
    ties = json.loads((R.rules.HERE / "ties.json").read_text())
    a = R.ablation(list(range(24)), rows, ties)
    b = R.ablation(CL, rows, ties)
    for r in R.rules.RULES:
        assert a[r]["tied"]["n_control"] == b[r]["tied"]["n_control"]
        assert abs(a[r]["tied"]["drop"] - b[r]["tied"]["drop"]) < 1e-9
        assert b[r]["tied"]["n_scenarios"] == len({CL[i] for i, _ in ties[r]})


def test_loose_key_ignores_order_but_not_content():
    c = copy.deepcopy(CASES[16])
    c["success_criteria"] = list(reversed(c["success_criteria"]))
    for k in c["success_criteria"]:
        k["id"] = "x" + k["id"]
    assert R.loose_key(c) == R.loose_key(CASES[16])
    c["success_criteria"][0]["param"] = {"tool": "make_payment"}
    assert R.loose_key(c) != R.loose_key(CASES[16])
