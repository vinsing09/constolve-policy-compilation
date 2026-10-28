#!/usr/bin/env bash
# Every model-free reproduction, from a clean checkout. Fails on the first difference.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
PY=${PYTHON:-python3}
out=$(mktemp -d)
git diff --quiet || { echo "working tree not clean; run from a clean checkout"; exit 1; }

echo "== 0. secret and personal-data scan"
$PY -B reproduce/secret_scan.py

echo; echo "== 1. +30.0 separation, and the original-code run, from stored traces"
$PY -B reproduce/separation.py

echo; echo "== 2. scenario-cluster bootstrap and sign-flip test for +30.0 (stats/cluster30.py)"
$PY -B stats/cluster30.py "$out/cluster30.csv"

echo; echo "== 3. deep-patch result on the 51 unseen tasks (stats/cluster20.py)"
$PY -B stats/cluster20.py "$out/cluster20.csv"

echo; echo "== 4. rule removal: main run and registered follow-up"
$PY -B reproduce/ablation.py

echo; echo "== 5. rule coverage (rewrites coverage.csv and coverage.md; must be unchanged)"
$PY -B experiments/rule-coverage/coverage.py
git diff --exit-code -- experiments/rule-coverage && echo "coverage.csv and coverage.md unchanged"

echo; echo "== 6. duplicate re-analysis (rewrites results.json; must be unchanged)"
$PY -B reproduce/reanalysis.py
git diff --exit-code -- experiments/duplicate-reanalysis && echo "results.json unchanged"

echo; echo "== 7. fault types and repair-call-set coverage (105/113, 90/94, 17/17) from records/coverage"
$PY -B reproduce/repair_coverage.py

echo; echo "== 8. determinism replays: archived reports (summaries only) in records/determinism"
$PY -B reproduce/determinism.py

echo; echo "== 9. verdict instability, Table 7, from the 834 per-run verdicts in records/verdict-instability"
$PY -B reproduce/instability.py

echo; echo "== 10. the record checks fail on altered records"
$PY -B reproduce/tests/test_records.py 2>&1 | tail -1

echo; echo "== 11. the paper's Tables 1-7, Figure 2 and Sections 5.1-5.2, checked against the printed values"
$PY -B reproduce/paper_tables.py

echo; echo "== 12. engine replay smoke test (needs requirements.txt; skipped otherwise)"
if $PY -c "import litellm, loguru, deepdiff, addict, toml" 2>/dev/null; then
  $PY -B reproduce/engine_replay.py 2>/dev/null | tail -1
else
  echo "skipped: litellm and the tau2 dependencies are not installed for $PY"
fi

echo; echo "ALL REPRODUCTIONS MATCH"
