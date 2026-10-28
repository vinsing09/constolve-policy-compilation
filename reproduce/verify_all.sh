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

echo; echo "== 7. engine replay smoke test (needs requirements.txt; skipped otherwise)"
if $PY -c "import litellm, loguru, deepdiff, addict, toml" 2>/dev/null; then
  $PY -B reproduce/engine_replay.py 2>/dev/null | tail -1
else
  echo "skipped: litellm and the tau2 dependencies are not installed for $PY"
fi

echo; echo "ALL REPRODUCTIONS MATCH"
