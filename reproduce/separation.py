#!/usr/bin/env python3
"""Recompute the +30.0 separation (and the original-code run) from the stored traces. No model calls.

    python -B reproduce/separation.py

For each of results/full-fixed-2 and results/full-original-2 this runs scripts/score.py's own
checks and scoring unchanged: the seal is in git, the seal matches every file, the manifest is
the registered Job 1 run (24 cases x 5 repeats x 2 arms, gpt-4.1, temperatures 0 / 0.7, cap 30),
its recorded command and wrapper version are right, every trace matches the suite, bench's own
report agrees, and the wrapper's completeness check still gives the manifest's summary. The
score is computed on a temporary copy (score.py deletes <run>/score.json before scoring) and
must equal the stored score.json exactly where one was stored (full-fixed-2).

Two substitutions, both forced by leaving the private development tree:
  - the code-copy checksum and tree hash are compared with the values committed alongside the
    run (reproduce/private_commits.json), because they hash the full private code copy, .env
    included, and the tree-*.txt files here hold later values (re-recorded after the .env
    API-key line changed);
  - the recorded command and working directory are compared under the project root the run
    was recorded in, and the git ancestry checks go through reproduce/offline_git.py.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import offline_git  # noqa: E402

offline_git.install()
import run_manifest as rm  # noqa: E402
import score as sc  # noqa: E402

RUNS = ("full-fixed-2", "full-original-2")


def one(name: str) -> dict:
    run = offline_git.ROOT / "results" / name
    seal = rm.seal_path(run)
    sc.committed(seal)
    m = json.loads((run / "manifest.json").read_text())
    recorded = offline_git.RECORD[m["project_commit"]]["stored"]
    real_stored, here, project = rm.stored, rm.HERE, rm.PROJECT
    rm.stored = lambda kind, copy: recorded.get(f"{kind}-{copy.name.removeprefix('code-')}")
    try:
        sc.check_manifest(run, m, job1=True)
        rm.PROJECT = Path(m["cwd"]).parent
        rm.HERE = rm.PROJECT / "scripts"
        try:
            sc.check_provenance(run, m)
        finally:
            rm.HERE, rm.PROJECT = here, project
        stored_file = run / "score.json"
        stored = json.loads(stored_file.read_text()) if stored_file.exists() else None
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp) / "results" / name
            shutil.copytree(run, copy)
            s = sc.score_run(copy, seal=seal, job1=False)
    finally:
        rm.stored = real_stored
    if stored is not None and json.loads(json.dumps(s)) != stored:
        raise SystemExit(f"{name}: recomputed score differs from the stored score.json")
    s["matched_stored"] = stored is not None
    return s


def main() -> int:
    for name in RUNS:
        s = one(name)
        print(sc.show(s))
        print("  identical to the stored score.json" if s["matched_stored"] else
              "  (no score.json was stored for this run)",
              f"; seal sha256 {s['seal_sha256'][:16]}...", sep="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
