"""Stand-in for the scorers' git provenance checks, for use outside the private development repo.

The scorers (scripts/score.py, experiments/*/score_*.py) refuse a run unless the commit that
recorded it is an ancestor of HEAD and the files it pinned are unchanged since. Those commits
live in the private development repository, not here. private_commits.json records, for each
run's recording commit, the sha256 of every pinned file at that commit (taken with `git show`
when this repository was built). This module replaces score.git_ok so that:

  cat-file -e C^{commit}, merge-base --is-ancestor C HEAD  ->  C is a recorded commit
  diff --quiet C HEAD -- PATHS  ->  every file under PATHS here is byte-identical to C's,
                                    with none missing and none added

Any other git call returns False. Everything else the scorers check (seals, manifests, the
traces against the suite, the run's own completeness checks) runs unchanged.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = json.loads((Path(__file__).with_name("private_commits.json")).read_text())


def _under(f: str, paths: list[str]) -> bool:
    return any(f == p or f.startswith(p.rstrip("/") + "/") for p in paths)


def _same(commit: str, paths: list[str]) -> bool:
    files = RECORD[commit]["files"]
    want = {f: h for f, h in files.items() if _under(f, paths)}
    if not paths or any(not any(_under(f, [p]) for f in want) for p in paths):
        return False
    here = set()
    for p in paths:
        q = ROOT / p
        here |= {x.relative_to(ROOT).as_posix() for x in (q.rglob("*") if q.is_dir() else [q])
                 if x.is_file()}
    return here == set(want) and all(
        hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == h for f, h in want.items())


def git_ok(*args: str) -> bool:
    a = list(args)
    if len(a) == 3 and a[:2] == ["cat-file", "-e"] and a[2].endswith("^{commit}"):
        return a[2][:-len("^{commit}")] in RECORD
    if len(a) == 4 and a[:2] == ["merge-base", "--is-ancestor"] and a[3] == "HEAD":
        return a[2] in RECORD
    if len(a) >= 6 and a[:2] == ["diff", "--quiet"] and a[3:5] == ["HEAD", "--"]:
        return a[2] in RECORD and _same(a[2], a[5:])
    return False


def install() -> None:
    """Put the research code on sys.path and swap score.git_ok for the recorded check."""
    for p in (ROOT / "scripts", ROOT / "experiments" / "rule-ablation",
              ROOT / "experiments" / "rule-ablation-followup"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    sys.dont_write_bytecode = True
    import score
    score.git_ok = git_ok
