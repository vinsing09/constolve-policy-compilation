#!/usr/bin/env python3
"""Bind the vendored τ³ to upstream: compare the vendored tree with upstream git's tree.

    .venv/bin/python -B scripts/tau3_provenance.py            # writes scripts/tau3-provenance.json
    .venv/bin/python -B scripts/tau3_provenance.py --check    # exit 1 if it no longer matches

The record lists the upstream commit, the vendored tree hash, every upstream path the
vendor omits, every vendor path upstream lacks, every shared file whose bytes or
executable bit differ, and the exact unified diff of changed files
(scripts/tau3_vendor.patch). It also states whether the runtime subset (src/ and
data/tau2/domains/) is complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from runlog import tree_hash  # noqa: E402

TAU3 = HERE.parent / "code-original" / "third_party" / "tau3-bench"
UPSTREAM = Path.home() / "Documents" / "my_projects" / "tau2-bench-src"
COMMIT = "6e1d57640fb25df43c3d8049abeefade6275c6f0"
RECORD, PATCH = HERE / "tau3-provenance.json", HERE / "tau3_vendor.patch"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def upstream(path: str) -> bytes | None:
    p = subprocess.run(["git", "-C", str(UPSTREAM), "show", f"{COMMIT}:{path}"],
                       capture_output=True, stdin=subprocess.DEVNULL)
    return p.stdout if p.returncode == 0 else None


RUNTIME = ("src/", "data/tau2/domains/")


def upstream_tree() -> dict[str, tuple[str, str]]:
    """path -> (git mode, git blob id) at COMMIT."""
    out = subprocess.run(["git", "-C", str(UPSTREAM), "ls-tree", "-r", COMMIT],
                         capture_output=True, text=True, check=True).stdout
    return {line.split("\t", 1)[1]: tuple(line.split()[0:3:2]) for line in out.splitlines()}


def blob_id(b: bytes) -> str:
    """git's object id for file bytes, so unchanged files need no `git show`."""
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def build() -> tuple[dict, str]:
    files = sorted(p.relative_to(TAU3).as_posix() for p in TAU3.rglob("*")
                   if p.is_file() or p.is_symlink())
    tree = upstream_tree()
    differ, extra, modes, diff = {}, [], {}, ""
    for f in files:
        if f not in tree:
            extra.append(f)
            continue
        mode, blob = tree[f]
        if (TAU3 / f).is_symlink() or mode == "120000":
            if not ((TAU3 / f).is_symlink() and mode == "120000" and
                    blob_id(os.readlink(TAU3 / f).encode()) == blob):
                differ[f] = {"upstream_mode": mode, "vendor_symlink": (TAU3 / f).is_symlink()}
            continue
        if (mode == "100755") != os.access(TAU3 / f, os.X_OK):
            modes[f] = {"upstream_mode": mode, "vendor_executable": os.access(TAU3 / f, os.X_OK)}
        mine = (TAU3 / f).read_bytes()
        if blob_id(mine) != blob:
            theirs = upstream(f)
            differ[f] = {"upstream_sha256": sha(theirs), "vendor_sha256": sha(mine)}
            diff += subprocess.run(
                ["git", "-C", str(UPSTREAM), "diff", "--no-index", "--no-color",
                 f"--src-prefix=a/", f"--dst-prefix=b/", "-", str(TAU3 / f)],
                input=theirs, capture_output=True).stdout.decode().replace(
                f"b{TAU3 / f}", f"b/{f}").replace("a/-", f"a/{f}")
    omitted = sorted(set(tree) - set(files))
    record = {"upstream_commit": COMMIT, "vendor_tree_hash": tree_hash(TAU3),
              "upstream_paths": len(tree), "vendor_paths": len(files),
              "identical_to_upstream": len(files) - len(differ) - len(extra),
              "differ": differ, "not_in_upstream": extra, "mode_differ": modes,
              "omitted_from_vendor": omitted,
              "runtime_subset": list(RUNTIME),
              "runtime_subset_complete": not [p for p in omitted if p.startswith(RUNTIME)],
              "patch_sha256": sha(diff.encode())}
    return record, diff


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    record, diff = build()
    if a.check:
        ok = RECORD.exists() and json.loads(RECORD.read_text()) == record and \
            PATCH.exists() and PATCH.read_text() == diff
        print("tau3 provenance", "matches" if ok else "DOES NOT MATCH")
        return 0 if ok else 1
    RECORD.write_text(json.dumps(record, indent=1) + "\n")
    PATCH.write_text(diff)
    print(json.dumps({k: (v if k != "omitted_from_vendor" else len(v)) for k, v in
                      record.items() if k != "differ"}, indent=1))
    print("differ:", list(record["differ"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
