#!/usr/bin/env python3
"""Write the control policy, the six variants, their diffs, variants.json and ties.json.

    PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/rule-ablation/make_variants.py

The control text is env.get_policy() from code-fixed's vendored tau3, the exact text
bench.build_agent(domain, None) gives the agent (code-fixed/scripts/bench.py:59-68).
"""
from __future__ import annotations

import difflib
import json
import sys

sys.dont_write_bytecode = True
from rules import HERE, PROJECT, RULES, CONTROL_SHA, remove, sha, ties  # noqa: E402


def control_policy() -> str:
    sys.path.insert(0, str(PROJECT / "code-fixed" / "third_party" / "tau3-bench" / "src"))
    from tau2.domains.telecom import environment
    return environment.get_environment().get_policy()


def main() -> int:
    policy = control_policy()
    if sha(policy) != CONTROL_SHA:
        raise SystemExit(f"control policy hash {sha(policy)} != {CONTROL_SHA}")
    pol = HERE / "policies"
    pol.mkdir(exist_ok=True)
    (pol / "control.md").write_text(policy)
    meta = {"control": {"file": "policies/control.md", "sha256": sha(policy),
                        "chars": len(policy)}}
    for rule in RULES:
        text = remove(policy, rule)
        (pol / f"{rule}.md").write_text(text)
        diff = "".join(difflib.unified_diff(policy.splitlines(True), text.splitlines(True),
                                            "control.md", f"{rule}.md", n=1))
        (pol / f"{rule}.diff").write_text(diff)
        meta[rule] = {"file": f"policies/{rule}.md", "sha256": sha(text), "chars": len(text),
                      "chars_removed": len(policy) - len(text),
                      "aspect": RULES[rule]["aspect"]}
    (HERE / "variants.json").write_text(json.dumps(meta, indent=1) + "\n")
    (HERE / "ties.json").write_text(json.dumps(ties(), indent=1) + "\n")
    for k, v in meta.items():
        print(f"{k:24s} {v['sha256'][:12]} chars {v['chars']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
