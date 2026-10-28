#!/usr/bin/env python3
"""Scan every file in the repository (including inside .xz archives) for credentials and personal data.

    python -B reproduce/secret_scan.py [root]

Exit 1 if any of these is found:
  - a credential-shaped value (OpenAI/Anthropic/GitHub/AWS/Google/Slack/Hugging Face tokens,
    private-key blocks, KEY/SECRET/TOKEN assignments with a long literal value);
  - a credential keyword (api_key, apikey, api-key, OPENAI_API_KEY=, ...) outside the
    reviewed allowlist below (upstream tau3-bench source that names environment variables,
    test fixtures that pass a deliberately invalid key, and empty placeholders);
  - an email address other than the simulated customers' (benchmark world data, and one address
    the simulated customer made up) or upstream tau3-bench's authors and examples, or a forbidden file (.env, *.lock,
    __pycache__, .venv, knowledge.md, errors.md, code_review/).
"""
from __future__ import annotations

import lzma
import re
import sys
from pathlib import Path

VALUE = re.compile(
    r"sk-(?!tripwire-invalid)[A-Za-z0-9_-]{20,}|sk-proj-|sk-ant-|gh[pousr]_[A-Za-z0-9]{36}|"
    r"github_pat_|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}(?![a-z])|AIza[0-9A-Za-z_-]{35}|xox[abpr]-|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY|hf_[A-Za-z0-9]{30,}|"
    r"(?i:(?:api_?key|secret|token|password)\"?\s*[=:]\s*[\"']?)[A-Za-z0-9_\-]{24,}")
KEYWORD = re.compile(r"api_key|apikey|api-key|[A-Z_]*API_KEY\s*=|aws_secret|AWS_BEARER_TOKEN",
                     re.I)
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
EMAIL_OK = re.compile(r"@(example\.(com|org)|anthropic\.com|users\.noreply\.github\.com)$|"
                      r"^noreply@|^git@github\.com$|"
                      r"^(sofia\.martinez@(fastmail|email)\.com|ethan\.brooks@mailbox\.com)$")
EMAIL_OK_PREFIX = "code-fixed/third_party/tau3-bench/"   # upstream authors and examples
KEYWORD_OK = ("code-fixed/third_party/tau3-bench/src/",
              "code-fixed/third_party/tau3-bench/README.md")
KEYWORD_OK_LINES = {
    ("experiments/duplicate-reanalysis/tests/test_reanalyse.py", "OPENAI_API_KEY=sk-tripwire-invalid"),
    ("experiments/rule-ablation-followup/tests/test_followup.py", 'OPENAI_API_KEY="sk-tripwire-invalid"'),
    ("scripts/tests/test_run_manifest.py", 'api_key="test"'),
    ("reproduce/secret_scan.py", ""),
    ("reproduce/separation.py", "API-key line changed"),
    ("README.md", "# then set OPENAI_API_KEY"),
    ("README.md", "your own OPENAI_API_KEY"),
    ("code-fixed/.env.example", "OPENAI_API_KEY="),
    ("reproduce/engine_replay.py", "OPENAI_API_KEY=not-a-key"),
    ("experiments/rule-ablation/rules.py", "the .env API-key line changed"),
    ("experiments/rule-ablation/prereg.md", "Registered engine hashes are literals"),
    ("experiments/rule-ablation-followup/prereg.md", "Narrator model is gpt-4.1"),
}
FORBIDDEN = re.compile(r"(^|/)(\.env|[^/]*\.lock|__pycache__|\.venv|knowledge\.md|errors\.md|"
                       r"code_review)(/|$)")


def text(p: Path) -> str | None:
    data = lzma.open(p).read() if p.suffix == ".xz" else p.read_bytes()
    if b"\0" in data[:4096]:
        return None
    return data.decode("utf-8", "replace")


def scan(root: Path) -> list[str]:
    bad = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root).as_posix()
        if rel.startswith(".git/") or rel == ".git":
            continue
        if FORBIDDEN.search(rel):
            bad.append(f"forbidden file: {rel}")
        if not p.is_file() or p.is_symlink():
            continue
        t = text(p)
        if t is None:
            continue
        for n, line in enumerate(t.splitlines(), 1):
            for m in VALUE.finditer(line):
                if rel != "reproduce/secret_scan.py":
                    bad.append(f"credential-shaped value: {rel}:{n}: {m.group()[:12]}...")
            if KEYWORD.search(line) and not rel.startswith(KEYWORD_OK) and not any(
                    rel == f and s in line for f, s in KEYWORD_OK_LINES):
                bad.append(f"credential keyword: {rel}:{n}: {line.strip()[:100]}")
            for m in EMAIL.finditer(line):
                if not EMAIL_OK.search(m.group()) and not rel.startswith(EMAIL_OK_PREFIX) \
                        and rel != "reproduce/secret_scan.py":
                    bad.append(f"email address: {rel}:{n}: {m.group()}")
    return bad


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
    bad = scan(root.resolve())
    for b in bad:
        print(b)
    print(f"secret scan: {len(bad)} finding(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
