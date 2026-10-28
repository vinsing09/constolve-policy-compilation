"""Tolerant decoding of JSON that came out of a language model.

WHY THIS EXISTS

Twenty modules call `json.loads` directly on model output, each with its own fence
stripping and its own bare `except`. On 2026-08-15 the transaction axis lost 37 of 87
designs — 43% — to `JSONDecodeError`, reported as one line naming the exception class
and nothing else. The payload was never kept, so the failures were uncountable and
undiagnosable; they simply lowered the yield of every axis by an unknown amount.

The errors are not random. `Expecting ',' delimiter` in the middle of an object means a
string terminated early, which is what an unescaped `"` inside a value does — and these
prompts ask the model to write customer utterances, which is exactly where quotes and
apostrophes appear. `Invalid control character` is the same story with a raw newline.

WHAT IT DOES NOT DO

It does not repair meaning. Every repair here is lexical and conservative, and a decode
that needed repair says so, so a caller can tell "parsed clean" from "parsed after we
patched its quoting". A decode that still fails returns the raw payload for the log
rather than an exception class on its own — the failure that cannot be inspected is the
one that survives for a week.
"""
from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*|\s*```\s*$")


class LLMJSONError(ValueError):
    """Decode failed. Carries the payload so the caller can log something useful."""

    def __init__(self, message: str, raw: str, attempts: list[str]):
        super().__init__(message)
        self.raw = raw
        self.attempts = attempts


def _strip_fences(text: str) -> str:
    out = _FENCE.sub("", (text or "").strip())
    return out.strip()


def _balanced(text: str) -> str | None:
    """The outermost balanced {...} or [...], ignoring braces inside strings.

    Models prepend "Here is the JSON:" and append explanations. Slicing on the first
    `{` and last `}` is the usual shortcut and it breaks whenever the trailing prose
    itself contains a brace, so this tracks string state properly.
    """
    start = next((i for i, c in enumerate(text) if c in "{["), None)
    if start is None:
        return None
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _escape_inner_quotes(text: str) -> str:
    """Escape `"` that appear INSIDE a string value rather than ending it.

    A quote ends the string only when the next non-space character is one of `,:}]`
    (or end of input). Anything else means the model wrote a quotation mark inside its
    prose — `the user says "my data is slow"` — and json.loads reports it, misleadingly,
    as a missing comma several characters later.

    This is the single most common defect in practice and it is why the axis lost 43%.
    """
    out, in_str, esc = [], False, False
    for i, c in enumerate(text):
        if not in_str:
            out.append(c)
            if c == '"':
                in_str = True
            continue
        if esc:
            out.append(c)
            esc = False
            continue
        if c == "\\":
            out.append(c)
            esc = True
            continue
        if c == '"':
            nxt = next((d for d in text[i + 1:] if not d.isspace()), "")
            if nxt in (",", ":", "}", "]", ""):
                out.append(c)
                in_str = False
            else:
                out.append('\\"')
            continue
        out.append(c)
    return "".join(out)


def _escape_control(text: str) -> str:
    """Raw newlines and tabs inside strings, which JSON forbids."""
    out, in_str, esc = [], False, False
    for c in text:
        if in_str and not esc and c in "\n\r\t":
            out.append({"\n": "\\n", "\r": "\\r", "\t": "\\t"}[c])
            continue
        out.append(c)
        if esc:
            esc = False
        elif c == "\\" and in_str:
            esc = True
        elif c == '"':
            in_str = not in_str
    return "".join(out)


def _drop_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


# JSON permits exactly these after a backslash. Anything else is invalid, and models
# produce `\'` constantly because it is valid in Python and JavaScript source.
_VALID_ESCAPE = set('"\\/bfnrtu')


def _fix_escapes(text: str) -> str:
    r"""Drop backslashes that do not introduce a valid JSON escape.

    Observed live on 2026-08-15: `"if 3GB isn\'t allowed, add 2GB instead"`. Strict
    json.loads reports `Invalid \escape`, which is accurate; the model is writing the
    string the way it would in Python.
    """
    out, i = [], 0
    in_str, esc = False, False
    while i < len(text):
        c = text[i]
        if not in_str:
            out.append(c)
            if c == '"':
                in_str = True
            i += 1
            continue
        if esc:
            # keep the backslash only when what follows is a legal escape
            if c in _VALID_ESCAPE:
                out.append("\\")
            out.append(c)
            esc = False
            i += 1
            continue
        if c == "\\":
            esc = True
            i += 1
            continue
        out.append(c)
        if c == '"':
            in_str = False
        i += 1
    return "".join(out)


def loads(text: str, *, want: type | None = None) -> tuple[Any, list[str]]:
    """Decode model output. -> (value, repairs_applied)

    `repairs_applied` is empty when the payload was already valid JSON. Callers that
    care about generation quality should count non-empty repairs rather than ignore
    them: a rising repair rate is a prompt problem, not a parser problem.
    """
    raw = text or ""
    body = _strip_fences(raw)
    candidate = _balanced(body) or body

    attempts: list[tuple[str, str]] = [("clean", candidate)]
    step = candidate
    for name, fn in (("bad-escapes", _fix_escapes),
                     ("control-chars", _escape_control),
                     ("inner-quotes", _escape_inner_quotes),
                     ("trailing-commas", _drop_trailing_commas)):
        step = fn(step)
        attempts.append((name, step))

    tried: list[str] = []
    for idx, (name, payload) in enumerate(attempts):
        try:
            value = json.loads(payload)
        except Exception as exc:  # noqa: BLE001
            tried.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if want is not None and not isinstance(value, want):
            tried.append(f"{name}: parsed as {type(value).__name__}, wanted {want.__name__}")
            continue
        # repairs are applied cumulatively, so success at idx means every transform up
        # to and including idx was in play.
        return value, [n for n, _ in attempts[1:idx + 1]]

    raise LLMJSONError(
        f"could not decode {len(raw)} chars of model output after "
        f"{len(attempts)} attempts", raw=raw, attempts=tried)


def loads_or_none(text: str, *, want: type | None = None,
                  label: str = "llm") -> Any | None:
    """Decode, or print a diagnosable failure and return None.

    The failure line carries the head and tail of the payload. The previous behaviour
    across the codebase — printing only `JSONDecodeError: Expecting ',' delimiter` —
    made a 43% loss rate look like ambient noise for as long as anyone looked at it.
    """
    try:
        value, repairs = loads(text, want=want)
    except LLMJSONError as exc:
        head = exc.raw[:220].replace("\n", " ")
        tail = exc.raw[-120:].replace("\n", " ") if len(exc.raw) > 340 else ""
        # The WHOLE payload goes to disk. A 220-char head was enough to form a
        # hypothesis about these failures and not enough to test one — the first
        # theory drawn from it (unescaped inner quotes) turned out to be wrong, and
        # the real cause (`\'`, an invalid JSON escape) was only visible in the body.
        # A failure that cannot be reproduced offline gets theorised about instead.
        path = ""
        try:
            import hashlib
            import pathlib
            d = pathlib.Path("logs/llm_json_failures")
            d.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256(exc.raw.encode("utf-8")).hexdigest()[:10]
            p = d / f"{label}-{digest}.txt"
            if not p.exists():
                p.write_text(exc.raw)
            path = f" saved={p}"
        except Exception:  # noqa: BLE001  — diagnostics must never break generation
            pass
        print(f"  [{label}] json undecodable ({len(exc.raw)} chars) "
              f"head={head!r}" + (f" tail={tail!r}" if tail else "") + path)
        for a in exc.attempts:
            print(f"  [{label}]   {a}")
        return None
    if repairs:
        print(f"  [{label}] json repaired via {'+'.join(repairs)}")
    return value
