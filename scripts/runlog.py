"""What a run did, recorded at the call sites without changing what they return.

tree_hash: a locale-independent fingerprint of a code copy (dirs, files incl. .env, links).
Recorder:  one row per litellm.acompletion call and per failed customer reply, each tagged
           with the conversation that made it.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import re
import time
from pathlib import Path

SIM_ERROR = "(user-sim error"
RETURNED_OK = re.compile(r"^gpt-4\.1(-\d{4}-\d{2}-\d{2})?$")

CONV = contextvars.ContextVar("conv", default=None)
ROLE = contextvars.ContextVar("role", default=None)
RETRIES = contextvars.ContextVar("retries", default=None)


def tree_hash(root: Path) -> str:
    """sha256 over every directory, file (content) and symlink (target), byte-sorted."""
    lines = []
    for dirpath, dirs, files in os.walk(root):
        dirs.sort(key=os.fsencode)
        rel = Path(dirpath).relative_to(root)
        lines.append(f"d {rel}")
        for name in sorted(files + [d for d in dirs if os.path.islink(Path(dirpath, d))],
                           key=os.fsencode):
            p = Path(dirpath, name)
            if p.is_symlink():
                lines.append(f"l {rel / name} -> {os.readlink(p)}")
            else:
                lines.append(f"f {rel / name} {hashlib.sha256(p.read_bytes()).hexdigest()}")
    return hashlib.sha256("\n".join(sorted(lines, key=os.fsencode)).encode()).hexdigest()


class _RetryCounter(logging.Handler):
    """The OpenAI SDK retries inside one litellm call and says so only in its log."""

    def emit(self, record):
        if record.getMessage().startswith("Retrying request"):
            box = RETRIES.get()
            if box is not None:
                box.append(1)


def _no_constant(name: str):
    raise ValueError(f"{name} is not JSON")


def reply_shape(text: str | None, wants_actions: bool) -> str:
    """The structure of the customer's raw reply against the format it is told to use
    (user_sim.py:67-69,201-202): strict JSON (no NaN/Infinity), no code fences, exactly
    message (non-empty string), done (true/false) and, when it has phone tools, actions
    (a list of exactly {tool, params}). Not judged: the 1-3 sentence length and the
    content. UserSim.reply accepts much more (fences, missing or falsey fields,
    coercion); this only counts what it is sent."""
    raw = (text or "").strip()
    if raw.startswith("```"):
        return "fenced"
    try:
        d = json.loads(raw, parse_constant=_no_constant)
    except ValueError:
        return "not_json"
    if not isinstance(d, dict):
        return "not_object"
    if set(d) - {"message", "done", "actions"}:
        return "extra_fields"
    if not isinstance(d.get("message"), str):
        return "message_not_string"
    if not d["message"].strip():
        return "message_empty"
    if not isinstance(d.get("done"), bool):
        return "done_not_bool"
    if "actions" not in d:
        return "actions_missing" if wants_actions else "ok"
    if not isinstance(d["actions"], list):
        return "actions_not_list"
    if not all(isinstance(a, dict) and set(a) == {"tool", "params"}
               and isinstance(a["tool"], str) and isinstance(a["params"], dict)
               for a in d["actions"]):
        return "bad_action"
    return "ok"


class Recorder:
    def __init__(self, calls: Path, customer_errors: Path):
        self.calls, self.customer_errors = calls, customer_errors

    def _write(self, path: Path, row: dict):
        with path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    def install(self, bench, user_sim_module, litellm) -> None:
        sdk_log = logging.getLogger("openai._base_client")
        sdk_log.setLevel(logging.INFO)
        sdk_log.addHandler(_RetryCounter())

        real_call, rec = litellm.acompletion, self

        async def acompletion(*args, **kw):
            box = []
            RETRIES.set(box)
            role = ROLE.get() or ("agent" if kw.get("tools") else "other")
            row = {"conv": CONV.get(), "caller": role, "requested_model": kw.get("model"),
                   "temperature": kw.get("temperature"),
                   "response_format": kw.get("response_format"), "t0": time.time()}
            try:
                resp = await real_call(*args, **kw)
            except BaseException as exc:
                row.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:300])
                raise
            else:
                usage = getattr(resp, "usage", None)
                try:
                    cost = litellm.completion_cost(completion_response=resp)
                except Exception:  # noqa: BLE001
                    cost = None
                row.update(ok=True, returned_model=getattr(resp, "model", None),
                           prompt_tokens=getattr(usage, "prompt_tokens", None),
                           completion_tokens=getattr(usage, "completion_tokens", None),
                           cost_usd=cost)
                if role == "customer":
                    choices = getattr(resp, "choices", None) or [None]
                    text = getattr(getattr(choices[0], "message", None), "content", None)
                    system = str(((kw.get("messages") or [{}])[0]).get("content") or "")
                    row["reply_text"] = text
                    row["reply_shape"] = reply_shape(text, '"actions": [' in system)
                return resp
            finally:
                row.update(sdk_retries=len(box), seconds=round(time.time() - row["t0"], 2))
                rec._write(rec.calls, row)

        litellm.acompletion = acompletion

        real_reply = user_sim_module.UserSim.reply

        async def reply(self, *args, **kw):
            token = ROLE.set("customer")
            try:
                out = await real_reply(self, *args, **kw)
            finally:
                ROLE.reset(token)
            if str(out.message).startswith(SIM_ERROR):
                rec._write(rec.customer_errors, {"conv": CONV.get(), "message": out.message})
            return out

        user_sim_module.UserSim.reply = reply

        real_one = bench.one

        async def one(case, agent, domain, cap, label, rep):
            CONV.set(conv_key(label, rep, case["id"]))
            return await real_one(case, agent, domain, cap, label, rep)

        bench.one = one


def conv_key(arm, rep, case_id) -> str:
    return f"{arm}|{rep}|{case_id}"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
