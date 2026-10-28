"""The follow-up's own spend ledger and the worst-case-before-call guard (as in ../rule-ablation).

Every guarded call first writes its worst case to the ledger (fsynced), then a settling line
(actual - worst) when it returns. A call is refused if spend + open reservations + its own
worst case would pass the hard line. One paid process at a time holds the ledger lock.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from pathlib import Path

MAX_OUT = 32768      # gpt-4.1's output-token limit, for calls that set no max_tokens


class BudgetStop(RuntimeError):
    pass


SETTLES = {"settle_call", "settle_failed_call_worst_case", "settle_call_cost_unknown_worst_case"}
PREFIX_KINDS = {"manual_probe_worst_case"}
FREEZE = "freeze_open_reservations"   # written by each paid invocation before its first call


def prior_spend(ledger: Path, prefix: tuple[int, str] | None = None) -> float:
    """Sum of the ledger, refused unless it is append-only over the registered prefix
    (rows, sha256 of those rows' bytes) and every row is well formed: a reservation is
    >= 0, a settlement takes back at most its own reservation (FIFO per run and
    conversation; calls within one conversation are sequential), no other kind after the
    prefix. Reservations still open at the end of the prefix, or at a FREEZE row (each paid
    invocation writes one before its first call), can never be settled."""
    lines = ledger.read_text().splitlines(keepends=True) if ledger.exists() else []
    if prefix is not None:
        n, digest = prefix
        if len(lines) < n or hashlib.sha256("".join(lines[:n]).encode()).hexdigest() != digest:
            raise BudgetStop("ledger is not append-only over its registered prefix")
    open_res, total = {}, 0.0
    for i, raw in enumerate(lines):
        if prefix is not None and i == prefix[0]:
            open_res = {}          # still open at the prefix: charged for good, never refunded
        row = json.loads(raw)
        usd, kind, key = float(row["usd"]), row["kind"], (row["run"], row["conv"])
        if not math.isfinite(usd):
            raise BudgetStop(f"ledger row {i + 1}: non-finite amount")
        if kind == FREEZE and usd == 0:
            open_res = {}          # an earlier invocation's open reservations stay charged
        elif kind == "reserve_worst_case" and usd >= 0:
            open_res.setdefault(key, []).append(usd)
        elif kind in SETTLES and usd <= 0 and open_res.get(key) and -usd <= open_res[key][0] + 1e-9:
            open_res[key].pop(0)
        elif not (kind in PREFIX_KINDS and usd >= 0 and prefix and i < prefix[0]):
            raise BudgetStop(f"ledger row {i + 1} is malformed: {row}")
        total += usd
    return total


def lock(ledger: Path):
    f = ledger.with_suffix(".lock").open("a")
    try:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        return None
    return f


def debit(ledger: Path, run: str, conv, usd: float, kind: str) -> None:
    with ledger.open("a") as f:
        f.write(json.dumps({"run": run, "conv": conv, "usd": usd, "kind": kind}) + "\n")
        f.flush()
        os.fsync(f.fileno())


def worst_cost(kw: dict, price: dict) -> float:
    """Every input character a token, plus the most output the call may return."""
    chars = len(json.dumps(kw.get("messages"), default=str)) + \
        len(json.dumps(kw.get("tools"), default=str))
    out = kw.get("max_tokens") or MAX_OUT
    return chars * price["input_cost_per_token"] + out * price["output_cost_per_token"]


def install_guard(litellm, budget, ledger: Path, models: set, conv, on_call=None) -> None:
    """Wrap litellm.acompletion: refuse unlisted models, reserve the worst case or refuse,
    then charge the actual cost (the worst case if unknown or failed). `budget` needs
    hard, spent, reserved, run and stopped; `conv()` names the current conversation."""
    inner = litellm.acompletion

    async def guarded(*a, **kw):
        model = kw.get("model")
        if model not in models:
            raise BudgetStop(f"model {model!r} is not registered")
        if on_call:
            on_call(kw)
        worst = worst_cost(kw, litellm.model_cost[model])
        if budget.spent + budget.reserved + worst > budget.hard:
            budget.stopped.add(conv())
            raise BudgetStop(f"spent ${budget.spent:.2f} + reserved ${budget.reserved:.2f}")
        budget.reserved += worst
        debit(ledger, budget.run, conv(), worst, "reserve_worst_case")
        cost, kind = worst, "failed_call_worst_case"
        try:
            resp = await inner(*a, **kw)
            try:
                cost, kind = float(litellm.completion_cost(completion_response=resp)), "call"
            except Exception:  # noqa: BLE001
                kind = "call_cost_unknown_worst_case"
            return resp
        finally:
            budget.reserved -= worst
            budget.spent += cost
            debit(ledger, budget.run, conv(), cost - worst, f"settle_{kind}")

    litellm.acompletion = guarded
