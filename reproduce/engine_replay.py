#!/usr/bin/env python3
"""Model-free smoke test of the released engine: replay stored conversations through bench.py.

    python -B reproduce/engine_replay.py          (needs requirements.txt installed; no API key)

Copies code-fixed/ to a temporary folder (with a placeholder .env), loads scripts/bench.py
unchanged, and replaces litellm.acompletion with a stand-in that replays repeat 0 of
results/full-fixed-2: the agent's recorded tool calls in order, then a closing message; the
customer's recorded messages and device actions, then "done". All 48 conversations
(24 scenarios x 2 arms) run through the compiled environment, the user simulator and the
grader. This shows the trimmed engine imports and executes; it is not a re-measurement (the
replay is not turn-for-turn faithful, so graded counts can differ from the stored ones).
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRACES = ROOT / "results" / "full-fixed-2" / "traces"


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    copy = tmp / "code-fixed"
    shutil.copytree(ROOT / "code-fixed", copy)
    (copy / ".env").write_text("DEFAULT_MODEL=gpt-4.1\nOPENAI_API_KEY=not-a-key\n")
    os.chdir(copy)
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("bench", copy / "scripts" / "bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    import litellm
    from litellm import ModelResponse

    queue = {"agent": [], "user": []}

    def reply(msg):
        return ModelResponse(model="gpt-4.1", choices=[{"index": 0, "finish_reason": "stop",
                                                         "message": msg}])

    async def fake(**kw):
        if kw.get("tools"):
            if queue["agent"]:
                tc = queue["agent"].pop(0)
                return reply({"role": "assistant", "content": None, "tool_calls": [
                    {"id": f"call_{len(queue['agent'])}", "type": "function",
                     "function": {"name": tc["tool_name"],
                                  "arguments": json.dumps(tc["arguments"])}}]})
            return reply({"role": "assistant", "content": "Is there anything else?"})
        if queue["user"]:
            u = queue["user"].pop(0)
            return reply({"role": "assistant", "content": json.dumps({
                "message": u["content"], "done": False,
                "actions": [{"tool": t["tool_name"], "params": t["arguments"]}
                            for t in u.get("tool_calls") or []]})})
        return reply({"role": "assistant", "content": json.dumps({"message": "Thanks.",
                                                                   "done": True})})

    litellm.acompletion = fake

    async def run():
        cases = bench.our_cases("telecom", 24)
        agents = {"with_policy": bench.build_agent("telecom", None),
                  "stub": bench.build_agent("telecom", bench.STUB)}
        rows = []
        for label, agent in agents.items():
            for c in cases:
                safe = "".join(ch if ch.isalnum() else "_" for ch in c["id"])[:70]
                d = json.loads((TRACES / f"ours_{label}_0_{safe}.json").read_text())
                queue["agent"] = [tc for t in d["history"] if t["role"] == "agent"
                                  for tc in t.get("tool_calls") or []]
                queue["user"] = [t for t in d["history"] if t["role"] == "user"][1:]
                r = await bench.one(c, agent, "telecom", 30, label, 0)
                rows.append((r.get("error"), r["k"], r["n"], d["actions_done"]))
        return rows

    try:
        rows = asyncio.run(run())
    finally:
        os.chdir(ROOT)
        shutil.rmtree(tmp)
    errors = sum(1 for r in rows if r[0])
    print(f"engine replay: {len(rows)} conversations, {errors} errors, "
          f"{sum(r[1] for r in rows)} of {sum(r[2] for r in rows)} graded actions completed "
          f"(stored run, repeat 0: {sum(r[3] for r in rows)}); "
          f"{sum(r[1] == r[3] for r in rows)} conversations grade the same as stored")
    return 1 if errors or len(rows) != 48 else 0


if __name__ == "__main__":
    sys.exit(main())
