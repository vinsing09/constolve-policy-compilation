"""run_manifest.main() with every litellm call answered by litellm's own mock_response.

The real acompletion still runs. The fake agent calls one real tool, then speaks, so
an agent turn makes two model calls. Writes the calls it answered to $FAKE_COUNTS.

    FAKE_CUSTOMER          the customer's JSON reply
    FAKE_CUSTOMER_BAD_FROM customer call number (1-based) from which replies are not JSON
    FAKE_AGENT_RAISE       agent calls raise
    FAKE_AGENT_TIMEOUT_ONCE the first agent call raises litellm.Timeout
    FAKE_DELAY             seconds each call waits
    FAKE_CUSTOMER_ACT      a user tool: the customer's opening reply performs it; the
                           next reply repeats the tool's real result if it was shown it
                           (counts saw_result), else says it saw none
    FAKE_CUSTOMER_EXPECT   the result text the tool should return
    FAKE_MUTATE            break the run at runtime, code copy untouched: setup_raise
                           (UserSim() raises), one_round (no re-ask after an action),
                           no_results (action results not passed back), no_actor
                           (customer actions lose actor "user"), blank_payload (result
                           entry kept, its content emptied)
    FAKE_RETURNED          model name put on every response
    FAKE_NO_USAGE          responses carry no token usage
    FAKE_COST_RAISE        litellm.completion_cost raises
    FAKE_FORMATS           append [caller, response_format] per model call to this file
"""
import asyncio
import json
import os
import sys
from pathlib import Path

import litellm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_manifest  # noqa: E402

real = litellm.acompletion
counts = {"agent": 0, "customer": 0}
saw_result = 0
SEEN = "ALREADY been carried out"
SAW, SAW_NONE = "My phone now says: {}", "I did not see any result."
env = os.environ.get
TOOL = [{"id": "call_1", "type": "function",
         "function": {"name": "get_customer_by_phone",
                      "arguments": json.dumps({"phone_number": "555-0100"})}}]


async def fake(**kw):
    who = "agent" if kw.get("tools") else "customer"
    counts[who] += 1
    if env("FAKE_FORMATS"):
        with open(env("FAKE_FORMATS"), "a") as f:
            f.write(json.dumps([who, kw.get("response_format")]) + "\n")
    n = counts[who]
    await asyncio.sleep(float(env("FAKE_DELAY", "0")))
    if who == "agent" and env("FAKE_AGENT_TIMEOUT_ONCE") and n == 1:
        raise litellm.Timeout("fake timeout", model=kw["model"], llm_provider="openai")
    if who == "customer" and env("FAKE_CUSTOMER_ACT"):
        kw["mock_response"] = customer_act(kw["messages"], env("FAKE_CUSTOMER_ACT"))
    elif who == "customer":
        bad = env("FAKE_CUSTOMER_BAD_FROM")
        kw["mock_response"] = ("not json" if bad and n >= int(bad) else
                               env("FAKE_CUSTOMER", '{"message": "Hi, my phone has no '
                                   'service.", "done": false, "actions": []}'))
    elif env("FAKE_AGENT_RAISE"):
        kw["mock_response"] = Exception("agent down")
    elif kw["messages"][-1].get("role") == "tool":
        kw["mock_response"] = "Thanks, I found your account."
    else:
        kw["mock_response"], kw["mock_tool_calls"] = "", TOOL
    resp = await real(**kw)
    if env("FAKE_RETURNED"):
        resp.model = env("FAKE_RETURNED")
    if env("FAKE_NO_USAGE"):
        resp.usage = None
    return resp


def customer_act(messages, tool) -> str:
    """Act on the opening turn; on the re-ask, answer from the result it was shown."""
    global saw_result
    if len(messages) == 2:
        return json.dumps({"message": "Let me try that.", "done": False,
                           "actions": [{"tool": tool, "params": {}}]})
    want = env("FAKE_CUSTOMER_EXPECT")
    if any(SEEN in str(m.get("content")) and f"{tool}: {json.dumps(want)}" in
           str(m.get("content")) for m in messages):
        saw_result += 1
        return json.dumps({"message": SAW.format(want), "done": False})
    return json.dumps({"message": SAW_NONE, "done": False})


def mutate(us, how):
    env_mod = sys.modules["services.environments.compiled_env"]
    if how == "setup_raise":
        def broken(*a, **k):
            raise RuntimeError("setup failed")
        us.UserSim.__init__ = broken
    elif how == "one_round":
        env_mod.CompiledEnv.MAX_USER_TOOL_ROUNDS = 1
    elif how == "no_results":
        reply = us.UserSim.reply

        async def blind(self, *a, rounds=None, observations=None, **k):
            return await reply(self, *a, rounds=[{**r, "results": []} for r in rounds or []],
                               observations=[], **k)
        us.UserSim.reply = blind
    elif how == "blank_payload":
        apply = env_mod.CompiledEnv._apply_user_action
        env_mod.CompiledEnv._apply_user_action = lambda self, act: {
            **(r := apply(self, act)), **({"result": None} if "result" in r else {})}
    elif how == "no_actor":
        pop = env_mod.CompiledEnv.pop_user_actions
        env_mod.CompiledEnv.pop_user_actions = lambda self: [
            {k: v for k, v in a.items() if k != "actor"} for a in pop(self)]


litellm.acompletion = fake
if env("FAKE_COST_RAISE"):
    def no_price(*a, **k):
        raise ValueError("no price")
    litellm.completion_cost = no_price
if env("FAKE_MUTATE"):
    real_install = run_manifest.Recorder.install

    def install(self, bench, us, llm):
        real_install(self, bench, us, llm)
        mutate(us, env("FAKE_MUTATE"))
    run_manifest.Recorder.install = install
try:
    rc = run_manifest.main(sys.argv[1:])
finally:
    Path(env("FAKE_COUNTS")).write_text(json.dumps(
        {**counts, "saw_result": saw_result} if env("FAKE_CUSTOMER_ACT") else counts))
sys.exit(rc)
