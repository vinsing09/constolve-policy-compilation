"""run_ablation.main() with every model call answered by litellm's mock_response (no API).

The customer opens, the agent answers in text, the customer ends the call. Env:
    FAKE_CAP_USD, FAKE_EST, FAKE_NEXT, FAKE_INFLIGHT, FAKE_WORST  budget constants
    FAKE_LEDGER  the spend-ledger file (the real one holds the paid smoke run)
"""
import collections
import os
import sys

os.environ["RULE_ABLATION_TEST"] = "1"
from pathlib import Path

import litellm

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_ablation as ra  # noqa: E402
from runlog import CONV  # noqa: E402

real = litellm.acompletion
seen = collections.Counter()


async def fake(**kw):
    if kw.get("tools") and os.environ.get("FAKE_FAIL"):
        raise RuntimeError("agent call failed")
    if kw.get("tools"):
        kw["mock_response"] = "Please turn airplane mode off."
    else:
        seen[CONV.get()] += 1
        kw["mock_response"] = ('{"message": "My phone has no service.", "done": false, '
                               '"actions": []}' if seen[CONV.get()] == 1 else
                               '{"message": "Thanks, bye.", "done": true, "actions": []}')
    return await real(**kw)


litellm.acompletion = fake
ra.LEDGER = Path(os.environ["FAKE_LEDGER"])
for env, name in (("FAKE_CAP_USD", "CAP_USD"), ("FAKE_EST", "EST_PER_CONV"),
                  ("FAKE_NEXT", "NEXT_CONV"), ("FAKE_INFLIGHT", "IN_FLIGHT"),
                  ("FAKE_WORST", "WORST_CALL")):
    if os.environ.get(env):
        setattr(ra, name, float(os.environ[env]))
sys.exit(ra.main(sys.argv[1:]))
