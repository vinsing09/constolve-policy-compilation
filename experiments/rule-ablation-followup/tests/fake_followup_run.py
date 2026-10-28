"""run_followup.main() with every model call answered by litellm's mock_response (no API).

The customer opens, the agent answers in text, the customer ends the call. Env:
    FAKE_LEDGER (required)  the ledger file; the real one holds paid spend
    FAKE_CAP_USD, FAKE_SUITE_SHA  override design constants
"""
import collections
import os
import sys
from pathlib import Path

os.environ["RULE_ABLATION_TEST"] = "1"
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import litellm  # noqa: E402

import design as D  # noqa: E402
import run_followup as rf  # noqa: E402
from runlog import CONV  # noqa: E402

real = litellm.acompletion
seen = collections.Counter()


async def fake(**kw):
    if kw.get("tools"):
        kw["mock_response"] = "Please check your SIM status."
    else:
        seen[CONV.get()] += 1
        kw["mock_response"] = ('{"message": "My phone has no service.", "done": false, '
                               '"actions": []}' if seen[CONV.get()] == 1 else
                               '{"message": "Thanks, bye.", "done": true, "actions": []}')
    return await real(**kw)


litellm.acompletion = fake
rf.LEDGER = Path(os.environ["FAKE_LEDGER"])
if os.environ.get("FAKE_CAP_USD"):
    D.CAP_USD = float(os.environ["FAKE_CAP_USD"])
if os.environ.get("FAKE_SUITE_SHA"):
    D.SUITE_SHA = os.environ["FAKE_SUITE_SHA"]
sys.exit(rf.main(sys.argv[1:]))
