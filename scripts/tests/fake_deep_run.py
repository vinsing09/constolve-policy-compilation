"""run_deep.main with only the model faked; everything else is τ³'s real harness.

    FAKE_COUNTS    write {"agent": n, "user": n} here at exit
    FAKE_MUTATE    no_patch | fail_call | wrong_model | undated | untagged | no_usage | seal_fail | seal_interrupt
    FAKE_PROMPTS   write the sha256 of the <policy> text seen in agent prompts here
"""
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
import run_deep  # noqa: E402

counts = {"agent": 0, "user": 0}
prompt_policies = set()
mutate = os.environ.get("FAKE_MUTATE", "")


def make_fake(real):
    def fake(**kw):
        system = str(kw["messages"][0].get("content") or "")
        who = "agent" if "<policy>" in system else "user"
        counts[who] += 1
        if who == "agent":
            got = re.search(r"\n<policy>\n(.*)\n</policy>\Z", system, re.S)
            prompt_policies.add(hashlib.sha256(got.group(1).encode()).hexdigest()
                                if got else None)
        if mutate == "fail_call" and who == "user" and counts["user"] == 2:
            raise RuntimeError("fake provider outage")
        if who == "agent":
            kw["mock_response"] = "I can help with that. Please restart your phone."
        else:
            first = sum(m.get("role") == "assistant" for m in kw["messages"]) <= 1
            kw["mock_response"] = "My phone has no service." if first else "###STOP###"
        resp = real(**kw)
        resp.model = {"wrong_model": "gpt-4o-2024-08-06", "undated": "gpt-4.1"}.get(
            mutate, "gpt-4.1-2025-04-14")
        if mutate == "no_usage":
            resp.usage = None
        return resp
    return fake


real_install = run_deep.install


def install(calls, policy):
    """τ³ is imported here, after preflight, exactly as in a real run."""
    sys.path.insert(0, str(run_deep.TAU3 / "src"))
    import tau2.utils.llm_utils as llm
    llm.completion = make_fake(llm.completion)
    batch = real_install(calls, policy)
    if mutate == "no_patch":
        from tau2.environment.environment import Environment
        Environment.get_policy = lambda self, *_a, **_k: self.policy
    elif mutate == "untagged":
        batch.run_single_task = batch.run_single_task.__closure__[0].cell_contents
    return batch


run_deep.install = install
if mutate in ("seal_fail", "seal_interrupt"):
    def seal(_run_dir):
        raise OSError("disk full") if mutate == "seal_fail" else KeyboardInterrupt()
    run_deep.rm.seal = seal
try:
    rc = run_deep.main(sys.argv[1:])
finally:
    Path(os.environ["FAKE_COUNTS"]).write_text(json.dumps(counts))
    if os.environ.get("FAKE_PROMPTS"):
        Path(os.environ["FAKE_PROMPTS"]).write_text(json.dumps(sorted(map(str, prompt_policies))))
sys.exit(rc)
