#!/usr/bin/env python3
"""Job 2: run τ³'s own telecom harness with one policy file (docs/job2-trace.md).

    .venv/bin/python -B scripts/run_deep.py --version v0 --run deep-v0 --mode full
    .venv/bin/python -B scripts/run_deep.py --version v2_deep --run smoke-v2 --mode smoke

Writes results/<run>/: manifest.json (before the run, completed after), calls.jsonl (one
line per model call, tagged with its simulation), stdout.log, tau3/results.json (τ³'s own
output), and last seals/<run>.sha256. A run counts only if manifest.summary.problems is
empty AND its seal exists, matches the folder and is committed (an interrupted seal
leaves no valid seal, so the run does not count).
"""
from __future__ import annotations

import argparse
import contextvars
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from importlib.metadata import version
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))
import run_manifest as rm  # noqa: E402
from runlog import RETRIES, _RetryCounter, read_jsonl, tree_hash  # noqa: E402

COPY = PROJECT / "code-original"
TAU3 = COPY / "third_party" / "tau3-bench"
POLICIES = {"v0": ("v0_f8df305a.md",
                   "065b48503bf780153fc671455cdfddb85b4c6ffb11e2ea93bd940233d804c3e9"),
            "v2_deep": ("v2_deep_c04ad136.md",
                        "f8ac98aa9ba716e31e09052b77e4fd249ef4d54a3edf53c56f8551874f150302")}
SUBSETS = {"unseen_51": 51, "fed_to_deep_62": 62, "seen_63": 63}
MODES = {"full": (114, 5), "smoke": (2, 1), "test": (2, 2)}   # test: never countable
MODEL, SEED, MAX_STEPS, CONCURRENCY = "gpt-4.1", 300, 200, 8
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
LLM_ARGS = {"temperature": 0.0, "num_retries": 0}
RETURNED_OK = re.compile(r"^gpt-4\.1-\d{4}-\d{2}-\d{2}$")
SIM = contextvars.ContextVar("sim", default=None)
POLICY_IN_PROMPT = re.compile(r"\n<policy>\n(.*)\n</policy>\Z", re.S)   # agent/llm_agent.py:35-41
INPUTS = ["data/tau2/domains/telecom/tasks.json", "data/tau2/domains/telecom/split_tasks.json",
          "data/tau2/domains/telecom/db.toml", "data/tau2/domains/telecom/user_db.toml",
          "src/tau2/utils/llm_utils.py", "src/tau2/data_model/tasks.py",
          "src/tau2/evaluator/evaluator.py", "src/tau2/agent/llm_agent.py",
          "src/tau2/user/user_simulator.py", "VENDOR_SHA"]


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def base_ids() -> list[str]:
    split = TAU3 / "data" / "tau2" / "domains" / "telecom" / "split_tasks.json"
    return json.loads(split.read_text())["base"]


def inside(p: Path, root: Path) -> bool:
    return p == root or root in p.parents


def preflight(a, run_dir: Path, environ) -> list[str]:
    bad = []
    if not SAFE_NAME.match(a.run):
        bad.append(f"--run {a.run!r} is not a single safe name")
    seals = rm.seal_path(run_dir).parent
    if seals.exists() and not seals.is_dir():
        bad.append(f"{seals} exists and is not a folder")
    for p in (run_dir, rm.seal_path(run_dir)):
        if inside(p.resolve(), COPY.resolve()):
            bad.append(f"{p} is inside code-original")
        if p.exists():
            bad.append(f"{p} already exists")
    for name, want in POLICIES.values():
        f = a.policy_dir / name
        if not f.is_file() or sha256(f.read_bytes()) != want:
            bad.append(f"{name} does not hash to {want[:12]}…")
    ids = base_ids()
    august = json.loads((a.policy_dir / "august_results_per_task.json").read_text())
    runs = august.get("v0", []), august.get("v2_deep", [])
    if len(ids) != 114 or len(set(ids)) != 114 or [len(r) for r in runs] != [3, 2] or \
            any(set(r) != set(ids) for r in runs[0] + runs[1]):
        bad.append("base split is not the 114 tasks of the 3 + 2 August runs")
    sub = {}
    for s, n in SUBSETS.items():
        sub[s] = (a.policy_dir / f"{s}_task_ids.txt").read_text().split()
        if len(sub[s]) != n or len(set(sub[s])) != n or not set(sub[s]) <= set(ids):
            bad.append(f"{s} is not {n} distinct base-split ids")
    u, f, s = (set(sub[k]) for k in SUBSETS)
    if u & s or u | s != set(ids) or not f <= s:
        bad.append("subsets do not split base into unseen + seen with fed inside seen")
    bad += provenance_problems()
    if tree_hash(COPY) != rm.stored("tree", COPY):
        bad.append("code-original does not match tree-original.txt")
    for k in environ:
        if k.startswith(("OPENAI_", "LITELLM_", "TAU2_", "AZURE_")):
            bad.append(f"shell sets {k}; it would override .env or move τ³'s data")
    return bad


PROVENANCE = HERE / "tau3-provenance.json"
UPSTREAM_COMMIT = "6e1d57640fb25df43c3d8049abeefade6275c6f0"


def record_ok(prov: dict, patch: bytes) -> bool:
    """What the committed provenance record must say for this run."""
    return prov.get("upstream_commit") == UPSTREAM_COMMIT and \
        prov.get("runtime_subset_complete") is True and prov.get("mode_differ") == {} and \
        list(prov.get("differ") or {}) == ["src/tau2/utils/llm_utils.py"] and \
        prov.get("not_in_upstream") == ["VENDOR_SHA"] and \
        prov.get("patch_sha256") == sha256(patch)


def provenance_problems() -> list[str]:
    """Rebuild the two-sided comparison now (tau3_provenance.build) and require it to
    equal the committed record and patch, which must also say what record_ok requires."""
    import tau3_provenance as tp
    patch = tp.PATCH.read_bytes() if tp.PATCH.exists() else b""
    prov = json.loads(PROVENANCE.read_text()) if PROVENANCE.exists() else {}
    if not (committed(PROVENANCE) and committed(tp.PATCH) and record_ok(prov, patch)):
        return ["scripts/tau3-provenance.json or tau3_vendor.patch is not committed or "
                "does not say upstream + one patch with a complete runtime subset"]
    built, diff = tp.build()
    if built != prov or diff.encode() != patch:
        return ["vendored τ³ no longer matches the committed provenance record"]
    return []


def committed(f: Path) -> bool:
    git = ["git", "-C", str(PROJECT)]
    return subprocess.run([*git, "ls-files", "--error-unmatch", str(f)],
                          capture_output=True).returncode == 0 and \
        subprocess.run([*git, "diff", "--quiet", "HEAD", "--", str(f)],
                       capture_output=True).returncode == 0


def expected_keys(ids: list[str], trials: int) -> list[tuple[str, int, int]]:
    """runner/batch.py:511-512,604: one seed per trial, from random.seed(config.seed)."""
    random.seed(SEED)
    seeds = [random.randint(0, 1000000) for _ in range(trials)]
    return [(t, trial, seeds[trial]) for trial in range(trials) for t in ids]


def install(calls: Path, policy: str):
    """Class-level policy patch; tag each simulation; log every model call."""
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(TAU3 / "src"))
    import litellm
    import tau2.runner.batch as batch
    import tau2.utils.llm_utils as llm
    from tau2.environment.environment import Environment

    Environment.get_policy = lambda self, *_a, **_k: policy   # type: ignore[method-assign]
    sdk_log = __import__("logging").getLogger("openai._base_client")
    sdk_log.setLevel(20)
    sdk_log.addHandler(_RetryCounter())

    real_task = batch.run_single_task

    def run_single_task(config, task, *a, seed=None, **k):
        token = SIM.set([task.id, seed])
        try:
            return real_task(config, task, *a, seed=seed, **k)
        finally:
            SIM.reset(token)
    batch.run_single_task = run_single_task

    real_completion = llm.completion
    write = threading.Lock()

    def completion(*a, **kw):
        box = []
        RETRIES.set(box)
        system = str(((kw.get("messages") or [{}])[0]).get("content") or "")
        row = {"sim": SIM.get(), "caller": "agent" if "<policy>" in system else
               "user" if "<scenario>" in system else "other",
               "requested_model": kw.get("model"), "temperature": kw.get("temperature"),
               "seed": kw.get("seed"), "num_retries": kw.get("num_retries"), "t0": time.time()}
        if row["caller"] == "agent":
            got = POLICY_IN_PROMPT.search(system)
            row["prompt_policy_sha256"] = sha256(got.group(1).encode()) if got else None
        try:
            resp = real_completion(*a, **kw)
        except BaseException as exc:
            row.update(ok=False, error=f"{type(exc).__name__}: {exc}"[:300])
            raise
        else:
            usage = getattr(resp, "usage", None)
            try:
                cost = litellm.completion_cost(completion_response=resp)
            except Exception:  # noqa: BLE001
                cost = None
            raw = resp.to_dict()
            row.update(ok=True, returned_model=getattr(resp, "model", None),
                       response_id=raw.get("id"), response_sha256=raw_hash(raw),
                       prompt_tokens=getattr(usage, "prompt_tokens", None),
                       completion_tokens=getattr(usage, "completion_tokens", None),
                       cost_usd=cost)
            return resp
        finally:
            row.update(sdk_retries=len(box), seconds=round(time.time() - row["t0"], 2))
            with write, calls.open("a") as f:
                f.write(json.dumps(row) + "\n")
    llm.completion = completion
    return batch


def raw_hash(raw: dict) -> str:
    return sha256(json.dumps(raw, sort_keys=True, default=str).encode())


def check(run_dir: Path, ids: list[str], trials: int, policy_sha: str) -> dict:
    """The run's saved files against what was asked for; reads only run_dir."""
    want = expected_keys(ids, trials)
    res = run_dir / "tau3" / "results.json"
    sims = json.loads(res.read_text())["simulations"] if res.exists() else []
    keys = [(s.get("task_id"), s.get("trial"), s.get("seed")) for s in sims]
    calls = read_jsonl(run_dir / "calls.jsonl")
    ok = [c for c in calls if c["ok"]]
    per_sim = {}
    for c in calls:
        per_sim.setdefault(tuple(c["sim"]) if c["sim"] else None, []).append(c)
    sk = {(k[0], k[2]) for k in want}
    raw_models = {str((m.get("raw_data") or {}).get("model")) for s in sims
                  for m in s.get("messages") or [] if m.get("raw_data")}
    def msg_row(m):
        u = m.get("usage") or {}
        return ("agent" if m.get("role") == "assistant" else "user",
                m["raw_data"].get("id"), raw_hash(m["raw_data"]),
                str(m["raw_data"].get("model")), u.get("prompt_tokens"),
                u.get("completion_tokens"), round(float(m.get("cost") or 0), 9))
    by_sim_msgs = {(x.get("task_id"), x.get("seed")): sorted(
        map(msg_row, (m for m in x.get("messages") or [] if m.get("raw_data"))))
        for x in sims}
    by_sim_calls = {k: sorted((c["caller"], c.get("response_id"), c.get("response_sha256"),
                               str(c.get("returned_model")), c.get("prompt_tokens"),
                               c.get("completion_tokens"),
                               round(float(c.get("cost_usd") or 0), 9))
                              for c in v if c["ok"]) for k, v in per_sim.items() if k}
    s = {"expected": len(want), "simulations": len(sims),
         "infrastructure_errors": sum(x.get("termination_reason") == "infrastructure_error"
                                      for x in sims),
         "no_reward": sum(not (x.get("reward_info") or {}).get("reward") in (0, 0.0, 1, 1.0)
                          for x in sims),
         "wrong_policy": sum(sha256(str(x.get("policy")).encode()) != policy_sha
                             for x in sims),
         "termination": dict(sorted(__import__("collections").Counter(
             str(x.get("termination_reason")) for x in sims).items())),
         "model_calls": len(calls), "failed_calls": len(calls) - len(ok),
         "sdk_retries": sum(c["sdk_retries"] for c in calls),
         "calls_by_caller": {k: sum(c["caller"] == k for c in calls)
                             for k in sorted({c["caller"] for c in calls})},
         "untagged_calls": len(per_sim.get(None, [])),
         "requested_models": sorted({str(c["requested_model"]) for c in calls}),
         "returned_models": sorted({str(c.get("returned_model")) for c in ok}),
         "message_models": sorted(raw_models),
         "temperatures": sorted({str(c["temperature"]) for c in calls}),
         "seeds_ok": all(c["sim"] and c["seed"] == c["sim"][1] for c in calls),
         "calls_with_retries_on": sum(c.get("num_retries") != 0 for c in calls),
         "agent_prompts_wrong_policy": sum(c.get("prompt_policy_sha256") != policy_sha
                                           for c in calls if c["caller"] == "agent"),
         "sims_calls_not_matching_messages": sum(
             by_sim_calls.get(k, []) != v for k, v in by_sim_msgs.items()) +
             len(set(by_sim_calls) - set(by_sim_msgs)),
         "calls_missing_usage_or_cost": sum(not rm.usable(c) for c in ok),
         "sims_without_user_call": len(sk - {k for k, v in per_sim.items() if k and
                                             any(c["caller"] == "user" for c in v)}),
         "sims_without_agent_call": len(sk - {k for k, v in per_sim.items() if k and
                                              any(c["caller"] == "agent" for c in v)}),
         "cost_usd": round(sum(c["cost_usd"] for c in ok if rm.usable(c)), 4)}
    p = []
    if sorted(keys) != sorted(want) or len(set(keys)) != len(keys):
        p.append("simulations are not exactly the expected (task, trial, seed) set")
    for k in ("infrastructure_errors", "no_reward", "wrong_policy", "failed_calls",
              "untagged_calls", "calls_missing_usage_or_cost", "sims_without_user_call",
              "sims_without_agent_call",
              "sdk_retries", "calls_with_retries_on", "agent_prompts_wrong_policy",
              "sims_calls_not_matching_messages"):
        if s[k]:
            p.append(f"{k}: {s[k]}")
    if set(s["calls_by_caller"]) - {"agent", "user"}:
        p.append("a model call came from neither the agent nor the customer")
    if s["requested_models"] != [MODEL] or s["temperatures"] != ["0.0"] or not s["seeds_ok"]:
        p.append(f"calls not gpt-4.1 / temperature 0 / trial seed: {s['requested_models']} "
                 f"{s['temperatures']} seeds_ok={s['seeds_ok']}")
    if not ok or not all(RETURNED_OK.match(m) for m in s["returned_models"] +
                         s["message_models"]):
        p.append(f"returned models {s['returned_models']} {s['message_models']}")
    s["problems"] = p
    return s


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=sorted(POLICIES))
    ap.add_argument("--run", required=True)
    ap.add_argument("--mode", required=True, choices=sorted(MODES))
    ap.add_argument("--results", type=Path, default=PROJECT / "results")
    ap.add_argument("--policy-dir", type=Path, default=PROJECT / "deep_policy")
    a = ap.parse_args(argv)
    run_dir = (a.results / a.run).resolve()
    held = rm.lock(COPY)
    bad = ["another run holds the lock on code-original"] if held is None else \
        preflight(a, run_dir, os.environ)
    if bad:
        print("REFUSED:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 2
    name, policy_sha = POLICIES[a.version]
    policy = (a.policy_dir / name).read_bytes().decode()
    n_tasks, trials = MODES[a.mode]
    ids = base_ids()[:n_tasks]
    run_dir.mkdir(parents=True)
    tree_before = tree_hash(COPY)
    config = dict(domain="telecom", task_set_name="telecom", task_split_name="base", task_ids=ids,
                  num_trials=trials, agent="llm_agent", llm_agent=MODEL,
                  llm_args_agent=LLM_ARGS, user="user_simulator", llm_user=MODEL,
                  llm_args_user=LLM_ARGS, seed=SEED, max_steps=MAX_STEPS, max_errors=10,
                  max_concurrency=CONCURRENCY, max_retries=0, auto_resume=False,
                  auto_review=False, hallucination_retries=0, log_level="ERROR",
                  timeout=None, enforce_communication_protocol=False, verbose_logs=False,
                  is_remote=False, retrieval_config=None, text_streaming_config=None,
                  save_to=str(run_dir / "tau3"))
    manifest = {
        "run": a.run, "mode": a.mode, "version": a.version,
        "tau3_provenance_sha256": sha256(PROVENANCE.read_bytes()),
        "policy_file": name, "policy_sha256": policy_sha,
        "tau3_vendor_sha": (TAU3 / "VENDOR_SHA").read_text().strip(),
        "copy_tree_hash": tree_before,
        "inputs_sha256": {f: sha256((TAU3 / f).read_bytes()) for f in INPUTS},
        "project_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT,
                                         capture_output=True, text=True).stdout.strip(),
        "command": [sys.executable, __file__, *argv], "python": sys.version.split()[0],
        "packages": {p: version(p) for p in ("litellm", "openai", "pydantic")},
        "config": config, "expected_keys": expected_keys(ids, trials), "start": now()}
    mpath = run_dir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=1))
    log = (run_dir / "stdout.log").open("w")
    sys.stdout, sys.stderr = rm.Tee(sys.__stdout__, log), rm.Tee(sys.__stderr__, log)
    err = None
    try:
        batch = install(run_dir / "calls.jsonl", policy)
        from tau2.data_model.simulation import TextRunConfig
        batch.run_domain(TextRunConfig(**config))
    except BaseException as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"[:300]
    finally:
        sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__
        log.close()
        s = check(run_dir, ids, trials, policy_sha)
        s["copy_unchanged"] = tree_hash(COPY) == tree_before
        if err:
            s["problems"].append(f"run raised {err}")
        if not s["copy_unchanged"]:
            s["problems"].append("code copy changed during the run")
        manifest.update(end=now(), summary=s)
        mpath.write_text(json.dumps(manifest, indent=1))
        try:
            rm.seal_path(run_dir).parent.mkdir(exist_ok=True)
            rm.seal_path(run_dir).write_text(rm.seal(run_dir))
            if rm.seal_path(run_dir).read_text() != rm.seal(run_dir):
                raise OSError("seal does not read back")
        except BaseException as exc:  # noqa: BLE001
            rm.seal_path(run_dir).unlink(missing_ok=True)
            s["problems"].append(f"seal not written: {type(exc).__name__}: {exc}"[:300])
            mpath.write_text(json.dumps(manifest, indent=1))
            if not isinstance(exc, Exception):
                raise
        held.close()
    print(json.dumps(manifest["summary"], indent=1))
    return 0 if not s["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
