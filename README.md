# Compiling Operating Policies into Executable Test Environments for Tool-Calling Agents: research artifacts

Code, data and conversation traces behind the telecom measurements in *Compiling Operating
Policies into Executable Test Environments for Tool-Calling Agents* (Vineet Singh, Constolve,
2026), plus three follow-up experiments run on the same suite.

**What this release covers, and what it does not.** It covers the paper's +30.0-point
separation (Section 5.4, Figure 2's first bar, Table D1), the deep-patch result on the 51
unseen tasks (+20.0 points, Table D2), and three follow-up analyses of the same 24-scenario
suite (rule removal, rule coverage, duplicate re-analysis). Every number in those can be
recomputed here from the stored traces without a model call, and was (see [Verified](#verified)).
The paper's compilation, coverage (105 of 113), mutation-adequacy, precision and
calibration figures (Tables 1 to 9, Figures 1 and 3 to 6) rest on artifacts that are **not**
in this repository. Re-running the paid experiments is possible (commands below) but gives a
new run, not the recorded one: the agent and customer are sampled from gpt-4.1.

## The paper in one paragraph

A deployment's operating policy (here, the 23,318-character telecom policy of the public
τ³-bench benchmark) defines what a correct agent action is. The method compiles that policy
and the agent's tool schemas into a runnable world: a language model writes, at build time, a
state model, one Python function per tool with its documented preconditions and effects,
named health predicates and a seed world; a tool is kept only if it parses, matches its
signature and executes. Test scenarios are then derived from that world by performing faults
on it, and are graded deterministically on the tool calls the agent makes (no model in the
grader). In one run of a 24-scenario suite with 5 repeats per arm, gpt-4.1 holding the policy
completed 73.8% of graded actions against 43.8% for gpt-4.1 holding a one-sentence
instruction, a separation of +30.0 points. Most graded actions check that the intended tool
was called, not the full final state; the instrument has not been calibrated against the
benchmark's reference environment; and the comparison is full policy against one sentence,
not policy knowledge at matched prompt length.

## Headline results reproduced here

| result | value | source | command (no model calls) |
|---|---|---|---|
| Separation, fixed customer (paper Sec. 5.4, Fig. 2) | actions completed 73.8% vs 43.8%, **+30.0** pts; per-repeat half-range 6.8 | `results/full-fixed-2` | `python -B reproduce/separation.py` |
| Same suite, original code with the broken simulated customer | **+29.2** pts (68.4% vs 39.2%) | `results/full-original-2` | same |
| Scenario-level uncertainty for +30.0 (Sec. 5.4, Table D1) | bootstrap 95% CI [24.6, 35.3]; 21 / 2 / 1 scenarios better / worse / tied; sign-flip p = 0.0001 | `results/full-fixed-2` | `python -B stats/cluster30.py table_d1.csv` |
| Deep-patched policy on the 51 unseen τ³ tasks (Table D2) | pass rate 62.4% to 82.4%, **+20.0** pts; CI [10.2, 30.2]; 28 / 9 / 14 tasks | `results/deep-v0`, `results/deep-v2_deep-2` | `python -B stats/cluster20.py table_d2.csv` |
| Rule removal, main run (pre-registered) | **1 of 6** removed rules caught (R2, APN reset: −34.6 pts, p = 0.002); no false alarm | `experiments/rule-ablation/results/main` | `python -B reproduce/ablation.py` |
| Rule removal, registered follow-up | neither R3 nor R5 caught alone; pooled with the main run (secondary), R3 meets the criterion (−21.1 pts, p = 0.016) | `experiments/rule-ablation-followup/results/main` | same |
| Rule coverage | 51 policy rules; 20 of the 40 checkable rules (50%) are checked by at least one of the 93 criteria | `experiments/rule-coverage` | `python -B experiments/rule-coverage/coverage.py` |
| Duplicate re-analysis | the 24 scenarios are 19 functionally distinct tests; +30.0 CI becomes [24.7, 35.4]; the rule-removal verdict stays 1 of 6 | `experiments/duplicate-reanalysis` | `python -B reproduce/reanalysis.py` |

Each experiment folder has its own `prereg.md` (design fixed before any paid call) and
`results.md` (full tables and caveats). Read the caveats there before quoting a number: for
example, the follow-up's R3 scenarios are variants of one situation, and in the main run R3
and R5 could not reach p < 0.05 at any effect size with the scenarios available.

## Repository layout

```
code-fixed/                 the engine the +30.0 run used, trimmed to the files the runs import
  scripts/bench.py            the benchmark entry point (both arms, one grader)
  generated/env_telecom.py    the compiled telecom environment (2,337 lines)
  generated/graph_suite_telecom.json   the generated suite (first 24 scenarios are the suite)
  generated/policy_graph_telecom.json, branches_telecom.json, env_telecom.meta.json
  runner/, services/, models.py         agent loop, grader, simulated customer, environments
  scripts/phase15_graph_suite.py + services/{policy_graph,atom_audit,case_gate,...}.py
                              the scenario generator (used by the follow-up's selection step)
  third_party/tau3-bench/     τ³-bench at 6e1d5764 (MIT, Sierra Research): src/tau2 and the
                              telecom domain data only, with one patch (see below)
code-original/              differs from code-fixed only in services/user_sim.py (the broken
                              customer); generated/ and third_party/ are symlinks to code-fixed
scripts/                    run wrapper (run_manifest.py, runlog.py), scorer (score.py),
                              τ³-native runner and scorer (run_deep.py, score_deep.py),
                              the two customer patches (role_fix.patch, json_mode.patch),
                              the τ³ vendor patch and provenance record, recorded code-copy
                              hashes, and the development tests (scripts/tests,
                              experiments/*/tests; written against the full private tree,
                              several need code copies or August runs not released here)
results/full-fixed-2/       the +30.0 run: manifest, 240 traces, per-call log, bench report, score
results/full-original-2/    the same with the original (broken-customer) code
results/deep-v0/, deep-v2_deep-2/   τ³-native runs of the original and deep-patched policy:
                              manifest and τ³ results.json (xz-compressed)
deep_policy/                the two policies, their diff, task-id subsets, SHA256SUMS
seals/                      sha256 of every file of each run, written by the run wrapper
experiments/rule-ablation/  rule-removal experiment: prereg, rules and ties, policy variants,
                              runner, scorer, tests, sealed results/main, spend ledger
experiments/rule-ablation-followup/   follow-up: prereg, scenario selection, replay
                              certificate, narrated suite, runner, scorer, sealed results/main
experiments/rule-coverage/  which policy rules any graded criterion checks
experiments/duplicate-reanalysis/     re-scoring with functionally duplicate scenarios merged
stats/                      the paper's cluster-level statistics (cluster30.py, cluster20.py)
reproduce/                  model-free verification scripts (below)
```

Run folders keep everything the run wrapper wrote, including the absolute paths of the
machine that recorded them. Folders named in some notes (`knowledge.md`, `errors.md`,
`code_review/`, `trace.md`, `docs/`) are internal process notes and are not released.

## Reproducing without model calls

Requirements: Python 3.10 or later and `git`. Scoring and statistics use only the standard
library. Run from a clean checkout:

```bash
git clone https://github.com/vinsing09/constolve-policy-compilation.git
cd constolve-policy-compilation
bash reproduce/verify_all.sh
```

`verify_all.sh` runs, and fails on the first difference:

1. `reproduce/secret_scan.py`: no credential, key or non-simulated email anywhere in the tree,
   including inside the compressed files.
2. `reproduce/separation.py`: `scripts/score.py`'s own checks and scoring on both Job 1 runs.
   The recomputed full-fixed-2 score must equal the stored `score.json` byte for byte (as JSON).
3. `stats/cluster30.py` and `stats/cluster20.py`: the paper's scenario- and task-level
   bootstrap and sign-flip statistics; the CSVs are the per-unit rows of Tables D1 and D2.
4. `reproduce/ablation.py`: both rule-removal runs scored in registered mode; results must
   equal each stored `score.json`.
5. `experiments/rule-coverage/coverage.py`: rewrites `coverage.csv` and `coverage.md`, which
   must be unchanged in git.
6. `reproduce/reanalysis.py`: runs `reanalyse.py` unchanged; `results.json` must be unchanged.
7. If the packages in `requirements.txt` are installed: `reproduce/engine_replay.py` replays
   the stored conversations through the released engine (`bench.py`, compiled environment,
   simulated customer, grader) with a stand-in for the model, to show the trimmed engine
   imports and runs. It is a smoke test, not a re-measurement.

**What the verification cannot check, and how it substitutes.** The scorers were written to
refuse any run that is not exactly the recorded one, and two of their checks point at the
private development repository: (a) the recording commit must be an ancestor of HEAD with the
pinned files unchanged since, and (b) the code copy's checksum and tree hash, which cover the
full private code copy including its `.env`, must match `scripts/{checksum,tree}-*.txt`.
`reproduce/offline_git.py` replaces (a) with a check against
`reproduce/private_commits.json`, which records the sha256 of every pinned file at each
recording commit (taken with `git show` when this repository was built); the pinned files here
must be byte-identical to those. For (b), `separation.py` compares the manifest with the
checksum and tree values committed alongside each run. Every other check (seals over every
file, manifests, each trace against the suite, bench's own report, the run wrapper's
completeness check, policy hashes) runs unchanged. The tree hash itself cannot be recomputed
from this repository. The deep runs' seals list `tau3/results.json`; decompress it
(`xz -dk`) to check that line; their `calls.jsonl` and `stdout.log` are not released.

## Reproducing with model calls (new runs, paid)

You need your own OpenAI API key. Both the agent and the simulated customer are **gpt-4.1**
via OpenAI (every recorded call returned `gpt-4.1-2025-04-14`); the agent runs at temperature
0, the customer at 0.7 in our suite (0 in the τ³-native runs). Grading uses no model.

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
cp code-fixed/.env.example code-fixed/.env      # then set OPENAI_API_KEY
cd code-fixed
python -B scripts/bench.py --suite ours --domain telecom --cases 24 --repeats 5 --max-turns 30
python -B ../reproduce/rescore_bench.py logs/bench      # score a new run the same way
```

`bench.py` writes traces to `code-fixed/logs/bench/` and a report to
`code-fixed/reports/bench_ours_telecom.json`. For the original-code arm, build a full copy
with the broken customer and run the same command inside it:

```bash
cp -r code-fixed code-original-run && cp code-original/services/user_sim.py code-original-run/services/
```

The recorded runs went through `scripts/run_manifest.py` (manifest, per-call model log,
seal). Its preflight refuses a code copy whose checksum and tree hash differ from
`scripts/*-fixed.txt`; the trimmed copy here always differs, so a new run made through it is
not, and should not be presented as, the recorded run. The same holds for the rule-removal
runners (`run_ablation.py`, `run_followup.py`), which are bound to the registered copy's
hashes in `experiments/rule-ablation/rules.py`, and to `run_deep.py`, which is bound to the
full τ³-bench vendor tree. Their code is released as the record of how the runs were made.

### Costs of the recorded runs (OpenAI list prices as computed by litellm)

| run | conversations | model calls | cost (USD) |
|---|---|---|---|
| full-fixed-2 (+30.0) | 240 | 6,599 | 26.36 |
| full-original-2 | 240 | 6,833 | 27.78 |
| deep-v0 (τ³-native, original policy, 114 tasks × 5) | 570 | 22,429 | 91.67 |
| deep-v2_deep-2 (deep-patched policy) | 570 | 21,315 | 91.60 |
| rule removal, main (7 arms × 24 × 3) | 504 | 11,682 | 54.35 (+0.33 smoke run) |
| rule removal, follow-up (3 arms × 14 × 5) | 210 | 4,264 | 17.99 (ledger total 37.64 with scenario narration and an unreleased test run) |

## τ³-bench dependency

The engine imports τ³-bench (`tau2`) from `code-fixed/third_party/tau3-bench`, upstream
commit **6e1d57640fb25df43c3d8049abeefade6275c6f0** of https://github.com/sierra-research/tau2-bench.
τ³-bench is MIT-licensed (© 2025 Sierra Research), so the parts this code uses are vendored
here with its `LICENSE`: `src/tau2/` and the telecom domain files (database, user database,
policy documents, `tasks.json`, `split_tasks.json`). Other domains' data, voice task files,
the `src/experiments/` folder and the tests are omitted. One file differs from upstream:
`src/tau2/utils/llm_utils.py` no longer pins a single 10-connection httpx pool for litellm
(`scripts/tau3_vendor.patch`; `scripts/tau3-provenance.json` lists the file-by-file comparison
with upstream). To use upstream instead, clone it at that commit, apply the patch, and copy
`src` and `data/tau2/domains/telecom` over the vendored folder.

## Licences

- Code written by Constolve: MIT (`LICENSE`).
- Data, traces and results: CC BY 4.0 (`DATA_LICENSE`).
- τ³-bench, its telecom policy, tools and data: MIT, © 2025 Sierra Research
  (`code-fixed/third_party/tau3-bench/LICENSE`). The policy variants in
  `experiments/rule-ablation/policies/` and `deep_policy/` are that policy and edits of it.

All customers, accounts, phone numbers and email addresses in the traces are simulated
benchmark data.

## Citation

See `CITATION.cff`. Vineet Singh, Constolve. *Compiling Operating Policies into Executable
Test Environments for Tool-Calling Agents*, 2026.

## Verified

On 28 October 2026 this repository was cloned fresh from GitHub into an empty temporary folder
and `bash reproduce/verify_all.sh` was run twice: with Python 3.14.4 and only the standard
library (step 7 skipped), and with the Python 3.12.14 environment of `requirements.txt`. Both
exited 0. Two negative controls were also run on the clone: flipping one criterion in one
stored trace makes `separation.py` refuse (`files differ from the seal full-fixed-2.sha256`),
and committing a one-line change to `scripts/runlog.py` makes it refuse
(`manifest provenance wrong: ['wrapper']`). Output of the run with the full environment
(step 6 shortened to its summary lines):

```
== 0. secret and personal-data scan
secret scan: 0 finding(s)

== 1. +30.0 separation, and the original-code run, from stored traces
full-fixed-2 (code-fixed, 240 conversations)
                      policy    stub     gap   per-repeat gaps
  solved fully          35.0     7.5   +27.5   [37.5, 20.8, 41.7, 16.7, 20.8] mean +27.5 half-range 12.5
  criteria passed       71.8    44.2   +27.7   [34.4, 18.2, 37.0, 27.6, 21.1] mean +27.7 half-range 9.4
  actions completed     73.8    43.8   +30.0   [35.1, 21.6, 35.1, 32.4, 25.7] mean +30.0 half-range 6.8
  identical to the stored score.json; seal sha256 6ac00f98e79fb9cd...
full-original-2 (code-original, 240 conversations)
                      policy    stub     gap   per-repeat gaps
  solved fully          35.0     5.0   +30.0   [29.2, 33.3, 29.2, 33.3, 25.0] mean +30.0 half-range 4.2
  criteria passed       70.3    41.5   +28.8   [30.4, 38.3, 18.4, 27.8, 29.0] mean +28.8 half-range 9.9
  actions completed     68.4    39.2   +29.2   [32.4, 36.5, 20.3, 27.0, 29.7] mean +29.2 half-range 8.1
  (no score.json was stored for this run); seal sha256 07791791e308a5f0...

== 2. scenario-cluster bootstrap and sign-flip test for +30.0 (stats/cluster30.py)
pooled gap (must be 30.0): 30.0
scenarios: 24 graded actions/arm: 370
scenario-cluster bootstrap 95% CI: [24.6, 35.3]
scenarios policy better / worse / tied: 21 2 1
paired sign-flip permutation p (one-sided, scenario unit): 0.0001

== 3. deep-patch result on the 51 unseen tasks (stats/cluster20.py)
unseen tasks: 51 v0 62.4 patch 82.4 gap 20.0
task-cluster bootstrap 95% CI: [10.2, 30.2]
tasks better / worse / tied: 28 9 14
paired sign-flip permutation p (one-sided, task unit): 0.0002

== 4. rule removal: main run and registered follow-up
RULE REMOVAL, main run: valid blocks 63 of 72, cost $54.3454
| removed rule | tied criteria | control pass % | variant pass % | drop | p | caught? | untied change |
|---|---|---|---|---|---|---|---|
| R1_airplane | 28 in 11 scenarios | 85.7 | 92.9 | -7.1 | 0.7969 | no | -1.4 (p=0.564) |
| R2_apn | 52 in 10 scenarios | 100.0 | 65.4 | +34.6 | 0.0020 | yes | -3.7 (p=0.315) |
| R3_payment | 16 in 3 scenarios | 100.0 | 75.0 | +25.0 | 0.2500 | no | -10.1 (p=0.008) |
| R4_suspension | 24 in 9 scenarios | 100.0 | 91.7 | +8.3 | 0.2500 | no | -3.2 (p=0.239) |
| R5_escalate_unfixable | 26 in 5 scenarios | 61.5 | 34.6 | +26.9 | 0.0625 | no | -1.8 (p=0.413) |
| R6_transfer_scope | 63 in 24 scenarios | 69.8 | 54.0 | +15.9 | 0.0367 | no | -0.6 (p=0.152) |
pair R5_escalate_unfixable: own-only drop +30.8 p=0.1250; partner-only drop +12.0 p=0.1372; lands on right rule: False
pair R6_transfer_scope: own-only drop +22.0 p=0.0146; partner-only drop +30.8 p=0.1250; lands on right rule: True
  identical to the stored score.json

RULE REMOVAL, registered follow-up: valid blocks 70 of 70, cost $17.9887
PRIMARY (follow-up alone)
| removed rule | tied criteria | control pass % | variant pass % | drop | p | caught? | untied change |
|---|---|---|---|---|---|---|---|
| R3_payment | 60 in 6 scenarios | 96.7 | 76.7 | +20.0 | 0.0625 | no | -1.1 (p=0.324) |
| R5_escalate_unfixable | 40 in 8 scenarios | 42.5 | 37.5 | +5.0 | 0.4219 | no | -2.4 (p=0.037) |
SECONDARY (main + follow-up pooled, functional duplicates merged)
| removed rule | tied criteria | control pass % | variant pass % | drop | p | caught? | untied change |
|---|---|---|---|---|---|---|---|
| R3_payment | 76 in 9 scenarios | 97.4 | 76.3 | +21.1 | 0.0156 | yes | -6.1 (p=0.008) |
| R5_escalate_unfixable | 61 in 12 scenarios | 49.2 | 34.4 | +14.8 | 0.0879 | no | -2.1 (p=0.064) |
  identical to the stored score.json

== 5. rule coverage (rewrites coverage.csv and coverage.md; must be unchanged)
{"not_checkable": 11, "not_covered": 20, "covered": 20}
coverage.csv and coverage.md unchanged
== 6. duplicate re-analysis (rewrites results.json; must be unchanged)
original_24 {'units': 24, 'pooled_gap': 30.0, 'gap_over_units': 30.0, 'ci95': [24.55696202531646, 35.294117647058826], 'signflip_p_mc': 0.0001, 'signflip_p_exact': 0.0, ...}
clustered_19 {'units': 19, 'pooled_gap': 30.0, 'gap_over_units': 30.0, 'ci95': [24.65753424657534, 35.43859649122807], 'signflip_p_mc': 0.0001, 'signflip_p_exact': 0.0001, ...}
results.json unchanged

== 7. engine replay smoke test (needs requirements.txt; skipped otherwise)
engine replay: 48 conversations, 0 errors, 110 of 186 graded actions completed (stored run, repeat 0: 118); 40 conversations grade the same as stored

ALL REPRODUCTIONS MATCH
```
