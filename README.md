# Compiling Operating Policies into Executable Test Environments for Tool-Calling Agents

Research artifacts for the preprint *Compiling Operating Policies into Executable Test
Environments for Tool-Calling Agents* (Vineet Singh, Constolve, October 2026): the compiled
telecom environment, the generated suite, the scenario generator, the run, scoring and
statistics code, the conversation traces of every run the paper's telecom results rest on,
and scripts that recompute those results without calling a model.

From the paper's abstract:

> Tool-calling agents work under operating policies that a general benchmark cannot know. We
> ask whether executable environments compiled from a deployment's written policy can produce
> solvable tests that catch policy violations by such agents. Our pipeline calls a language
> model only at build time. The model writes a state model and one function per tool from the
> policy and the tool schemas, and we keep what survives execution. We then discover faults by
> perturbing the compiled world, compose them into scenarios, and emit a scenario only with a
> replay certificate showing that a reference repair sequence solves it. [...] We ran a
> 24-scenario compiled suite, which contains 19 distinct tests, on the 2,337-line build. An
> agent given the policy completes 73.8% of graded actions and an otherwise identical agent
> given one sentence completes 43.8%, a separation of +30.0 percentage points (95% CI
> [24.7, 35.4] over the 19 distinct scenarios), with the policy arm ahead on 16 of 19. In a
> pre-registered experiment we removed one of six rules at a time from the agent's policy
> (3 repeats, 63 valid scenario blocks). [...] One of the 6 removals meets the pre-registered
> criterion, and the count is best read by group. [...] A pre-registered follow-up on 14 new
> scenarios did not catch either of the two underpowered rules on its own. Pooled with the
> first run in a secondary analysis, removal of the payment rule meets the criterion
> (21.1 points, p = 0.016).

**Scope.** Every number in the paper falls into one of three categories, listed in full in the
table below: (1) recomputed here from released raw records, (2) checked against archived
outputs only, because the run that produced them kept summaries and no raw records, and (3) not
reproducible from this release, with the reason. Everything in category 1 or 2 is checked
against the printed value by `bash reproduce/verify_all.sh` (see [Verified](#verified)).
Re-running the paid experiments is possible but gives a new run, not the recorded one, because
the agent and the simulated customer are sampled from gpt-4.1.

## Paper ↔ artifact

Numbers refer to the preprint; labels in brackets are the LaTeX labels. All commands run from
the repository root, need no API key and no package outside the standard library, and are run
in order by `bash reproduce/verify_all.sh`. `reproduce/paper_tables.py` prints Tables 1 to 7,
Figure 2 and the numbers of Sections 5.1 and 5.2, and fails if any value differs from the paper.

### 1. Recomputed from released raw records

| Paper item | What it reports | Reproduce with | Raw records |
|---|---|---|---|
| Table 1 [tab:domains] | Telecom column: 114 tasks, 43 schemas, 23,318 characters of policy, 94 tasks with customer actions, 43/43 compiled, 11 health predicates, 5 derived observables, 2,337 lines; compiled counts for airline (14/14) and retail (16/16) | `python -B reproduce/paper_tables.py` | τ³ telecom data and source, `code-fixed/generated/env_*.py` and `.meta.json` |
| Section 5.1 text | The benchmark's own telecom environment is 2,851 lines | `paper_tables.py` | `code-fixed/third_party/tau3-bench/src/tau2/domains/telecom/*.py` |
| Section 5.2 | Discovery matches all 17 agent-repairable fault types by repair-tool name (16 complete, 1 partial: overdue-bill suspension); 2 of 19 are escalation-only. Coverage 105 of 113 tasks, 90 of the 94 repair-graded, 15 escalation-graded tasks among the 105, 8 uncovered, 92,184 emitted scenarios; one task excluded because its repair set is empty | `python -B reproduce/repair_coverage.py` (also compares the covered and missing lists and the unmatched repair sets with the archived report) | `records/coverage/emitted_repair_sets.json.xz` (the repair-call set of every emitted scenario, as 21,606 distinct sets with counts, plus the sha256 of the 183 MB scenario file they come from), `records/coverage/tasks_small.json` (τ³'s 20 single-fault tasks), the 28 atoms `experiments/rule-ablation-followup/inputs/phase3_atoms_telecom.json`, τ³ `tasks.json` and `split_tasks.json` |
| Table 2 [tab:suite] | The 24-scenario suite: 19 distinct tests, 93 criteria (74 requiring a call, 48 of them with no argument values; 19 forbidding a call), 120 conversations and 370 graded actions per arm | `paper_tables.py` | `code-fixed/generated/graph_suite_telecom.json`, `results/full-fixed-2` |
| Table 3 [tab:rulecov] | Policy rule coverage: 51 rules, 20 covered, 20 not covered, 11 not checkable, by section | `python -B experiments/rule-coverage/coverage.py` (rewrites `coverage.csv`/`coverage.md`, which must not change), then `paper_tables.py` | `experiments/rule-coverage/rules.csv`, policy `experiments/rule-ablation/policies/control.md` |
| Section 5.3 text, Table 4 [tab:perscen] | 73.8% (273/370) vs 43.8% (162/370), +30.0 pp; per-scenario counts; per-repeat half-range 6.8 | `python -B reproduce/separation.py` (the run's own scorer, equal to the stored `score.json`); `paper_tables.py`; `python -B stats/cluster30.py out.csv` (24-scenario rows) | the 240 traces in `results/full-fixed-2` |
| Section 5.3 text, Figure 2 [fig:perscen] | 19 distinct scenarios: bootstrap 95% CI [24.7, 35.4], ahead on 16, behind on 2, tied on 1, sign-flip p < 0.001; the 19 plotted points | `python -B reproduce/reanalysis.py` (writes `experiments/duplicate-reanalysis/results.json`, which must not change), then `paper_tables.py` | `results/full-fixed-2` |
| Table 5 [tab:known] | Catching removed rules: 63 valid blocks, 1 of 6 caught (R2, 34.6 points, p = 0.008), 0 off-target | `python -B reproduce/ablation.py` (registered scorer, equal to the stored `score.json`, which uses the 24 scenarios as units), then `reanalysis.py` and `paper_tables.py` (19 distinct scenarios as units) | traces in `experiments/rule-ablation/results/main` |
| Section 5.4 text | R5/R6 overlap pair (R6 own-only 22.0 points, p = 0.024); R3 untied change 75.8% to 65.6% (−10.1, p = 0.008) | `reanalysis.py` prints the pair rows; untied rates are the `untied` entries of `results.json` | same |
| Table 6 [tab:followup] | Follow-up alone: R3 20.0 points (p = 0.063), R5 5.0; pooled secondary: R3 21.1 (p = 0.016, meets), R5 14.8 | `ablation.py`, then `paper_tables.py` | traces in `experiments/rule-ablation-followup/results/main` |
| Section 5.5, Table 7 [tab:rel] | 834 runs over 278 tasks; 29.8% (34/114) of telecom, 30.7% (35/114) of retail and 34.0% (17/50) of airline tasks change verdict; retail 78, 77 and 77 of 114 (68% each); airline 23, 31 and 27 of 50 (46/62/54%); no run recorded an error; pass^1 − pass^3 16.0 / 14.6 / 14.3 points, spread 1.7, base rates 30% to 68% | `python -B reproduce/instability.py` (also compares the per-task counts with the archived summaries in `records/verdict-instability/archived/`) | `records/verdict-instability/verdicts.csv`: one row per task run (domain, task, repeat, run id, pass/fail, error flag), exported from the run database of the earlier internal system |

### 2. Verified against archived outputs only

| Paper item | What it reports | Check with | Why there are no raw records |
|---|---|---|---|
| Section 5.1 text | Determinism: 30 telecom scenarios replayed 3 times, 7 airline and 4 retail scenarios 4 times, no divergence in final state or predicate values (within and across processes) or tool return values (within a process) | `python -B reproduce/determinism.py` | The replay script compared the replays in memory and wrote only these counts (`records/determinism/phase8_determinism_*.json`); no per-replay record was kept |
| Section 5.4 text | $54.35 for the 444 conversations of the rule-removal run | the run's `manifest.json` | The cost was computed by litellm at run time and stored in the manifest |
| Section 5.4 text | Follow-up selection: 14 scenarios by rule tag with a fixed seed, each replay-certified | `selection.json`, `certificate.json`, `suite_followup.json`, `pool_index.json` in `experiments/rule-ablation-followup/` | `select_scenarios.py` and `certify.py` need the full engine copy whose hash they check, so they refuse on this tree |

### 3. Not reproducible from this release

| Paper item | What it reports | Reason |
|---|---|---|
| Figure 1 [fig:pipeline] | The pipeline | A diagram, no data. Code for each stage: compile output `code-fixed/generated/env_*.py`; discovery output (atoms) as above; compose, certify and filter `code-fixed/scripts/phase15_graph_suite.py` with `services/{policy_graph,atom_audit,case_gate,prefix_gate,...}.py`; render `services/narrator.py` |
| Algorithm 1 [alg:discover], scenario emission | Running discovery and emitting the 92,184 scenarios | The discovery and emission scripts and the build they ran on are in the private development repository. Their outputs (the 28 atoms and the repair-call sets of every emitted scenario) are released and give the Section 5.2 numbers above |
| Section 5.1 text | First compilation 1,795 lines, recompilation 2,241, six hand edits to 2,337; coverage, fault types and determinism measured on the 2,330-line build | The build history is in the private repository; only the final 2,337-line file is released. The records do not carry a hash of the build that produced them: they were written on 18 August 2026, before the commit that holds the 2,330-line file, and the 11 health predicates named in the emitted scenarios are the classification first committed in that file |
| Section 5.1 text | Re-running the determinism replays | Needs the private replay script and the 2,330-line build |
| Table 1 [tab:domains] | Airline and retail task counts, schema counts and policy lengths | Upstream τ³-bench data, public but not vendored here (only telecom is) |
| Section 5.5 | The conversations behind the 834 verdicts, and the simulated customer's model | The transcripts sit in the internal system's 10 GB run database and are not released; the run rows record no model names |

Also included, though not reported in this version of the paper: the same 24-scenario suite run
with the original code and its broken simulated customer (`results/full-original-2`, +29.2
points, scored by `reproduce/separation.py`), and τ³-native runs of the original and a
deep-patched telecom policy on the benchmark's 114 tasks (`results/deep-v0`,
`results/deep-v2_deep-2`; `python -B stats/cluster20.py out.csv` gives +20.0 points on the 51
tasks the patch never saw, 95% CI [10.2, 30.2]).

Each experiment folder has its own `prereg.md` (design fixed before any paid call) and
`results.md` (full tables and caveats). The rule-removal `score.json` files and `results.md`
treat the 24 scenarios as units; the paper's Tables 5 and 6 use the 19 distinct tests, as
computed in `experiments/duplicate-reanalysis/`.

## Repository layout

```
code-fixed/                 the engine the +30.0 run used, trimmed to the files the runs import
  scripts/bench.py            the benchmark entry point (both arms, one grader)
  generated/env_telecom.py    the compiled telecom environment (2,337 lines)
  generated/graph_suite_telecom.json   the generated suite (first 24 scenarios are the suite)
  generated/policy_graph_telecom.json, branches_telecom.json, env_telecom.meta.json
  generated/env_airline.py, env_retail.py (+ .meta.json)   compiled airline and retail
                              environments (Table 1; not used by any run here)
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
records/coverage/           repair-call sets of the 92,184 emitted scenarios, tau3's single-fault
                              tasks, and the archived coverage report (Section 5.2)
records/determinism/        the three archived determinism reports (Section 5.1)
records/verdict-instability/   the 834 per-run verdicts behind Table 7, and the archived
                              per-task summaries they are checked against
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
2. `reproduce/separation.py`: `scripts/score.py`'s own checks and scoring on both separation
   runs (fixed and original code).
   The recomputed full-fixed-2 score must equal the stored `score.json` byte for byte (as JSON).
3. `stats/cluster30.py` and `stats/cluster20.py`: bootstrap and sign-flip statistics with the
   24 scenarios as units (an earlier draft's analysis; the paper's 19-scenario version comes
   from step 6) and for the deep-patch runs on the 51 unseen tasks.
4. `reproduce/ablation.py`: both rule-removal runs scored in registered mode; results must
   equal each stored `score.json`.
5. `experiments/rule-coverage/coverage.py`: rewrites `coverage.csv` and `coverage.md`, which
   must be unchanged in git.
6. `reproduce/reanalysis.py`: runs `reanalyse.py` unchanged; `results.json` must be unchanged.
7. `reproduce/repair_coverage.py`: fault types and repair-call-set coverage from
   `records/coverage`, checked against the paper and the archived coverage report.
8. `reproduce/determinism.py`: the archived determinism reports against the paper.
9. `reproduce/instability.py`: Table 7 and the Section 5.5 numbers from the 834 verdicts,
   checked against the paper and the archived per-task summaries.
10. `reproduce/tests/test_records.py`: steps 7 to 9 must fail on an altered record (one
    flipped verdict, one recorded error, one missing repair set, one divergent replay).
11. `reproduce/paper_tables.py`: Tables 1 (telecom column and compiled counts) to 7, the
    points of Figure 2 and the numbers of Sections 5.1 and 5.2, each checked against the paper.
12. If the packages in `requirements.txt` are installed: `reproduce/engine_replay.py` replays
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

On 29 October 2026 this repository (commit 04b15d5) was cloned fresh from GitHub into an empty
temporary folder and `bash reproduce/verify_all.sh` was run twice: with Python 3.14.4 and only
the standard library (step 12 skipped), and with the Python 3.12.14 environment of
`requirements.txt`. Both exited 0 and left the checkout unchanged. Step 10 is the negative
control for the released records: one flipped verdict, one recorded error, one removed repair
set or one divergent replay makes the matching check fail. Two negative controls were run on an
earlier clone: flipping one criterion in one stored trace makes `separation.py` refuse
(`files differ from the seal full-fixed-2.sha256`), and committing a one-line change to
`scripts/runlog.py` makes it refuse (`manifest provenance wrong: ['wrapper']`). Changing an
expected value in `paper_tables.py` makes it report a mismatch and exit 1. Output of the run
with the full environment (step 6 shortened to its summary lines):

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

== 7. fault types and repair-call-set coverage (105/113, 90/94, 17/17) from records/coverage
emitted scenarios 92,184 (21,516 distinct repair sets match no benchmark task)
benchmark tasks 114; excluded (repair set empty): ['[service_issue]contract_end_suspension|lock_sim_card_pin[PERSONA:Hard]']
covered 105 of 113; repair-graded 90 of 94; escalation-graded tasks among the covered 15; uncovered 8
fault types 19: agent-repairable 17 (complete 16, partial 1 ['overdue_bill_suspension']), escalation-only 2
  (counting every call of each atom's fix sequence: 17 complete)
coverage: all values match the paper and the archived report

== 8. determinism replays: archived reports (summaries only) in records/determinism
telecom  30 scenarios x 3 replays; identical: in-process state 30, in-process predicates 30, in-process returns 30, cross-process state 30, cross-process predicates 30; diverged []
airline  7 scenarios x 4 replays; identical: in-process state 7, in-process predicates 7, in-process returns 7, cross-process state 7, cross-process predicates 7; diverged []
retail   4 scenarios x 4 replays; identical: in-process state 4, in-process predicates 4, in-process returns 4, cross-process state 4, cross-process predicates 4; diverged []
determinism: archived reports match the paper (no divergence)

== 9. verdict instability, Table 7, from the 834 per-run verdicts in records/verdict-instability
Table 7 (Verdict instability, three identical runs per task)
  domain   tasks runs flipped        passes/run   pass^1 pass^3 drop
  airline     50  150  17 (34.0%) [23, 31, 27]     54.0   38.0 16.0pp
  telecom    114  342  34 (29.8%) [34, 37, 30]     29.5   14.9 14.6pp
  retail     114  342  35 (30.7%) [78, 77, 77]     67.8   53.5 14.3pp
  total 278 tasks, 834 runs; no run recorded an error
instability: all values match the paper and the archived summaries

== 10. the record checks fail on altered records
OK

== 11. the paper's Tables 1-7, Figure 2 and Sections 5.1-5.2, checked against the printed values
Table 1 (Domains and compilation outcomes), telecom column and compiled tools
  telecom: tasks 114, tool schemas 43, policy length 23,318, tasks with customer actions 94, tools compiled 43/43, health predicates 11, derived observables 5, environment size 2,337 lines
  airline tools compiled 14/14; retail tools compiled 16/16
  Section 5.1: the benchmark's own telecom environment, 2,851 lines

Table 2 (The 24-scenario telecom suite)
  scenarios 24, distinct tests 19, criteria 93, requiring a call 74, no argument values 48, forbidding a call 19, repeats 5, conversations per arm 120, graded actions per arm 370

Table 3 (Policy rule coverage by policy section)
  section                                  rules covered not_cov not_chk
  General                                      8       2       2       4
  Customer lookup                              1       0       1       0
  Technical support                            1       0       1       0
  Overdue bill payment                         7       2       3       2
  Line suspension                              3       2       1       0
  Data refueling                               4       1       1       2
  Change plan                                  6       0       3       3
  Data roaming                                 2       1       1       0
  Cellular service                             9       6       3       0
  Mobile data                                  5       4       1       0
  MMS                                          5       2       3       0
  Total                                       51      20      20      11

Table 4 (Per-scenario results; # = suite index + 1)
   1  policy 18/25  stub 11/25  delta  +28.0
   2  policy 16/25  stub  9/25  delta  +28.0
   3  policy  5/20  stub  0/20  delta  +25.0
   4  policy 11/15  stub  8/15  delta  +20.0
   5  policy 15/25  stub  5/25  delta  +40.0
   6  policy 12/15  stub  7/15  delta  +33.3
   7  policy 10/10  stub  7/10  delta  +30.0
   8  policy  5/ 5  stub  3/ 5  delta  +40.0
   9  policy 20/20  stub 12/20  delta  +40.0
  10  policy  8/10  stub  4/10  delta  +40.0
  11  policy  5/ 5  stub  5/ 5  delta   +0.0
  12  policy 16/20  stub 12/20  delta  +20.0
  13  policy  5/10  stub  2/10  delta  +30.0
  14  policy 15/15  stub  5/15  delta  +66.7
  15  policy  5/ 5  stub  2/ 5  delta  +60.0
  16  policy 20/25  stub 13/25  delta  +28.0
  17  policy 15/20  stub  7/20  delta  +40.0
  18  policy  9/15  stub  5/15  delta  +26.7
  19  policy  9/10  stub  7/10  delta  +20.0
  20  policy  4/ 5  stub  5/ 5  delta  -20.0
  21  policy 20/25  stub 14/25  delta  +24.0
  22  policy 15/20  stub  9/20  delta  +30.0
  23  policy 15/15  stub  8/15  delta  +46.7
  24  policy  0/10  stub  2/10  delta  -20.0
  pooled policy 273/370 (73.8%), stub 162/370 (43.8%), separation +30.0

Figure 2 (Per-scenario separation on the 19 distinct scenarios), points sorted:
  -20, -20, 0, 20, 23.33, 24, 25, 28, 28, 30, 30, 30, 33.33, 35, 40, 40, 46.67, 50, 66.67
Discrimination (Section 5.3): 19 clusters, bootstrap 95% CI [24.7, 35.4], ahead / behind / tied 16 / 2 / 1, sign-flip p (exact) 0.00007

Table 5 (Catching removed rules; 63 valid blocks, 19 distinct scenarios)
  rule                   checks(scen) ctrl%  var%   drop   p      caught untied(p)
  R2_apn                  52 ( 7)    100.0  65.4  +34.6 0.0078 yes    -3.7 (0.249)
  R6_transfer_scope       63 (19)     69.8  54.0  +15.9 0.0581 no     -0.6 (0.234)
  R3_payment              16 ( 3)    100.0  75.0  +25.0 0.2500 no     -10.1 (0.008)
  R5_escalate_unfixable   26 ( 4)     61.5  34.6  +26.9 0.1250 no     -1.8 (0.358)
  R1_airplane             28 ( 8)     85.7  92.9   -7.1 0.7188 no     -1.4 (0.557)
  R4_suspension           24 ( 7)    100.0  91.7   +8.3 0.2500 no     -3.2 (0.230)
  caught 1 of 6; off-target untied drops >= 20 with p < 0.05: 0 of 6

Table 6 (Pre-registered follow-up and pooled secondary analysis)
  primary  R3_payment              60 ( 6)  96.7  76.7  +20.0 0.0625 no
  primary  R5_escalate_unfixable   40 ( 8)  42.5  37.5   +5.0 0.4219 no
  pooled   R3_payment              76 ( 9)  97.4  76.3  +21.1 0.0156 meets
  pooled   R5_escalate_unfixable   61 (12)  49.2  34.4  +14.8 0.0879 no

Section 5.2: coverage 105/113, repair-graded 90/94, fault types 16 complete + 1 partial of 17, 92,184 emitted
Section 5.1: determinism telecom 30x3, airline 7x4, retail 4x4
Table 7 (Verdict instability): airline 50 tasks 150 runs, 34.0% flip, drop 16.0; retail 114 tasks 342 runs, 30.7% flip, drop 14.3; telecom 114 tasks 342 runs, 29.8% flip, drop 14.6

paper tables: all values match the paper

== 12. engine replay smoke test (needs requirements.txt; skipped otherwise)
engine replay: 48 conversations, 0 errors, 110 of 186 graded actions completed (stored run, repeat 0: 118); 40 conversations grade the same as stored

ALL REPRODUCTIONS MATCH
```
