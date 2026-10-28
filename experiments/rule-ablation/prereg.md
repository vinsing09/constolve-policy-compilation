# Pre-registration: rule-removal experiment on the frozen telecom engine

Written 28 Oct 2026 before any paid call. Owner approval relayed by the coordinating session: 3 repeats, hard total cap $60.

## What is being measured

If one rule is cut from the telecom policy, does the 24-scenario test suite notice? "Notice" means the criteria that depend on that rule pass less often than with the full policy, and the criteria that don't depend on it stay put.

## Fixed inputs

- Engine: `code-fixed/` (the +30.0 run's copy, tree hash in `scripts/tree-fixed.txt`), run through its own `scripts/bench.py` `build_agent` and `one` (trace.md). Agent gpt-4.1 at temperature 0, customer gpt-4.1 at 0.7, cap 30 turns, concurrency 8.
- Scenarios: the first 24 cases of `code-original/generated/graph_suite_telecom.json` (sha256 3c238d13…, the same file in code-fixed), the set score.py uses.
- Control policy: `env.get_policy()` from code-fixed's vendored tau3, sha256 065b4850…c3e9, saved as `policies/control.md`. Line numbers below refer to that file.

## (b) Every graded criterion and the policy sentence that requires it

The suite only uses two rule types in these 24 cases: `tool_called_with` (some with an order constraint, "before") and `tool_not_called`. It has no `tool_called_before` criteria. Counts are criteria across the 24 scenarios.

| criterion (tool) | n | scenarios | policy line(s) that require it |
|---|---|---|---|
| toggle_airplane_mode (user) | 11 | 0,4,7,8,11,14,15,18,20,21,22 | 239 (also 238, 292 by reference) |
| reset_apn_settings (user, before reboot) | 10 | 0,4,6,8,9,11,15,20,21,22 | 252 (service), 359 (MMS) |
| reboot_device (user) | 11 | the 10 above + 18 | 253, 360 (after an APN reset); 129 (after resume_line). Case 18 has neither an APN reset nor a resume: no explicit sentence |
| resume_line (assistant) | 9 | 0,1,8,10,11,13,15,16,20 | 126 (may lift after overdue bills are paid), 258 |
| send_payment_request (assistant, before make_payment) | 3 | 1,13,16 | 108, 118 |
| make_payment (user, before resume_line) | 3 | 1,13,16 | 112 |
| refuel_data (assistant) | 2 | 1,19 | 134-139 (also 318) |
| enable_roaming (assistant) | 2 | 1,16 | 156 (also 301-302) |
| toggle_roaming (user) | 2 | 2,20 | 300 |
| toggle_data (user) | 3 | 0,15,21 | 296 |
| toggle_data_saver_mode | 1 | 2 | 306 |
| disconnect_vpn | 1 | 2 | 311 |
| set_network_mode_preference | 1 | 2 | 324 (also 354) |
| toggle_wifi_calling | 1 | 4 | 364 |
| grant_app_permission | 1 | 4 | 368 |
| check_sim_status (user, before transfer) | 3 | 3,12,17 | 245 |
| diagnosed_before_handoff: check_sim_status or check_status_bar, before transfer | 2 | 5,23 | 245; 232, 238 |
| get_data_usage or get_details_by_id (assistant, before transfer) | 3 | 3,5,17 | no sentence names it; nearest are 313-315 (data limit) and 160 (identify the customer). Inferred, not explicit |
| transfer_to_human_agents (must call) | 5 | 3,5,12,17,23 | 14 ("if and only if" out of scope); 247 (locked SIM); 259 (suspension the agent cannot lift) |
| no_premature_escalation: transfer_to_human_agents NOT called | 19 | all repair cases | 14 ("only if"), 16, 171 |

**Tie rule** (code: `rules.tied`, frozen list: `ties.json`). A criterion is tied to a removed rule when the removed text tells the agent to make that criterion's tool call (its tool or an `any_of` tool) under a condition the scenario meets. Order constraints ("before") are not tied separately. Every other criterion is "untied" for that rule.

## (c) The six removed rules and why

The "measurable" test looked at two things. First, how often each criterion passed in full-fixed-2. Second, how much the policy matters for it: full-fixed-2's with-policy arm against its one-sentence stub arm. A rule whose tied criteria already fail with the full policy (roaming, mobile data, MMS: 0-10% pass) can't show a drop. A rule the model follows without being told (resume_line: 98% with the policy, 100% with the stub) probably won't show one either.

| id | aspect (policy heading) | text removed (lines) | tied criteria | scenarios | full-fixed-2 tied pass: policy / stub |
|---|---|---|---|---|---|
| R1_airplane | cellular service: Airplane Mode | 239 | 11 | 11 | 94.5 / 43.6 |
| R2_apn | network settings: APN reset + reboot | 252-253, 359-360 | 20 | 10 | 96.0 / 53.0 |
| R3_payment | billing: Overdue Bill Payment | 108-112, 118 | 6 | 3 | 100.0 / 0.0 |
| R4_suspension | line suspension: permission to lift | 126 | 9 | 9 | 97.8 / 100.0 |
| R5_escalate_unfixable | escalation, specific: check SIM; escalate locked SIM or unliftable suspension | 245-248, 259 | 10 | 5 | 44.0 / 18.0 |
| R6_transfer_scope | escalation, general: transfer only if out of scope, after trying everything | 14 (first sentence), 16, 171 | 24 | 24 | 65.0 / 41.7 |

Why each:
- **R1** and **R2** are the two troubleshooting rules the policy clearly drives: big policy-vs-stub gaps, 10-11 scenarios each. R2 removes both statements of the APN rule (service and MMS) together with their "then reboot" sentences. Removing only one statement would leave the other in place, and the dangling "After resetting the APN settings…" sentence would still give the rule away.
- **R3** is the clearest rule the policy drives (100 vs 0), but it covers only 3 scenarios (see the power limit below).
- **R4** is a deliberate "the model does this anyway" rule. The prediction is no catch. If it comes out caught, that is a false-positive signal about the method.
- **R5 / R6 are the overlap pair.** Both govern `transfer_to_human_agents` but with different conditions. R5 is specific (escalate after a SIM check shows it is locked, or when a suspension can't be lifted). R6 is general (transfer only when the request is out of scope, and only after trying everything). They share the 5 must-transfer criteria. R5 alone owns the 5 SIM/diagnosis criteria (3,5,12,17,23). R6 alone owns the 19 no-premature-escalation criteria.
- Not chosen: roaming, mobile data, MMS, refuel (tied criteria pass 0-40% even with the full policy, or only 1-2 scenarios), and identity/lookup (no graded criterion names a lookup tool except an `any_of` fallback in 3 scenarios).

**Deviations from the brief, stated up front.**
1. Aspects are the policy's own headings. The brief's examples put airplane mode and APN under one aspect, and suspension and billing under one. Under that grouping only four measurable aspects exist in this suite (troubleshooting, billing/suspension, escalation, and arguably mobile data, which can't show a drop). Six rules from different aspects are not possible in that taxonomy. Each chosen rule sits under a different heading and has different tools, except the deliberate pair.
2. **Power limit (headline caveat).** A one-sided sign-flip test over k scenarios can't give p below 1/2^k. **R3 (3 scenarios, minimum p = 0.125) can never be "caught" by the pre-registered rule, however large its drop.** R5 (5 scenarios, minimum p = 0.031) can be caught only if all five scenarios drop. The brief's "≥2 scenarios" threshold does not make a catch measurable under its own test; 5 is the minimum. R3 is kept to show this limit with real data. It is reported, not counted as a miss of the suite.
3. Rules were chosen after looking at full-fixed-2's per-criterion pass rates (a power consideration). This favours rules that can show a drop. It does not touch the variant runs, which are new data.

## (d) Variant files

`policies/<rule>.md` is the control with exactly the spans in `rules.py` cut out. Each span occurs exactly once in the control (tested). The minimal diff is `policies/<rule>.diff`, and the hashes are in `variants.json`. `make_variants.py` rebuilds all of them from the engine's own policy and refuses if the control hash differs.

## (e) Analysis plan

**Arms.** Control (full policy, `build_agent(domain, None)`, the same call as full-fixed-2) plus the 6 variants. Each arm runs 24 scenarios × 3 repeats = 72 conversations, 504 in all. Order: repeat, then scenario, then arm, so the arms run interleaved in time.

**Block rule.** A (repeat, scenario) block counts only if all 7 arms' conversations in it are clean: no error, no failed model call, no customer error, and the agent's prompt is that arm's policy (checked on every agent call). Blocks lost to the budget stop are dropped and reported. If more than 10% of blocks (more than 7 of 72) are lost to failures, the run does not count.

**Primary measure, per variant.** The pass rate of its tied criteria in the variant against the control, pooled over valid blocks. Drop = control % − variant %.

**Test.** For each scenario s with tied criteria: d_s = control pass fraction − variant pass fraction on those criteria. T = Σ d_s. The one-sided paired sign-flip p is P(Σ ±d_s ≥ T). It is exact (enumerated, observed assignment counted) when there are ≤ 20 nonzero d_s, otherwise Monte Carlo with 200,000 draws, seed 20261028, p = (hits+1)/(N+1). Scenarios with d_s = 0 drop out.

**Caught** = drop ≥ 20 points AND p < 0.05.

**Secondary measures.**
- Untied change: the variant's pass rate on all criteria not tied to the rule, minus the control's, with the same test. A **false alarm** is an untied drop ≥ 20 with p < 0.05.
- Overlap pair: for R5 and R6, the same test on own-only criteria and on partner-only criteria. A catch "lands on the right rule" when own-only is caught and partner-only is not.
- Spend, conversations per status, and per-arm cost.

**Expectations** (from full-fixed-2; not part of the test): R1, R2 caught; R3 large drop but not caught (power limit); R4 not caught; R5, R6 uncertain (control rates 44% and 65%).

**Budget.** Estimate: 504 × $0.1147 (full-fixed-2's with-policy cost per conversation) = $57.81. With the smoke test (2 conversations, ~$0.25), that leaves about $2 of headroom under $60. The driver refuses to start if the ledger (all calls in `results/*/calls.jsonl`) plus the estimate exceeds $60 − $0.50. While running, it starts no conversation once spend + $0.10 per in-flight conversation + $0.20 would pass that line, and any model call fails once spend reaches it. If conversations run longer than in full-fixed-2, the last blocks of repeat 2 may be skipped; the block rule handles that.

## Caveats (they belong in any headline)

- 3 repeats, one model (gpt-4.1), one domain (telecom), one simulated customer. Per-scenario rates are in thirds.
- R3 can't be caught by construction, and R5 only if every one of its 5 scenarios drops.
- Pass rates on the criteria are what is measured. "Actions completed" mostly means "the right tool was called" (knowledge.md, 20 Oct). Order constraints matter only when the later tool was called.
- The control is a fresh run, not full-fixed-2's with-policy arm. Full-fixed-2 (5 repeats, 20 Oct) is reported beside it only as a drift check.
- The R3 and R5 spans also remove sentences no criterion grades (R3: "inform the user, check_payment_request"; R5: "Missing → reseat SIM", "Active → OK"). Any effect of those shows up as untied change.

## Amendment 1 (28 Oct, after my own code review, before any paid call)

- Hard-stop margin is $0.30, not $0.50: at most 8 calls are in flight, about $0.03 each. The line is $60 − ledger − $0.30.
- A conversation cut off by the hard stop is classed "budget_stopped", like a skipped one: its block is dropped and it does not count as a failure toward the 10% limit. Without this, one trip of the hard stop (up to 8 in-flight conversations) could have voided the whole run for a reason that has nothing to do with data quality.
- The scorer refuses a run whose policy hashes differ from `variants.json`, or if `ties.json` differs from `rules.tied` on the suite.

## Amendment 2 (28 Oct, after Codex audit r1, before the main run; only the 2-conversation smoke has been paid for)

- **Spend cap is now hard by construction.** Before each model call the guard reserves that call's worst case: every input character counted as a token, plus gpt-4.1's maximum output of 32,768 tokens, at litellm's gpt-4.1 prices (about $0.35 for a full agent call). A call is refused if spend + all open reservations + its own worst case would pass $60 − ledger. Real cost is added when the call returns; for a failed call, the worst case is added. The ledger is always `experiments/rule-ablation/results/`, plus the run's own folder, and can't be pointed elsewhere from the command line.
- A conversation starts only if spend + open reservations + $0.10 per running conversation + $0.20 + 8 × $0.36 stays under the line. This leaves about $4 unspent. **Expected effect: roughly the last 15-20 conversations (2-3 of the 72 blocks, in repeat 2) are skipped, not run.** The block rule drops them from all arms.
- The scorer accepts only the registered design: all 7 arms, the 24 cases, 3 repeats, cap 30, concurrency 8, code-fixed with its stored checksum and tree hash, the copy unchanged, the plain command (no `--copy`, `--results` or `--arms`), and a project commit that is an ancestor of HEAD with the pinned files unchanged since. The runner refuses a non-code-fixed copy and, for a real run, uncommitted changes to the pinned files.
- The scorer recomputes the runner's conversation checks from the saved raw files (`rows.json`, `calls.jsonl`, `customer_errors.jsonl`, `prompts.json`, `stopped.json`, traces) and refuses if they differ from the manifest. Duplicate traces for one conversation are refused by both runner and scorer.

## Amendment 3 (28 Oct, after Codex audit r2, before the main run)

- **One durable ledger.** `spend-ledger.jsonl` gets one line per guarded call, fsynced, whatever results folder a run uses: the actual cost, or the worst case for a failed call or a call with unknown cost. The cap check reads only this file. It was seeded with the smoke run's total ($0.327, all calls succeeded), which predates the ledger.
- **Registered engine hashes are literals** in `rules.py`. The checksum 684554a5… (all files except `.env`) is full-fixed-2's. The tree hash is e83f11aa…: full-fixed-2's 35d1fbb9… with only the `.env` API-key line changed, re-recorded by commit 60e0cb8 on 27 Oct. The code is identical; full-fixed-2's tree hash can no longer be reproduced. The runner refuses any other copy, and the scorer refuses any other manifest value.
- **Pinned files now also include** `score_ablation.py`, `scripts/score.py`, `prereg.md`, `scripts/checksum-fixed.txt` and `scripts/tree-fixed.txt`. The scorer records its own commit in `score.json`.

## Amendment 4 (28 Oct, after an independent review of the r2 fixes, before the main run)

- Each call writes its worst case to the ledger before the call ("reserve_worst_case"), then a settling line (actual − worst) after it. A crash mid-call leaves the worst case charged.
- An exclusive lock on `spend-ledger.lock` is taken before the spend is read and held until the run ends, so only one paid run of this experiment can run at a time.
- `--copy` and `--results` exist only in fake-model test mode. A paid run always uses `code-fixed/` and `experiments/rule-ablation/results/`.
- The scorer also refuses uncommitted changes to the pinned files.
