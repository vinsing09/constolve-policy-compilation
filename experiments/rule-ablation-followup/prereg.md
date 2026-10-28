# Pre-registration: rule-removal follow-up, more scenarios for R3 and R5

Written 28 Oct 2026 before any paid call of the follow-up. The main experiment (`../rule-ablation/`, results.md) stands unchanged. This is a separate run with a separate report.

**Approval.** Vineet approved this follow-up directly ("Let us do the experiments") after being told it would cost about $35-40. The coordinating session (constolve-dogfood) relayed this, and states the approval was given directly by Vineet, not by relay delegation. Hard cap for the follow-up: $45, in its own ledger (`spend-ledger.jsonl` in this folder), the main run's ledger untouched.

## Headline caveats (they belong in any headline)

1. **R3 gets 6 new scenarios, not 10. The generator cannot make more.** The scenario generator behind the 24-scenario suite emits 14 byte-distinct scenarios tagged R3 in all. 6 of them are functionally one of the suite's 3 payment scenarios: the 3 themselves, and 3 that differ only in a criterion's description text or in the letter case of one value ("Overdue" vs "overdue"). 2 more duplicate other candidates the same way. That leaves 6, and all 6 are used. They are not 6 independent situations either. All 6 are the same customer with the same overdue bill and a suspended line. They differ by an added data-allowance fault (refuel), an added roaming fault, or extra suspension fields on the line record. With 6 scenarios the smallest possible one-sided sign-flip p is 1/64.
2. **The main run's 24 scenarios are 19 distinct tests.** Rebuilding the pool shows 5 pairs of byte-identical scenarios (same walk, world, criteria and reference solution; only the model-written opening line differs): 0=15, 3=17, 6=9, 7=14, 8=11. The main run's sign-flip test counted each pair as two independent scenarios. For R5 the main run had 4 distinct scenarios, not 5 (3=17). This follow-up does not re-score the main run. The pooled secondary analysis below counts each distinct scenario once.
3. The R5 sample mixes the two escalation states the R5 span covers: 6 locked-SIM and 4 contract-ended-suspension scenarios. In the main run the one R5 scenario that did not move (23) was contract-ended. The mix comes from the seeded draw, not from any choice of mine.
4. Same limits as the main run: one agent model (gpt-4.1), one domain, one simulated customer, criteria graded by tool calls.

## Fixed inputs (identical to the main run unless stated)

- Engine copy: `code-fixed/`, registered checksum 684554a5… and tree e83f11aa… (`../rule-ablation/rules.py`). Agent gpt-4.1 at temperature 0, customer gpt-4.1 at 0.7, cap 30 turns, concurrency 8, `bench.build_agent` and `bench.one` unchanged (trace.md).
- Policies: the main run's files, `../rule-ablation/policies/control.md`, `R3_payment.md` and `R5_escalate_unfixable.md`, with hashes from its `variants.json`.
- **Tie rule, R3**: `../rule-ablation/rules.tied`, unchanged. It ties send_payment_request and make_payment.
- **Tie rule, R5: stricter than the main run** (`design.tied`, at the coordinating session's request, based on the rule-coverage analysis at commit 4d756f2). A criterion is tied to R5 only if every tool that can pass it is a call the removed R5 text directs, under a condition the scenario's starting state meets. The R5 span directs three calls, which are the rule-coverage specs C02, C04 and C09 (`experiments/rule-coverage/coverage.py`, imported unchanged and pinned):
  - line 245: check_sim_status when cellular service is broken;
  - line 247: transfer_to_human_agents when the SIM is Locked;
  - line 259: transfer_to_human_agents when the suspension cannot be lifted (contract ended).

  `diagnosed_before_handoff` (any_of check_sim_status, check_status_bar) is **not** tied: it can pass on a status-bar check, which the R5 text does not direct. The main run counted it, and the main result stays as registered.
  - The coordinator's message named check_sim_status and the Locked escalation. I also include line 259 because it is part of the removed R5 text and coverage.py ties it (scenario 23). Leaving it out would leave the 4 contract-ended scenarios with no tied criterion.
  - On the main suite this strict tie gives exactly coverage.py's ties for C02, C04 and C09: 8 criteria in scenarios 3, 5, 12, 17 and 23 (the main run had 10).
  - It does not change which scenarios are R5-tagged: every escalation scenario has a handoff tied by line 247 or 259. So the selection below is the same as under the main tie rule.
- Arms: control, R3_payment, R5_escalate_unfixable. Every arm runs every follow-up scenario.

## Scenarios

**Generator** (trace.md). These are code-original's graph-walk generator (`phase15_graph_suite.py`) and its 2,337-line engine, run by `select_scenarios.py` with the suite's settings: `max_len` 4, `atom_audit.refile`, and the atoms file at 7257eda687 (git blob 2066b1a6…). The script refuses unless all 24 suite scenarios come out byte-identical, and they do: 2,948 walks give 1,437 distinct certified scenarios. The 92,184-combination pool named in the brief is the other generator (the atom composer). Its candidates were never certified and the suite was not drawn from it, so it is not used.

**Selection rule (rule tag only, no outcome data).** For each rule r in (R3_payment, R5_escalate_unfixable):
1. Tagged: at least one of the scenario's criteria is tied to r by the tie rule above (`design.tied`).
2. Not a duplicate. Functional key = kind + world + graded criteria (description text dropped), with all strings lower-cased (`design.functional_key`). Scenarios whose key equals any suite scenario's are excluded. Of scenarios sharing a key, the lowest walk index is kept.
3. Draw `min(10, n)` with `random.Random("20261028:<rule>").sample` over the candidates in walk order.

Result (`selection.json`):
- R3: 14 tagged, 6 functional duplicates of the suite (including the suite's own 3), 6 candidates after the remaining 8 collapse to 6 functional keys. All 6 picked: walks 2924, 2928, 2929, 2936, 2937, 2946.
- R5: 457 tagged, 4 duplicates, 453 candidates. 10 picked: walks 90, 97, 372, 702, 715, 1245, 1255, 1320, 1396, 2680.

The two sets are disjoint (R3 scenarios are repair cases; R5 ties require escalation cases).

**Replay certificate** (`certify.py`, before narration). `case_gate.check` replays each scenario's reference solution on its own world in code-fixed's engine. This is the gate every suite scenario passed. Result: 16 of 16 certified, none dropped. `prefix_gate.check` (each step still needed when taken) also passes 16 of 16, and 24 of 24 suite scenarios.

**Customer brief** (`narrate.py`). The same path as the suite: `meta_user.build` writes the opening complaint with the code's default narrator model, claude-opus-5. Everything else is assembled from the world, then the roaming note is added. **Added to every scenario: "You are contacting support by web chat from another device."**, appended to `known_facts.known_info`, which only the customer sees. It is the same in all three arms. The main run's scenarios did not have this line, a difference the pooled analysis carries. Narration spend goes in the follow-up ledger. Once `suite_followup.json` is written, its sha256 goes into `design.SUITE_SHA` by amendment. The runner and scorer refuse any other file.

## Run

3 arms × 16 scenarios × 5 repeats = 240 conversations, ordered repeat, then scenario, then arm. Scenario order is R3's 6 (indices 0-5), then R5's 10 (6-15).

**Budget.**
- Flat estimate: 240 × $0.1224 (the main run's actual cost per conversation) = $29.38.
- From the main run's costs for these scenario types (payment about $0.142, escalation about $0.069 per conversation): about $23.
- Narration: under $1.
- Cap: $45 in the follow-up ledger. The runner refuses if the estimate exceeds $45 minus the ledger.
- Guard, same design as the main run (`ledger.py`): each call reserves its worst case first (every input character a token plus the most output it may return) and is refused if that could pass the line. The reservation goes to the fsynced ledger before the call, and a settling line follows after it. Conversations start only with the main run's headroom rule.
- No reduction to 8 scenarios per rule is needed: the estimate is under the cap.

**Block rule** (as the main run). A (repeat, scenario) block counts only if all 3 arms' conversations are clean. Blocks lost to the budget guard are dropped and reported. If more than 10% of blocks are lost to failures, the run does not count.

## Analysis plan

**Primary (follow-up alone), per rule r.**
- Tied criteria: those the tie rule above ties to r in the 16 scenarios. R3: send_payment_request and make_payment in its 6 scenarios (12 criteria). R5: the handoff in each of its 10 scenarios, plus check_sim_status in walk 90, where it is graded on its own (11 criteria). In the other 9, the check is a `diagnosed_before_handoff` criterion, which is untied.
- Drop = control pass % − variant pass % over valid blocks.
- Test: one-sided paired sign-flip over scenarios on d_s = control fraction − variant fraction. Exact up to 20 nonzero d_s (`../rule-ablation/score_ablation.signflip_p`, unchanged).
- **Caught = drop ≥ 20 points AND p < 0.05.**
- False alarm: on all criteria not tied to r (in all 16 scenarios), the variant's pass rate minus control's; a drop ≥ 20 with p < 0.05.

**Secondary, labelled as such: main + follow-up pooled, per rule.**
- Rows from the main run's valid blocks (its control and r arms; the main run is loaded through its own scorer's registered checks) plus the follow-up's.
- The same tie rule for both runs: R5 strict, so the main run's R5 rows are tied by the strict rule here, not by its registered ties.json.
- Scenario unit = functional key, so main's 3 and 17 count as one scenario. Follow-up scenarios never share a key with a main scenario, by construction.
- Same drop, p and caught definitions.
- Caveats: the arms ran on different days, with 3 vs 5 repeats, and with vs without the web-chat line.

**Expectations (not part of the test).**
- Main run: R3 dropped 25 points (control 100%), R5 dropped 26.9 (control 61.5%).
- If the effects are real and of that size, R5 (10 scenarios) should be caught.
- R3 needs at least 5 of its 6 scenarios to drop with none rising.

## Amendment 1 (28 Oct, after the first narration attempt, before any conversation)

- **Narrator model is gpt-4.1, not claude-opus-5.** All 16 calls to the code's default narrator model failed with "invalid x-api-key": the Anthropic key in the copies' `.env` is dead. One 5-token probe of the same model through the `.env`'s AWS Bedrock token also failed authentication. `.env` is part of the frozen tree hash and cannot be changed here, so the opening complaint is written by gpt-4.1. That model has a working key and is the run's own agent and customer model. `meta_user.build(..., model="gpt-4.1")` keeps the same prompt, 600-token limit, contradiction gate and 2 attempts. Everything else in the brief is assembled without a model, as before.
- Effect: the wording of each customer's opening complaint comes from a different model than the suite's did. That is identical across the three arms, so the primary within-run comparison is unaffected. It is one more difference between the main run's scenarios and these, alongside the web-chat line, for the pooled analysis.
- Spend so far, in the follow-up ledger: $0.3824 charged at worst case for the 16 failed calls (they cost nothing in fact; the guard charges failed calls their worst case by design), plus $0.0002 entered by hand for the Bedrock probe, which ran outside the guard.

## Amendment 2 (28 Oct, after narration, before any conversation)

- **Registered scenario file:** `suite_followup.json`, sha256 6253fd03…db20 (`design.SUITE_SHA`). It holds 16 scenarios: R3's 6 (indices 0-5), then R5's 10 (6-15). Narration cost $0.016 with gpt-4.1. Narration changed only the brief fields; world, criteria, reference solution and symptoms are byte-identical to `selection.json` (tested). Every brief ends with the web-chat line, once.
- **Tie sets on this file:** R3 has 12 criteria, R5 has 11.
- **Expected consequence of the strict tie (stated now so it is not read later as a finding):** the `diagnosed_before_handoff` criteria in 9 R5 scenarios count as untied for R5. The removed R5 text includes the "use check_sim_status" sentence, so a drop on them is plausible and would appear in R5's untied change (the false-alarm check). It is reported as it falls.
- **Spend before the run:** $0.3986 in the follow-up ledger (failed narration worst-case charges $0.3824, the probe $0.0002, and narration $0.016). The runner's line is $45 − ledger = $44.60. Estimate 240 × $0.1224 = $29.38.

## Amendment 3 (28 Oct, after Codex audit r1 and an unplanned paid run, before the registered run)

- **Unplanned spend: $19.2521** (errors.md 4). A test I ran while checking my audit fixes started a real run on code-fixed: 3,892 gpt-4.1 calls, killed after 10 minutes. It is charged to the follow-up ledger: $16.701 settled plus 8 in-flight calls at worst case. It is not used in any analysis: unregistered, unsealed, outcomes not opened. Ledger now **$19.650498** (7,857 rows), leaving $25.35 under the $45 cap.
- **R5 cut from 10 to 8 scenarios**, as the brief requires when the estimate would pass the cap. At 16 scenarios the amendment-2 flat estimate is $29.38, over $25.35. `reduce_r5.py` keeps all 6 R3 scenarios (fewer than 8 exist) and a seeded sample (`"20261028:R5_escalate_unfixable:reduce8"`) of 8 of the 10 narrated, certified R5 scenarios, in order. No outcome data is used.
  - Dropped: walk 90 (locked SIM; the only R5 scenario grading check_sim_status on its own) and walk 715 (contract-ended).
  - Kept: 97, 372, 702, 1245, 1255, 1320, 1396, 2680. That is 5 locked-SIM and 3 contract-ended.
  - The registered file is now `suite_followup.json`, sha256 8885530081d4…7e47 (14 scenarios: indices 0-5 R3, 6-13 R5). The 16-scenario file is kept as `suite_followup_16.json`.
  - R5's tie set is now the 8 handoffs. R3's is unchanged (12).
  - Smallest possible p: R3 1/64, R5 1/256.
- **Estimate.** 3 arms × 14 × 5 = 210 conversations. The flat $0.1224 rate gives $25.70, just over the line. So the runner's refusal check now uses the main run's actual cost per conversation by scenario kind, over the control, R3 and R5 arms: $0.1417 for payment (repair) scenarios and $0.0694 for escalation scenarios. That gives 90 × 0.1417 + 120 × 0.0694 = **$21.08**. The per-call guard still enforces the hard line. If costs run high, the last blocks are skipped and dropped by the block rule, as in the main run.
- **Audit r1 fixes (both blocking findings accepted):**
  1. The runner validates the ledger before reading its sum (`ledger.prior_spend`). It must be append-only over a registered prefix (7,857 rows, sha256 ec8bae36…) and well formed: reservations ≥ 0, each settlement no larger than its own reservation, no other row kinds after the prefix. The ledger is also in the runner's uncommitted-changes check.
  2. The registered scorer requires all 14 case ids in order and `cases == 14`. It reconciles every field of the recomputed run summary, not only the per-conversation status.
