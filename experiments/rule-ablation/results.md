# Rule-removal experiment: results

Run `main`, sealed at commit dc122fd (seal `seals/main.sha256`). Scored by `score_ablation.py`, re-run on 28 Oct 2026 with output identical to `results/main/score.json`. Design and criteria: `prereg.md` (with amendments 1-4), fixed before any paid call.

## Verdict (pre-registered criterion)

**1 of 6 removed rules caught.** Caught means drop ≥ 20 points AND one-sided paired sign-flip p < 0.05 on the criteria tied to the rule. Only R2 (APN reset + reboot) meets both. 3 repeats, one agent model (gpt-4.1), one domain (telecom), one simulated customer, 63 of 72 blocks valid.

## Results

| removed rule | tied criteria | control pass % | variant pass % | drop (pts) | p | caught? | untied change (pts, p) |
|---|---|---|---|---|---|---|---|
| R1_airplane | 28 in 11 scenarios | 85.7 | 92.9 | −7.1 | 0.797 | no | −1.4 (0.564) |
| R2_apn | 52 in 10 scenarios | 100.0 | 65.4 | +34.6 | 0.002 | **yes** | −3.7 (0.315) |
| R3_payment | 16 in 3 scenarios | 100.0 | 75.0 | +25.0 | 0.250 | no | −10.1 (0.008) |
| R4_suspension | 24 in 9 scenarios | 100.0 | 91.7 | +8.3 | 0.250 | no | −3.2 (0.239) |
| R5_escalate_unfixable | 26 in 5 scenarios | 61.5 | 34.6 | +26.9 | 0.0625 | no | −1.8 (0.413) |
| R6_transfer_scope | 63 in 24 scenarios | 69.8 | 54.0 | +15.9 | 0.037 | no | −0.6 (0.152) |

"Drop" is control minus variant on tied criteria (positive = the variant passes less). "Untied change" is the variant's pass rate on all other criteria minus the control's (negative = the variant passes less). No variant has an untied drop ≥ 20 with p < 0.05, so there is no false alarm under the pre-registered definition.

**Overlap pair (R5 / R6).**

| variant | own-only drop | p | partner-only drop | p | lands on the right rule |
|---|---|---|---|---|---|
| R5_escalate_unfixable | +30.8 | 0.125 | +12.0 | 0.137 | no (own-only not caught) |
| R6_transfer_scope | +22.0 | 0.015 | +30.8 | 0.125 | yes (own-only caught, partner-only not) |

**Per-scenario changes on tied criteria** (control passes / variant passes, pooled over valid repeats; scenarios with no change omitted):

- R1: 0 (2/3 → 3/3), 7 (3/3 → 2/3), 8 (2/3 → 3/3), 11 (2/3 → 3/3), 18 (2/2 → 1/2), 20 (1/2 → 2/2). Two scenarios down, four up.
- R2: 9 of 10 scenarios down, none up (scenario 15 unchanged). p = 1/512.
- R3: scenarios 1 and 13 down (6/6 → 4/6 each); scenario 16 unchanged. With 2 changed scenarios the smallest possible p is 0.25, which is what was observed.
- R4: scenarios 13 (3/3 → 2/3) and 15 (2/2 → 1/2) down; the other 7 unchanged.
- R5: scenarios 3, 5, 12, 17 down; scenario 23 unchanged. With 4 changed scenarios the smallest possible p is 1/16 = 0.0625, which is what was observed.
- R6: 15 scenarios changed, 11 down and 4 up.

**Drift check against full-fixed-2** (with-policy arm, 5 repeats, 20 Oct; different blocks, so indicative only), tied pass % control here vs there: R1 85.7 / 94.5, R2 100.0 / 96.0, R3 100.0 / 100.0, R4 100.0 / 97.8, R5 61.5 / 44.0, R6 69.8 / 65.0.

## Spend

| | USD |
|---|---|
| main run, total | 54.3454 |
| control | 7.7786 |
| R1_airplane | 7.9312 |
| R2_apn | 7.8818 |
| R3_payment | 7.2076 |
| R4_suspension | 7.6068 |
| R5_escalate_unfixable | 8.3236 |
| R6_transfer_scope | 7.6158 |
| smoke run (before the main run) | 0.327 |
| **experiment total** | **54.6724** |

The durable ledger `spend-ledger.jsonl` sums to $54.672354, matching the run's guard ($54.3454) plus the smoke seed ($0.327). Cap was $60 (hard stop $59.67 after the $0.30 margin). 11,682 model calls, 0 failed, 1 SDK retry.

## Why 9 of 72 blocks are invalid

All 9 were lost to the budget guard's start rule, not to failures. 444 conversations ran clean; 60 were **skipped** (never started). No conversation errored, none was cut off mid-run (`stopped.json` is empty, so there are no "budget_stopped" conversations), no model call failed, and `blocks_lost_to_failures` is 0. The 10% failure limit therefore does not apply and the run counts.

- The skipped conversations are all in the last repeat (repeat index 2), scenarios 15-23: 4 of 7 arms in scenario 15 (R3-R6), all 7 arms in scenarios 16-23. Under the block rule any block with a skipped arm is dropped from every arm, so 9 blocks (repeat 2 × scenarios 15-23) are invalid.
- Mechanism (amendment 2): a conversation starts only if spend + open worst-case reservations + $0.10 per running conversation + $0.20 + 8 × $0.36 ≤ $59.67. That keeps roughly $3 of start headroom plus the in-flight reservations. Once spend reached about $54.3 with 8 calls reserved, no new conversation could start, and the in-flight ones finished normally.
- Conversations cost more than estimated: $54.3454 / 444 = $0.1224 each against the pre-registered $0.1147 (+6.7%). Amendment 2 predicted 2-3 blocks skipped; 9 were.
- Incomplete conversations played no part. One conversation (R4, repeat 2, scenario 2) reached the 30-turn cap; that is a normal graded outcome, not a failure, and its block is valid.

Consequence: scenarios 15-23 have 2 valid repeats instead of 3. This removes repeat-2 data for scenario 16 (R3's third scenario), scenario 23 (one of R5's five) and several R1, R2, R4 and R6 scenarios.

## Interpretation (factual)

- **Drop ≥ 20 is met by R2, R3 and R5; p < 0.05 only by R2.** R2 is the single catch.
- **R3 is underpowered by construction.** Its 3 scenarios give a minimum possible p of 0.125, so it could never be caught under the pre-registered test, whatever the drop (prereg, deviation 2). Only 2 of the 3 changed, so the observed p is 0.25. The 25-point drop is large but not evidence under this test.
- **R5 is one scenario short of the minimum p.** 4 of its 5 scenarios dropped and the fifth (23) did not change. A catch required all 5 to drop (p = 1/32); 4 of 5 gives at best 1/16 = 0.0625, which is what was observed.
- **R6 is significant but under 20 points.** p = 0.037, drop 15.9. Its own-only criteria (the 19 no-premature-escalation checks) drop 22.0 points with p = 0.015, which is caught on that subset; the shared must-transfer criteria pull the pooled figure down.
- **R1 and R4 show no drop.** Without the rule, the tied steps were still performed: R1 92.9% (above control's 85.7%), R4 91.7% (2 of 24 criteria lost, p = 0.25). The model performs those steps without the rule, so no violation arises. Both removed spans are single sentences; the R1 variant file still describes airplane mode as blocking cellular service (its lines 223, 237, 273) and still refers to "instructions on how to check and turn off Airplane Mode" (its line 291), and the R4 variant still says the suspension may not be lifted when the contract has ended "even if the user has paid", which implies the permitted case. R4 was pre-registered as a predicted non-catch. This is a limit of measuring recall by rule removal: a "miss" can mean no violation occurred, not that the suite failed to see one. The 1-of-6 figure should be read with that in mind; it is not a recall estimate over rules whose removal produced violations.
- **R3's untied change of −10.1 points (p = 0.008)** means removing the payment rule also changed other behaviour: across the 24 scenarios, criteria not tied to the payment rule passed 65.6% under the variant against 75.8% under control. It is below the 20-point false-alarm threshold. The R3 span also removed sentences no criterion grades (prereg caveats), and the drops are spread across many troubleshooting scenarios (0, 2, 5, 10, 11, 17, 18, 21 among others).
- 3 repeats is the minimum design; per-scenario rates are in thirds (halves for scenarios 15-23). Adequate power needs more scenarios per rule (at least 5 changed scenarios for any p < 0.05; more for robustness) and more repeats.
