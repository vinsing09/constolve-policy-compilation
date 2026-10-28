# Rule-removal follow-up: results

Run `main`, sealed (`seals/main.sha256`, committed). Scored by `score_followup.py` (registered checks passed). Design: `prereg.md` with amendments 1-3, all committed before the run; Codex audit r4 APPROVE. The main experiment's result (`../rule-ablation/results.md`, 1 of 6 caught) stands unchanged. This is a separate run.

## Verdict (pre-registered criterion)

**Neither rule is caught in the follow-up.** Caught means a drop of at least 20 points AND a one-sided paired sign-flip p < 0.05 on the tied criteria. Setup: 3 arms, 5 repeats, 14 new scenarios (6 for R3, 8 for R5), 70 of 70 blocks valid, gpt-4.1 agent and customer, one domain.

- **R3 (overdue-bill payment)** dropped 20.0 points. Only 4 of its 6 scenarios changed (all down), so the smallest possible p was 1/16, and p = 0.0625. This is the same power floor that limited R3 in the main run, now one step higher.
- **R5 (SIM check / escalate when unfixable)** showed no drop on its strictly tied criterion, the handoff: 42.5% vs 37.5%, p = 0.42.

**Secondary (pooled with the main run, labelled secondary):**
- R3 meets the criterion: 21.1 points, p = 0.016, over 9 distinct scenarios.
- R5 does not: 14.8 points, p = 0.088, over 12 distinct scenarios.
- This is not a pre-registered primary result. The two runs differ in date, repeats (3 vs 5), the customer's web-chat line and the narrator model.

## Headline caveats

1. **R3's 6 scenarios are variants of one situation.** They have the same customer, the same overdue bill and the same suspended line, with different add-on faults. The generator cannot make more R3 scenarios that are not functionally suite duplicates (prereg caveat 1).
2. **R5 was cut from 10 to 8 scenarios** to stay under the $45 cap, after an unplanned paid run I triggered during testing used $19.25 of it (errors.md 4, amendment 3).
3. **R5 uses a stricter tie than the main run.** Only the handoff (policy lines 247 and 259) is tied. `diagnosed_before_handoff`, which can pass on a status-bar check, is untied. On that untied check the R5 variant fell from 19/40 (47.5%) to 11/40 (27.5%), against 21/40 under the R3 variant. Under the main run's tie rule it would have counted toward R5. This was stated in advance as a plausible untied drop (amendment 2). It is reported here, not counted.
4. **The main run's 24 scenarios are 19 distinct tests.** 0=15, 3=17, 6=9, 7=14 and 8=11 are byte-identical apart from the opening line. The main run's sign-flip test treated each pair as independent. The pooled analysis counts each distinct scenario once. The main run's registered numbers are not re-scored here.
5. The customer's opening complaint was written by gpt-4.1, not by the suite's narrator (claude-opus-5): its credentials in `.env` are dead (amendment 1). Every customer was also told they are on web chat from another device. Both are the same in all arms.

## Results, follow-up alone (primary)

| removed rule | tied criteria | control pass % | variant pass % | drop (pts) | p | caught? | untied change (pts, p) |
|---|---|---|---|---|---|---|---|
| R3_payment | 60 in 6 scenarios | 96.7 | 76.7 | +20.0 | 0.0625 | no | −1.1 (0.324) |
| R5_escalate_unfixable | 40 in 8 scenarios | 42.5 | 37.5 | +5.0 | 0.4219 | no | −2.4 (0.037) |

"Drop" is control minus variant on tied criteria (positive = the variant passes less). "Untied change" is the variant minus control on all other criteria in the 14 scenarios (negative = the variant passes less). No untied drop reaches 20 points, so there is no false alarm.

**Per scenario, tied criteria** (control passes / variant passes over 5 repeats):
- R3, 2 criteria each (send_payment_request, make_payment):
  - walk 2924: 10/10 → 10/10
  - walk 2928: 8/10 → 4/10
  - walk 2929: 10/10 → 10/10
  - walk 2936: 10/10 → 7/10
  - walk 2937: 10/10 → 8/10
  - walk 2946: 10/10 → 7/10

  By criterion: send_payment_request 29/30 → 24/30, make_payment 29/30 → 22/30.
- R5, handoff only:
  - walk 97: 0/5 → 0/5
  - walk 372: 4/5 → 1/5
  - walk 702: 3/5 → 4/5
  - walk 1245: 1/5 → 2/5
  - walk 1255: 2/5 → 2/5
  - walk 1320: 3/5 → 2/5
  - walk 1396: 3/5 → 2/5
  - walk 2680: 1/5 → 2/5

  3 down, 3 up, 2 unchanged.

## Results, main + follow-up pooled (secondary)

Same tie rules as above (R5 strict for both runs). Unit = functionally distinct scenario.

| removed rule | tied criteria | control pass % | variant pass % | drop (pts) | p | meets criterion? | untied change (pts, p) |
|---|---|---|---|---|---|---|---|
| R3_payment | 76 in 9 scenarios | 97.4 | 76.3 | +21.1 | 0.0156 | yes | −6.1 (0.008) |
| R5_escalate_unfixable | 61 in 12 scenarios | 49.2 | 34.4 | +14.8 | 0.0879 | no | −2.1 (0.064) |

## Run integrity

- 210 of 210 conversations clean: 4,264 model calls, 0 failed, 0 SDK retries, every call returned gpt-4.1-2025-04-14, code-fixed unchanged. 70 of 70 (repeat, scenario) blocks valid; none skipped by the budget guard.
- Scenarios come from the suite's own generator: all 24 suite scenarios reproduced byte-identically, and all 14 new ones are replay-certified on the 2,337-line engine (none dropped). They were selected by rule tag with a fixed seed.

## Spend (follow-up ledger, `spend-ledger.jsonl`, cap $45)

| item | USD |
|---|---|
| narration, first attempt (dead Anthropic key; charged at worst case, real cost $0) | 0.3822 |
| Bedrock credential probe (worst case, entered by hand) | 0.0002 |
| narration with gpt-4.1 | 0.0160 |
| **unplanned test-triggered run, unused** (errors.md 4: $16.701 settled plus 8 killed calls at worst case) | 19.2521 |
| registered run `main` | 17.9887 |
| **total** | **37.6392** |

The registered run by arm: control $5.909, R3 $6.048, R5 $6.032. The pre-run estimate was $21.08; the run cost $17.99.

## Reproduce

```
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/rule-ablation-followup/score_followup.py \
    experiments/rule-ablation-followup/results/main
```
