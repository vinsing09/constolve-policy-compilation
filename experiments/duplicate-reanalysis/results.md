# Duplicate re-analysis (no model calls)

Script: `reanalyse.py` (writes `results.json`). Tests: `tests/test_reanalyse.py` (6 pass, real suite and sealed runs).

## Duplicates, verified
Functional key (kind, world, graded criteria) merges exactly 5 pairs of the 24: 0=15, 3=17, 6=9, 7=14, 8=11, so 19 clusters.
Within each pair, world_setup, symptoms, reference_solution, success_criteria, acceptance, direct_state, known_facts,
caller, wont_volunteer and will_reveal_if_asked are identical. What differs:
- `family` label (all 5 pairs), and with it the list of 4 device checks the simulated customer is told it can run
  (in `task_instructions`; all 5 pairs);
- the opening line (`reason_for_call`, `goal`) only in 3=17 and 6=9. In 0=15, 7=14 and 8=11 the opening line is identical.
So "byte-identical apart from the opening line" is not exact: the agent-side test is identical; the customer brief differs
in the family-derived check list in all 5 pairs and in the opening line in 2.

## 1. +30.0 separation (results/full-fixed-2, cluster30.py method, same seed)
| | 24 scenarios (reproduces cluster30.py) | 19 clusters |
|---|---|---|
| bootstrap 95% CI | [24.6, 35.3] | [24.7, 35.4] |
| sign-flip p, MC 20,000 | 0.00005 (printed 0.0001) | 0.00015 |
| sign-flip p, exact | 3.7e-6 | 7.2e-5 |
| better / worse / tied | 21 / 2 / 1 | 16 / 2 / 1 |

## 2. Main rule-removal run (score_ablation.py method; identity clustering reproduces score.json exactly)
| rule | scen. -> clusters | drop | p 24 -> 19 | min possible p 24 -> 19 | caught | untied p 24 -> 19 |
|---|---|---|---|---|---|---|
| R1 | 11 -> 8 | -7.1 | 0.797 -> 0.719 | 2^-11 -> 2^-8 | no | 0.564 -> 0.557 |
| R2 | 10 -> 7 (7 of 7 down) | +34.6 | 0.0020 -> 0.0078 | 2^-10 -> 2^-7 | yes | 0.315 -> 0.249 |
| R3 | 3 -> 3 | +25.0 | 0.25 -> 0.25 | 0.125 unchanged | no | 0.0081 -> 0.0080 |
| R4 | 9 -> 7 | +8.3 | 0.25 -> 0.25 | 2^-9 -> 2^-7 | no | 0.239 -> 0.230 |
| R5 | 5 -> 4 (3 down, 1 same) | +26.9 | 0.0625 -> 0.125 | 1/32 -> 1/16 | no | 0.413 -> 0.358 |
| R6 | 24 -> 19 | +15.9 | 0.0367 -> 0.0581 | 2^-24 -> 2^-19 | no | 0.152 -> 0.234 |

Pass rates and drops unchanged (pooling keeps every graded check). Caught: still 1 of 6 (R2), 0 false alarms.
R5 can no longer be caught in the main run at any effect size (4 clusters, min p 1/16). R6 is no longer p < 0.05.
Overlap pair: R5 own-only +30.8 p 0.125 -> 0.25, partner-only +12.0 p 0.137 -> 0.125; R6 own-only +22.0 p 0.0146 -> 0.0244,
partner-only +30.8 p 0.125 -> 0.25. R6 still lands on the right rule; R5 still does not.

## 3. Follow-up and pooled
- Overlap: 14 follow-up scenarios, 14 distinct functional keys, 0 shared with the 24.
- Stricter check (criterion ids and order ignored): follow-up index 2 (walk 2929) has the same world and the same graded
  checks as suite scenario 16; it differs in direct_state (the customer is told the bill is overdue and the line suspended),
  reference step order, and the brief. Both are unchanged under R3 (10/10 -> 10/10 and unchanged in the main run), so merging
  them changes nothing but the count: R3 pooled 8 instead of 9 clusters, +21.1, p 0.0156.
- Follow-up primary (unchanged): R3 96.7 -> 76.7, +20.0, p 0.0625; R5 42.5 -> 37.5, +5.0, p 0.4219.
- Pooled secondary (already clustered by score_followup.py; recomputed, unchanged): R3 +21.1, p 0.0156 (meets criterion);
  R5 +14.8, p 0.0879.
