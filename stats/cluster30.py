import json, random, sys
from pathlib import Path
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import score as sc

run = R / "results/full-fixed-2"
m = json.loads((run / "manifest.json").read_text())
rows = sc.load(run / "traces", m["case_ids"], m["repeats"])
assert len(rows) == 240
print("pooled gap (must be 30.0):", round(sc.compare(rows)["gap"]["action"], 2))

def per_case(arm):
    out = {}
    for d in rows:
        if d["arm"] != arm:
            continue
        a = out.setdefault(d["case"], [0, 0])
        for ok, must in zip(d["ok"], d["must"]):
            if must:
                a[0] += ok; a[1] += 1
    return out

w, s = per_case("with_policy"), per_case("stub")
cases = sorted(w)
def gap(cs):
    return 100 * (sum(w[c][0] for c in cs) / sum(w[c][1] for c in cs)
                  - sum(s[c][0] for c in cs) / sum(s[c][1] for c in cs))
rng = random.Random(20261027)
boot = sorted(gap([rng.choice(cases) for _ in cases]) for _ in range(20000))
print("scenarios:", len(cases), "graded actions/arm:", sum(w[c][1] for c in cases))
print("scenario-cluster bootstrap 95%% CI: [%.1f, %.1f]" % (boot[499], boot[19499]))
d = {c: w[c][0] / w[c][1] - s[c][0] / s[c][1] for c in cases}
pos = sum(v > 0 for v in d.values()); neg = sum(v < 0 for v in d.values())
print("scenarios policy better / worse / tied:", pos, neg, len(cases) - pos - neg)
obs = sum(d.values()); hits = 0
for _ in range(20000):
    if sum(v * rng.choice((1, -1)) for v in d.values()) >= obs - 1e-12:
        hits += 1
print("paired sign-flip permutation p (one-sided, scenario unit): %.4f" % (hits / 20000))
import csv
with open(sys.argv[1], "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["scenario", "policy_done", "policy_required", "stub_done", "stub_required", "diff_pp"])
    for c in cases:
        wr.writerow([c, w[c][0], w[c][1], s[c][0], s[c][1], round(100 * d[c], 2)])
