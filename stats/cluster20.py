import json, lzma, random, sys
from collections import defaultdict
from pathlib import Path
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "scripts"))
import score_deep as sd

ids = sd.subsets(sd.rd.PROJECT / "deep_policy")["unseen_51"]
def per_task(run):
    sims = json.loads(lzma.open(R / "results" / run / "tau3" / "results.json.xz").read())["simulations"]
    c = defaultdict(list)
    for x in sims:
        if x["task_id"] in ids:
            c[x["task_id"]].append(sd.success(x["reward_info"]["reward"]))
    return c
a, b = per_task("deep-v0"), per_task("deep-v2_deep-2")
assert set(a) == set(b) == ids and all(len(a[t]) == len(b[t]) == 5 for t in ids)
rate = lambda c, ts: 100 * sum(sum(c[t]) for t in ts) / (5 * len(ts))
ts = sorted(ids)
print("unseen tasks:", len(ts), "v0 %.1f patch %.1f gap %.1f" % (rate(a, ts), rate(b, ts), rate(b, ts) - rate(a, ts)))
rng = random.Random(20261027)
boot = sorted(rate(b, s) - rate(a, s) for s in ([rng.choice(ts) for _ in ts] for _ in range(20000)))
print("task-cluster bootstrap 95%% CI: [%.1f, %.1f]" % (boot[499], boot[19499]))
d = {t: (sum(b[t]) - sum(a[t])) / 5 for t in ts}
print("tasks better / worse / tied:", sum(v > 0 for v in d.values()), sum(v < 0 for v in d.values()), sum(v == 0 for v in d.values()))
obs = sum(d.values())
p = sum(sum(v * rng.choice((1, -1)) for v in d.values()) >= obs - 1e-12 for _ in range(20000)) / 20000
print("paired sign-flip permutation p (one-sided, task unit): %.4f" % p)
import csv
with open(sys.argv[1], "w", newline="") as f:
    wr = csv.writer(f); wr.writerow(["task_id", "original_passes_of_5", "patched_passes_of_5"])
    for t in ts:
        wr.writerow([t, sum(a[t]), sum(b[t])])
