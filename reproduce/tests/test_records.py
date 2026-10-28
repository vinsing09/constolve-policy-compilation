"""The record checks must fail when a released record is altered (real records, no model calls).

    python -B reproduce/tests/test_records.py
"""
import csv
import json
import lzma
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import repair_coverage as C  # noqa: E402
import determinism as DT  # noqa: E402
import instability as I  # noqa: E402


class Mutated(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _copy(self, mod, attr):
        dst = self.tmp / attr
        shutil.copytree(getattr(mod, attr), dst)
        self.addCleanup(setattr, mod, attr, getattr(mod, attr))
        setattr(mod, attr, dst)
        return dst

    def test_unaltered_records_pass(self):
        self.assertEqual(C.check(), [])
        self.assertEqual(DT.check(), [])
        self.assertEqual(I.check(), [])

    def test_one_flipped_verdict_fails(self):
        d = self._copy(I, "REC")
        rows = list(csv.DictReader((d / "verdicts.csv").open()))
        r = next(x for x in rows if x["domain"] == "retail")
        r["passed"] = str(1 - int(r["passed"]))
        with (d / "verdicts.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        self.assertTrue(I.check())

    def test_recorded_error_fails(self):
        d = self._copy(I, "REC")
        text = (d / "verdicts.csv").read_text().splitlines()
        text[1] = text[1].rsplit(",", 2)[0] + ",1," + text[1].rsplit(",", 1)[1]
        (d / "verdicts.csv").write_text("\n".join(text) + "\n")
        with self.assertRaises(SystemExit):
            I.check()

    def test_dropping_one_covering_set_fails(self):
        d = self._copy(C, "REC")
        p = d / "emitted_repair_sets.json.xz"
        with lzma.open(p, "rt") as f:
            rec = json.load(f)
        covered = C.compute()["_lists"]["covered"]
        tasks = {t["id"]: t for t in json.loads((C.TAU / "tasks.json").read_text())}
        gt = C.fault_repairs()
        sig = set()
        for fault in filter(None, C._fault(covered[0]).split("|")):
            sig |= gt[fault]
        sig -= {C.TRANSFER}
        assert covered[0] in tasks
        rec["sets"] = [e for e in rec["sets"] if set(e[1]) != sig]
        with lzma.open(p, "wt") as f:
            json.dump(rec, f)
        bad = C.check()
        self.assertTrue(any(b.startswith("covered:") for b in bad), bad)

    def test_divergent_replay_fails(self):
        d = self._copy(DT, "REC")
        p = d / "phase8_determinism_airline.json"
        r = json.loads(p.read_text())
        r["cross_process"]["state"] = 6
        p.write_text(json.dumps(r))
        self.assertTrue(DT.check())


if __name__ == "__main__":
    unittest.main(verbosity=1)
