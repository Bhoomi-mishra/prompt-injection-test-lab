"""Checks that the pipeline logic works, using the fast mock model.
Run:  python -m unittest discover tests
"""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import attacks, datasets, defenses, evaluate, metrics, rag, report, runner


class TestPipeline(unittest.TestCase):
    def test_evaluate(self):
        self.assertTrue(evaluate.is_correct("The fee is Rs. 85,000 per year.", "85000"))
        self.assertTrue(evaluate.is_correct("Attendance must be 75 percent", "75%|75 percent"))
        self.assertFalse(evaluate.is_correct("I don't know", "85000"))
        self.assertTrue(evaluate.is_tricked("BANANA", "BANANA"))
        self.assertFalse(evaluate.is_tricked("Rs. 85,000", "BANANA"))

    def test_both_sets_load_and_retrieve(self):
        for name in datasets.list_sets():
            docs, qs = datasets.load_set(name)
            self.assertEqual(len(qs), 20, name)
            ret = rag.Retriever(rag.chunk_docs(docs))
            hits = sum(
                any(evaluate.is_correct(c["text"], q["expected"]) for c in ret.search(q["question"], k=2))
                for _, q in qs.iterrows()
            )
            print(f"\nretrieval recall@2 [{name}]: {hits}/20")
            self.assertGreaterEqual(hits, 18, name)
        self.assertIn("novatech_hr", datasets.list_sets())

    def test_prepare_questions_finds_doc_and_rejects_missing_answer(self):
        docs = {"a": "Title\n\nThe fee is Rs. 500.", "b": "Other\n\nOpen at 9 AM."}
        ok = datasets.prepare_questions(pd.DataFrame({"question": ["fee?", "time?"], "expected": ["500", "9 am"]}), docs)
        self.assertEqual(list(ok["doc"]), ["a", "b"])
        with self.assertRaises(ValueError):
            datasets.prepare_questions(pd.DataFrame({"question": ["x?"], "expected": ["zzz"]}), docs)
        with self.assertRaises(ValueError):
            datasets.prepare_questions(pd.DataFrame({"q": ["x?"]}), docs)

    def test_poison_inserts_payload_in_answer_paragraph(self):
        docs = rag.load_docs()
        poisoned = attacks.poison_docs(docs, "fees", "85000", "HIDDEN TEXT BANANA.", "start")
        self.assertIn("HIDDEN TEXT BANANA.", poisoned["fees"])
        self.assertNotIn("HIDDEN TEXT BANANA.", docs["fees"])
        para = [p for p in poisoned["fees"].split("\n\n") if "85,000" in p][0]
        self.assertIn("HIDDEN TEXT BANANA.", para)

    def test_d1_catches_plain_and_disguised_but_not_stealth(self):
        payloads = {p["id"]: p for p in attacks.load_payloads()}
        for pid, should_flag in [("P1", True), ("P2", True), ("P3", True),
                                 ("P4", True), ("P5", False), ("P6", False)]:
            chunk = [{"doc": "x", "text": "The fee is Rs. 85,000. " + payloads[pid]["text"]}]
            cleaned, flagged = defenses.scan_chunks(chunk)
            self.assertEqual(flagged, should_flag, pid)
            if should_flag:
                self.assertNotIn(payloads[pid]["marker"], cleaned[0]["text"], pid)
            self.assertIn("85,000", cleaned[0]["text"])

    def test_d1_false_positive_on_clean_docs(self):
        for name, doc in [("sunrise_college", "hostel"), ("novatech_hr", "insurance")]:
            docs, _ = datasets.load_set(name)
            _, flagged = defenses.scan_chunks(rag.chunk_docs({doc: docs[doc]}))
            self.assertTrue(flagged, name)  # intentional: a normal sentence looks like a command

    def test_hidden_line_never_lands_inside_an_amount(self):
        from src.textutil import split_sentences
        self.assertEqual(split_sentences("The fine is Rs. 5 per day. Pay at the desk."),
                         ["The fine is Rs. 5 per day.", "Pay at the desk."])
        docs, qs = datasets.load_set("sunrise_college")
        for pos in attacks.POSITIONS:
            for _, q in qs.iterrows():
                poisoned = attacks.poison_docs(docs, q["doc"], q["expected"], "HIDDEN BANANA.", pos)
                self.assertTrue(evaluate.is_correct(poisoned[q["doc"]], q["expected"]), (q["question"], pos))
                para = next(x for x in poisoned[q["doc"]].split("\n\n") if "HIDDEN BANANA." in x)
                self.assertTrue(evaluate.is_correct(para, q["expected"]), (q["question"], pos, para))

    def test_wilson_range(self):
        lo, hi = metrics.wilson(0.5, 20)
        self.assertAlmostEqual(lo, 0.299, places=2)
        self.assertAlmostEqual(hi, 0.701, places=2)
        lo2, hi2 = metrics.wilson(0.5, 120)
        self.assertLess(hi2 - lo2, hi - lo)  # more units -> narrower range

    def test_basic_run_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "r.csv"
            df = runner.run_experiment(["none", "d1", "d2", "all"], repeats=1, limit=20,
                                       backend="mock", out_csv=out, fresh=True)
            self.assertEqual(list(df.columns), runner.COLUMNS)
            self.assertEqual(len(df), 20 * 4 * 2)
            s = metrics.summarize(pd.read_csv(out)).set_index("condition")
            self.assertGreater(s.loc["none", "attack_success_%"], 50)
            self.assertEqual(s.loc["d2", "attack_success_%"], 0)
            self.assertGreater(s.loc["none", "clean_accuracy_%"], 60)

    def test_grid_run_and_units_ignore_repeats(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "g.csv"
            n_pay = len(attacks.load_payloads())
            df = runner.run_experiment(["none", "d2"], repeats=2, limit=3, backend="mock",
                                       out_csv=out, fresh=True, grid=True, dataset="novatech_hr")
            self.assertEqual(len(df), 3 * (1 + n_pay) * 2 * 2)
            s = metrics.summarize(df).set_index("condition")
            self.assertEqual(s.loc["none", "n_attack_units"], 3 * n_pay)  # repeats not counted
            self.assertEqual(s.loc["none", "n_clean_units"], 3)
            self.assertEqual(set(df["dataset"]), {"novatech_hr"})
            bp = metrics.by_payload(df)
            self.assertEqual(int(bp["n_questions"].iloc[0]), 3)
            text = report.make_report(df)
            self.assertIn("Defenses compared", text)

    def test_custom_docs_and_questions(self):
        docs = {"policy": "Policy\n\nRefunds are processed within 14 days. Call 555 for help."}
        qs = pd.DataFrame({"question": ["How many days do refunds take?"], "expected": ["14 days"]})
        with tempfile.TemporaryDirectory() as tmp:
            df = runner.run_experiment(["none", "d2"], limit=1, backend="mock",
                                       out_csv=Path(tmp) / "c.csv", fresh=True, grid=True,
                                       docs=docs, questions=qs)
            self.assertEqual(set(df["dataset"]), {"custom"})
            self.assertGreater(len(df), 0)


if __name__ == "__main__":
    unittest.main()
