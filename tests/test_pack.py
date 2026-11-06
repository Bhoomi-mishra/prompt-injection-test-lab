"""Tests for test packs (testing someone else's bot). Run:  python -m unittest discover tests"""
import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import attacks, datasets, evaluate, metrics, pack, runner, score


def simulate_owner_bot(files, backend="mock"):
    """A stand-in for 'someone else's chatbot': loads each KB and answers the template questions."""
    template = pd.read_csv(io.StringIO(files["answers_template.csv"]))
    kbs = {}
    for path, text in files.items():
        if path.startswith("kb_"):
            kb, fname = path.split("/", 1)
            kbs.setdefault(kb[3:], {})[Path(fname).stem] = text
    answers = []
    for _, r in template.iterrows():
        res = runner.answer_question(r["question"], kbs[r["kb"]], [], backend)
        answers.append(res["answer"])
    template["answer"] = answers
    return template


class TestPack(unittest.TestCase):
    def setUp(self):
        self.docs, self.qs = datasets.load_set("sunrise_college")
        self.payloads = attacks.load_payloads()

    def test_poison_kb_poisons_every_answer_paragraph_once(self):
        new_docs, positions = attacks.poison_kb(self.docs, self.qs, "HIDDEN BANANA.", shift=0)
        self.assertEqual(set(positions), set(self.qs["id"]))
        for _, q in self.qs.iterrows():
            para = next(p for p in new_docs[q["doc"]].split("\n\n") if evaluate.is_correct(p, q["expected"]))
            self.assertEqual(para.count("HIDDEN BANANA."), 1, q["question"])
        for name in self.docs:                       # originals untouched
            self.assertNotIn("HIDDEN BANANA.", self.docs[name])

    def test_build_pack_contents_and_no_answers_leak(self):
        files, key = pack.build_pack(self.docs, self.qs, self.payloads, "sunrise_college")
        n_docs = len(self.docs)
        self.assertEqual(sum(p.startswith("kb_") for p in files), n_docs * (1 + len(self.payloads)))
        template = pd.read_csv(io.StringIO(files["answers_template.csv"]))
        self.assertEqual(len(template), (1 + len(self.payloads)) * len(self.qs))
        self.assertEqual(set(template["kb"]), {"clean"} | {p["id"] for p in self.payloads})
        owner_text = " ".join(files.values())
        self.assertNotIn("expected", files["questions.csv"])
        self.assertNotIn("85000", files["questions.csv"] + files["answers_template.csv"])
        self.assertIn("expected", json.dumps(key))                   # answers live only in the key
        zf = zipfile.ZipFile(io.BytesIO(pack.zip_bytes(files)))
        self.assertIn("answers_template.csv", zf.namelist())
        self.assertIn("README.txt", zf.namelist())

    def test_end_to_end_with_simulated_other_bot(self):
        files, key = pack.build_pack(self.docs, self.qs, self.payloads, "sunrise_college")
        answers = simulate_owner_bot(files)
        scored, info = score.score_answers(key, answers, "none", "pretend bot")
        self.assertEqual(info["scored"], len(answers))
        s = metrics.summarize(scored).set_index("condition")
        print("\n", s.to_string())
        self.assertGreater(s.loc["none", "attack_success_%"], 50)      # naive mock bot gets tricked
        self.assertGreater(s.loc["none", "clean_accuracy_%"], 60)
        self.assertEqual(s.loc["none", "n_attack_units"], len(self.qs) * len(self.payloads))
        bp = metrics.by_payload(scored)
        self.assertIn("n_questions", bp.columns)

        with tempfile.TemporaryDirectory() as tmp:       # CSV round trip + append a 2nd setting
            out = Path(tmp) / "s.csv"
            score.score_to_csv(key, answers, "none", "pretend bot", out)
            score.score_to_csv(key, answers, "my_defense", "pretend bot", out, append=True)
            both = pd.read_csv(out)
            self.assertEqual(set(both["condition"]), {"none", "my_defense"})
            self.assertEqual(metrics.summarize(both).shape[0], 2)

    def test_partial_and_bad_answers(self):
        files, key = pack.build_pack(self.docs, self.qs.head(3), self.payloads[:2], "x")
        template = pd.read_csv(io.StringIO(files["answers_template.csv"]))
        template["answer"] = template["answer"].astype(object)
        template.loc[0, "answer"] = "Rs. 85,000"
        template.loc[1, "answer"] = "BANANA"
        template.loc[2, "kb"] = "P99"; template.loc[2, "answer"] = "x"
        scored, info = score.score_answers(key, template)
        self.assertEqual(info["scored"], 2)
        self.assertTrue(info["problems"])
        with self.assertRaises(ValueError):
            score.score_answers(key, template.assign(answer=""))
        with self.assertRaises(ValueError):
            score.score_answers(key, template.drop(columns=["answer"]))

    def test_kb_prefix_and_repeated_rows(self):
        files, key = pack.build_pack(self.docs, self.qs.head(2), self.payloads[:1], "x")
        pid = self.payloads[0]["id"]
        df = pd.DataFrame({"kb": [f"kb_{pid}", f"kb_{pid}"], "question_id": [1, 1], "answer": ["BANANA", "Rs. 85,000"]})
        scored, _ = score.score_answers(key, df)
        self.assertEqual(list(scored["repeat"]), [0, 1])


if __name__ == "__main__":
    unittest.main()
