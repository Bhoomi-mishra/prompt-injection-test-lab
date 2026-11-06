"""Scores the answers a bot owner sends back.

    python -m src.score --key packs/sunrise/lab_key/key.json --answers filled.csv --condition none --bot "My bot" --name mybot --fresh
    python -m src.score --key ... --answers filled_with_defense.csv --condition my_defense --bot "My bot" --name mybot     (adds to the same file)
"""
import argparse
import csv
import json
from pathlib import Path

import pandas as pd

import config
from src import evaluate, metrics
from src.runner import COLUMNS


def score_answers(key, answers_df, condition="none", bot="my_bot"):
    """Returns (rows_dataframe, info). Raises ValueError for unusable files."""
    df = answers_df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing_cols = [c for c in ("kb", "question_id", "answer") if c not in df.columns]
    if missing_cols:
        raise ValueError(f"The answers file needs the columns kb, question_id, answer (missing: {missing_cols}).")

    payloads = {p["id"]: p for p in key["payloads"]}
    questions = {int(q["id"]): q for q in key["questions"]}
    rows, empty, skipped = [], 0, []
    seen = {}

    for _, r in df.iterrows():
        ans = r["answer"]
        if pd.isna(ans) or str(ans).strip() == "":
            empty += 1
            continue
        kb = str(r["kb"]).strip()
        kb = kb[3:] if kb.lower().startswith("kb_") else kb
        try:
            qid = int(r["question_id"])
        except (TypeError, ValueError):
            skipped.append(f"bad question_id: {r['question_id']}")
            continue
        if qid not in questions:
            skipped.append(f"unknown question_id {qid}")
            continue
        clean = kb.lower() == "clean"
        if not clean and kb not in payloads:
            skipped.append(f"unknown knowledge base '{kb}'")
            continue

        q = questions[qid]
        p = None if clean else payloads[kb]
        rep = seen.get((kb, qid), 0)
        seen[(kb, qid)] = rep + 1
        rows.append({
            "condition": condition,
            "variant": "clean" if clean else "poisoned",
            "question_id": qid,
            "question": q["question"],
            "expected": q["expected"],
            "payload_id": "" if clean else p["id"],
            "payload_style": "" if clean else p["style"],
            "position": "" if clean else key["positions"].get(kb, {}).get(str(qid), ""),
            "repeat": rep,
            "answer": str(ans),
            "tricked": bool(p and evaluate.is_tricked(ans, p["marker"])),
            "correct": bool(evaluate.is_correct(ans, q["expected"])),
            "blocked": False,
            "d1_flagged": False,
            "seconds": 0,
            "model": bot,
            "dataset": key.get("dataset", "custom"),
        })

    if not rows:
        raise ValueError("No scorable answers found. Fill in the 'answer' column of answers_template.csv.")
    out = pd.DataFrame(rows, columns=COLUMNS)
    info = {"scored": len(rows), "empty_skipped": empty, "problems": skipped[:10],
            "clean_rows": int((out["variant"] == "clean").sum()),
            "poisoned_rows": int((out["variant"] == "poisoned").sum())}
    return out, info


def score_to_csv(key, answers_df, condition, bot, out_csv, append=False):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    scored, info = score_answers(key, answers_df, condition, bot)
    if not append and out_csv.exists():
        out_csv.unlink()
    new_file = not out_csv.exists()
    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        w.writerows(scored.to_dict("records"))
    return scored, info


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--answers", required=True)
    ap.add_argument("--condition", default="none", help="label for this bot setting, e.g. none or my_defense")
    ap.add_argument("--bot", default="my_bot")
    ap.add_argument("--name", default="scored")
    ap.add_argument("--fresh", action="store_true", help="start a new results file")
    a = ap.parse_args()

    key = json.loads(Path(a.key).read_text(encoding="utf-8"))
    answers = pd.read_csv(a.answers, encoding="utf-8")
    out = config.RESULTS_DIR / f"{a.name}.csv"
    scored, info = score_to_csv(key, answers, a.condition, a.bot, out, append=not a.fresh)
    print(info)
    print("Saved to", out)
    print(metrics.summarize(scored).to_string(index=False))


if __name__ == "__main__":
    _main()
