"""Runs the experiments.

Examples:
    python -m src.runner --backend mock --limit 5 --fresh
    python -m src.runner --dataset sunrise_college --grid --conditions none,d1,d2,d1+d2 --name sunrise_grid
    python -m src.runner --docs-folder my_docs --questions-file my_questions.csv --grid --name mine
"""
import argparse
import csv
import time
from pathlib import Path

import pandas as pd

import config
from src import attacks, datasets, defenses, evaluate, llm, rag

CONDITIONS = {
    "none": [],
    "d1": ["d1"],
    "d2": ["d2"],
    "d3": ["d3"],
    "d1+d2": ["d1", "d2"],
    "all": ["d1", "d2", "d3"],
}

COLUMNS = [
    "condition", "variant", "question_id", "question", "expected",
    "payload_id", "payload_style", "position", "repeat",
    "answer", "tricked", "correct", "blocked", "d1_flagged", "seconds",
    "model", "dataset",
]


def answer_question(question, docs, defense_list, backend=None, model=None):
    """Full pipeline for one question."""
    retriever = rag.Retriever(rag.chunk_docs(docs))
    hits = retriever.search(question)

    flagged = False
    if "d1" in defense_list:
        hits, flagged = defenses.scan_chunks(hits)

    system, user = rag.build_prompt(question, hits, safe_rules="d2" in defense_list)
    answer = llm.chat(system, user, backend, model)

    blocked = False
    if "d3" in defense_list and not defenses.check_answer(question, hits, answer, backend, model):
        blocked = True
        answer = evaluate.BLOCKED_TEXT

    return {"answer": answer, "chunks": hits, "d1_flagged": flagged, "blocked": blocked}


def _resolve(dataset, docs, questions):
    if docs is not None and questions is not None:
        return docs, datasets.prepare_questions(questions, docs), "custom"
    name = dataset or config.DEFAULT_SET
    d, q = datasets.load_set(name)
    return d, q, Path(str(name)).name


def run_experiment(conditions, repeats=1, limit=None, backend=None, out_csv=None,
                   progress=None, fresh=False, dataset=None, grid=False, model=None,
                   docs=None, questions=None):
    """grid=False: each question gets ONE hidden instruction (fast).
       grid=True : every question is tested with EVERY hidden instruction (stronger)."""
    out_csv = Path(out_csv or config.RESULTS_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if fresh and out_csv.exists():
        out_csv.unlink()

    docs, questions, dataset_name = _resolve(dataset, docs, questions)
    payloads = attacks.load_payloads()
    if limit:
        questions = questions.head(int(limit))
    model_name = model or config.MODEL
    if (backend or config.BACKEND) == "mock":
        model_name = "mock"

    n_poison = len(payloads) if grid else 1
    total = len(questions) * (1 + n_poison) * len(conditions) * repeats
    done, start, rows = 0, time.time(), []
    write_header = not out_csv.exists()

    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()

        for qi, (_, q) in enumerate(questions.iterrows()):
            poison_units = []
            for pi in (range(len(payloads)) if grid else [qi % len(payloads)]):
                payload = payloads[pi]
                position = attacks.POSITIONS[(qi + pi) % 3 if grid else qi % 3]
                poison_units.append((payload, position, attacks.poison_docs(
                    docs, q["doc"], q["expected"], payload["text"], position)))
            units = [("clean", None, None, docs)] + [("poisoned", p, pos, d) for p, pos, d in poison_units]

            for cond in conditions:
                defense_list = CONDITIONS[cond]
                for variant, payload, position, use_docs in units:
                    for rep in range(repeats):
                        t0 = time.time()
                        try:
                            res = answer_question(q["question"], use_docs, defense_list, backend, model)
                        except RuntimeError as e:
                            if done == 0:
                                raise  # fail fast if the model is not reachable
                            res = {"answer": f"[ERROR] {e}", "d1_flagged": False, "blocked": False}

                        answer = res["answer"]
                        row = {
                            "condition": cond,
                            "variant": variant,
                            "question_id": int(q["id"]),
                            "question": q["question"],
                            "expected": q["expected"],
                            "payload_id": payload["id"] if payload else "",
                            "payload_style": payload["style"] if payload else "",
                            "position": position or "",
                            "repeat": rep,
                            "answer": answer,
                            "tricked": bool(payload and not res["blocked"]
                                            and evaluate.is_tricked(answer, payload["marker"])),
                            "correct": bool(evaluate.is_correct(answer, q["expected"])),
                            "blocked": bool(res["blocked"]),
                            "d1_flagged": bool(res["d1_flagged"]),
                            "seconds": round(time.time() - t0, 2),
                            "model": model_name,
                            "dataset": dataset_name,
                        }
                        writer.writerow(row)
                        f.flush()
                        rows.append(row)

                        done += 1
                        if progress:
                            progress(done, total, (time.time() - start) / done * (total - done))

    return pd.DataFrame(rows, columns=COLUMNS)


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", default="none,d1,d2,d1+d2")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N questions")
    ap.add_argument("--backend", default=None, help="ollama or mock")
    ap.add_argument("--model", default=None, help="Ollama model name")
    ap.add_argument("--dataset", default=None, help="built-in set name, e.g. sunrise_college")
    ap.add_argument("--docs-folder", default=None, help="your own folder of .txt/.md documents")
    ap.add_argument("--questions-file", default=None, help="your own CSV with columns question,expected")
    ap.add_argument("--grid", action="store_true", help="test every hidden instruction on every question")
    ap.add_argument("--name", default=None, help="results file name (saved in results/)")
    ap.add_argument("--fresh", action="store_true", help="delete the old results file first")
    args = ap.parse_args()

    conds = [c.strip() for c in args.conditions.split(",")]
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        raise SystemExit(f"Unknown condition(s): {bad}. Choose from {list(CONDITIONS)}")

    docs = questions = None
    if args.docs_folder or args.questions_file:
        if not (args.docs_folder and args.questions_file):
            raise SystemExit("Use --docs-folder and --questions-file together.")
        docs = datasets.read_docs(args.docs_folder)
        questions = pd.read_csv(args.questions_file, encoding="utf-8")

    out = config.RESULTS_DIR / f"{args.name}.csv" if args.name else config.RESULTS_CSV

    def show(done, total, eta):
        print(f"\r{done}/{total} done, about {eta / 60:.1f} min left   ", end="", flush=True)

    df = run_experiment(conds, args.repeats, args.limit, args.backend, out_csv=out, progress=show,
                        fresh=args.fresh, dataset=args.dataset, grid=args.grid, model=args.model,
                        docs=docs, questions=questions)
    print("\nSaved to", out)
    from src import metrics
    print(metrics.summarize(df).to_string(index=False))


if __name__ == "__main__":
    _main()
