"""Loading document sets and question files (built-in or your own)."""
from pathlib import Path

import pandas as pd

import config
from src import evaluate


def list_sets():
    if not config.SETS_DIR.exists():
        return []
    return sorted(
        p.name for p in config.SETS_DIR.iterdir()
        if (p / "docs").is_dir() and (p / "questions.csv").exists()
    )


def read_docs(folder):
    folder = Path(folder)
    files = sorted(list(folder.glob("*.txt")) + list(folder.glob("*.md")))
    return {p.stem: p.read_text(encoding="utf-8").strip() for p in files}


def _doc_has_answer(text, expected):
    return any(evaluate.is_correct(p, expected) for p in text.split("\n\n"))


def find_doc_for_answer(docs, expected):
    for name, text in docs.items():
        if _doc_has_answer(text, expected):
            return name
    return None


def prepare_questions(df, docs):
    """Check a questions table and work out which document holds each answer.

    Required columns: question, expected   (expected may list alternatives with |)
    Optional columns: id, doc
    """
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    missing = [c for c in ("question", "expected") if c not in df.columns]
    if missing:
        raise ValueError(f"The questions file needs the columns 'question' and 'expected' (missing: {missing}).")
    if not docs:
        raise ValueError("No documents found. Add .txt or .md files.")

    df["expected"] = df["expected"].astype(str)
    if "id" not in df.columns:
        df.insert(0, "id", range(1, len(df) + 1))
    if "doc" not in df.columns:
        df["doc"] = None

    chosen, problems = [], []
    for _, r in df.iterrows():
        doc = r["doc"] if isinstance(r["doc"], str) else None
        if doc not in docs or not _doc_has_answer(docs[doc], r["expected"]):
            doc = find_doc_for_answer(docs, r["expected"])
        if doc is None:
            problems.append(f"question {r['id']}: the expected answer '{r['expected']}' is not in any document")
        chosen.append(doc)
    if problems:
        raise ValueError("Every expected answer must appear in one of your documents.\n" + "\n".join(problems))

    df["doc"] = chosen
    return df[["id", "question", "expected", "doc"]].reset_index(drop=True)


def load_set(name_or_path):
    base = Path(name_or_path)
    if not base.is_dir():
        base = config.SETS_DIR / str(name_or_path)
    docs = read_docs(base / "docs")
    questions = prepare_questions(pd.read_csv(base / "questions.csv", encoding="utf-8"), docs)
    return docs, questions
