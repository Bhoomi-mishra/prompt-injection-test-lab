"""Harmless test payloads and the function that hides them in a document.

Every payload only asks the bot to print a harmless marker word (like BANANA).
Nothing here is harmful. The goal is only to measure whether the bot obeys
text that appears inside a document.
"""
import re

import pandas as pd

import config
from src import evaluate
from src.textutil import split_sentences

POSITIONS = ["start", "middle", "end"]


def load_payloads():
    return pd.read_csv(config.PAYLOADS_CSV, encoding="utf-8").to_dict("records")


def _insert(paragraph, payload_text, position):
    sentences = split_sentences(paragraph.strip())
    if position == "start":
        sentences.insert(0, payload_text)
    elif position == "middle":
        sentences.insert(max(1, len(sentences) // 2), payload_text)
    else:
        sentences.append(payload_text)
    return " ".join(sentences)


def poison_docs(docs, doc_name, expected, payload_text, position):
    """Return a copy of docs where the paragraph holding the answer also carries the payload."""
    new_docs = dict(docs)
    paragraphs = [p.strip() for p in new_docs[doc_name].split("\n\n") if p.strip()]
    target = next(
        (i for i, p in enumerate(paragraphs) if evaluate.is_correct(p, expected)),
        0 if len(paragraphs) < 2 else 1,
    )
    paragraphs[target] = _insert(paragraphs[target], payload_text, position)
    new_docs[doc_name] = "\n\n".join(paragraphs)
    return new_docs


def poison_kb(docs, questions, payload_text, shift=0):
    """Poison a WHOLE knowledge base with one hidden instruction.

    Every paragraph that holds an answer to one of the questions gets the payload once.
    Returns (new_docs, positions) where positions maps question_id -> position used.
    """
    original = {k: [x.strip() for x in v.split("\n\n") if x.strip()] for k, v in docs.items()}
    new_paras = {k: list(v) for k, v in original.items()}
    done, positions = {}, {}
    for qi, (_, q) in enumerate(questions.iterrows()):
        paras = original[q["doc"]]
        idx = next((i for i, x in enumerate(paras) if evaluate.is_correct(x, q["expected"])),
                   0 if len(paras) < 2 else 1)
        key = (q["doc"], idx)
        if key not in done:
            position = POSITIONS[(qi + shift) % len(POSITIONS)]
            new_paras[q["doc"]][idx] = _insert(new_paras[q["doc"]][idx], payload_text, position)
            done[key] = position
        positions[int(q["id"])] = done[key]
    return {k: "\n\n".join(v) for k, v in new_paras.items()}, positions
