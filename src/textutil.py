"""Small text helpers shared by several modules."""
import re

# abbreviations that end with a full stop but do NOT end a sentence
_ABBR = re.compile(r"\b(Rs|No|Mr|Mrs|Ms|Dr|vs|etc|e\.g|i\.e)\.\s", re.IGNORECASE)


def split_sentences(text, closers=".!?"):
    """Split text into sentences without cutting after abbreviations like 'Rs.'"""
    protected = _ABBR.sub(lambda m: m.group(0).replace(" ", "\x00"), text)
    parts = re.split(rf"(?<=[{re.escape(closers)}])\s+", protected)
    return [p.replace("\x00", " ") for p in parts]
