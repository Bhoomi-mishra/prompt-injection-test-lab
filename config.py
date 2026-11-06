"""All settings in one place."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SETS_DIR = ROOT / "data" / "sets"          # each set = docs/ folder + questions.csv
DEFAULT_SET = "sunrise_college"
PAYLOADS_CSV = ROOT / "data" / "payloads.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "results.csv"

# "ollama" = real local model, "mock" = fast fake model used for testing the code
BACKEND = os.getenv("LAB_BACKEND", "ollama")
MODEL = os.getenv("LAB_MODEL", "llama3.2:3b")
OLLAMA_URL = os.getenv("LAB_OLLAMA_URL", "http://localhost:11434/api/chat")
TEMPERATURE = 0.3
TIMEOUT_SECONDS = 300

# How many chunks the retriever passes to the model
TOP_K = 2
