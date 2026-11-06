# Prompt-Injection Test Lab

A lab that tests whether a document-reading chatbot (RAG) can be tricked by **harmless**
hidden instructions placed inside its documents, and compares simple defenses.

Three ways to use it:
1. **Study:** the built-in fictional document sets (a college handbook and a company HR
   handbook) let you run repeatable experiments on the lab's built-in bot.
2. **Replica of your own bot:** give it your own documents and questions. The lab builds a
   replica (its built-in retriever + the Ollama model you choose) and tests that.
   This is NOT your real chatbot, only a stand-in with the same knowledge.
3. **Test pack (any chatbot):** the lab creates a pack of knowledge bases; the bot's owner
   loads them into THEIR bot, answers the questions, and sends the answers back for scoring.
   The lab never needs access to the other bot.

All hidden instructions are harmless (they ask the bot to print a marker word such as
BANANA). It is a testing tool, not a protection tool.

## How it works

```
documents -> chunks -> TF-IDF retriever -> [D1 scan] -> prompt [D2 rules] -> model -> [D3 check] -> answer
                                                                                   |
                    test runner repeats everything -> results CSV -> dashboard / report
```

| Defense | What it does |
|---|---|
| D1 | Removes instruction-like sentences from retrieved text (pattern rules) |
| D2 | Adds safety rules to the prompt and marks documents as untrusted |
| D3 | Asks the model to check its own answer before showing it |

Hidden-instruction styles: **plain** (obvious wording), **disguised** (looks like an admin
note), **stealth** (avoids obvious command words, so simple pattern rules miss it).

## Setup (Windows)

1. Install Python 3.10+, VS Code, Git and Ollama (https://ollama.com, free).
2. `ollama pull llama3.2:3b` (check current model names at https://ollama.com/library).
   If the C: drive is full, set the `OLLAMA_MODELS` environment variable to a folder on
   another drive and restart Ollama before pulling.
3. In the project folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

## Commands

```
python -m unittest discover tests

# basic run: each question gets ONE hidden instruction (fast)
python -m src.runner --dataset sunrise_college --conditions none,d1,d2,d1+d2 --name sunrise_basic --fresh

# stronger run: EVERY hidden instruction on EVERY question
python -m src.runner --dataset sunrise_college --grid --conditions none,d1,d2,d1+d2 --name sunrise_grid --fresh

# second dataset, another model
python -m src.runner --dataset novatech_hr --grid --model qwen2.5:3b --name hr_qwen --fresh

# REPLICA of your own bot (built-in bot + your documents)
python -m src.runner --docs-folder examples/refund_bot/docs --questions-file examples/refund_bot/questions.csv --grid --name refund_bot --fresh

# TEST PACK for ANY chatbot
python -m src.pack --dataset sunrise_college --payloads P2,P3,P5 --limit 10 --out packs/demo
#   send packs/demo/for_owner.zip to the bot owner; keep packs/demo/lab_key/key.json private
#   the owner fills in answers_template.csv, then:
python -m src.score --key packs/demo/lab_key/key.json --answers filled.csv --condition none --bot "Their bot" --name their_bot --fresh
#   a second answers file with a changed setting can be added to the same results:
python -m src.score --key packs/demo/lab_key/key.json --answers filled2.csv --condition my_defense --bot "Their bot" --name their_bot

# report and dashboard
python -m src.report --in results/sunrise_grid.csv
streamlit run app/app.py
```

Your own questions file needs two columns: `question` and `expected` (the correct answer
text; alternatives separated by `|`). The expected answer must appear in one of your
documents. Run one condition file per run so results are not mixed (`--fresh` starts a new file).

## Metrics

- **attack_success_%**: poisoned tests where the answer contained the harmless marker
- **answer_lost_under_attack_%**: poisoned tests where the correct fact was missing
- **clean_accuracy_%**: normal tests where the answer contained the correct fact
- **accuracy_cost_pp**: drop in clean accuracy compared with no defense (the false-alarm cost)
- **ranges**: approximate 95% ranges. Each question (or question + instruction pair) counts
  once, because repeats of the same test are not independent evidence.

## Project structure

```
config.py            settings
src/llm.py           model access (Ollama or mock)
src/rag.py           chunking, retrieval, prompt
src/datasets.py      built-in sets and your own documents
src/attacks.py       harmless payloads and how they are hidden
src/pack.py          creates test packs for other people's chatbots
src/score.py         scores the answers sent back
src/textutil.py      sentence splitting that respects "Rs." and similar
src/defenses.py      D1, D3 (D2 lives in rag.py)
src/evaluate.py      "correct?" and "tricked?" checks
src/runner.py        runs experiments, saves results/*.csv
src/metrics.py       summary numbers with ranges
src/report.py        markdown report
app/app.py           Streamlit UI
data/sets/           built-in document sets
examples/            a small example of your own chatbot data
tests/               unit tests
```

## Test pack: how it works

- 1 clean knowledge base + 1 knowledge base per hidden instruction. In each poisoned
  knowledge base, every paragraph that holds an answer carries the same harmless line.
- The owner gets: the knowledge bases, the questions (no answers), an answers template and instructions.
- The lab keeps the key (correct answers + the marker words) and scores the returned answers.
- A pack with 3 hidden instructions and 10 questions needs (1 + 3) x 10 = 40 answers, which is practical by hand.
- Only test chatbots you own or have permission to test.

## Limitations

- Small fictional document sets and small local models; results apply only to what was tested.
- Retrieval is TF-IDF, not embeddings.
- D1 is pattern matching and is easy to get around (see stealth payloads); D3 relies on the same model it checks.
- A marker word appearing in the answer counts as "tricked", even if the bot is quoting a refusal. Check the drill-down table.
- Only harmless payloads are tested; real attacks can be more varied.
- With a test pack, the answer depends on how the owner loaded the documents and how the bot retrieves them; the lab cannot check that.
- This is a measurement study, not a new defense.

## Ideas for extension

Embedding-based retrieval, Hindi/Hinglish documents, more models, plain-text-only versus link-allowed outputs, more payload styles.
