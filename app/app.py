"""Prompt-Injection Test Lab (Streamlit UI).
Run from the project folder:   streamlit run app/app.py
"""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

import config
from src import attacks, datasets, evaluate, metrics, pack, report, runner, score

st.set_page_config(page_title="Prompt-Injection Test Lab", layout="wide")
payloads = attacks.load_payloads()

# ------------------------------------------------------------------ sidebar
st.sidebar.title("Prompt-Injection Test Lab")
page = st.sidebar.radio("Page", ["Chat demo", "Document lab", "Run tests", "Test pack", "Results"])
backend = st.sidebar.selectbox(
    "Model backend", ["ollama", "mock"], index=0 if config.BACKEND == "ollama" else 1,
    help="ollama = real local model. mock = fast fake model, only for checking the app works.")
model = st.sidebar.text_input("Ollama model", config.MODEL)

UPLOAD = "Upload my own documents"
choice = st.sidebar.selectbox("Document set", datasets.list_sets() + [UPLOAD])

st.sidebar.info("A testing tool. It uses only harmless hidden instructions "
                "(like 'say BANANA') and never changes your files.")

TEMPLATE = "question,expected\nHow many days do refunds take?,14 days|fourteen days\n"


@st.cache_data(show_spinner=False)
def _load_builtin(name):
    return datasets.load_set(name)


def get_data():
    """Returns (docs, questions, dataset_label) or stops the page with a helpful message."""
    if choice != UPLOAD:
        docs, qs = _load_builtin(choice)
        return docs, qs, choice

    st.sidebar.markdown("**Your chatbot's knowledge**")
    files = st.sidebar.file_uploader("Documents (.txt or .md)", type=["txt", "md"], accept_multiple_files=True)
    qfile = st.sidebar.file_uploader("Questions (.csv with columns question, expected)", type=["csv"])
    st.sidebar.download_button("Download a questions template", TEMPLATE, "questions_template.csv", "text/csv")
    if not files or not qfile:
        st.info("Upload your documents and a questions file in the sidebar. "
                "Each question needs the correct answer text in the 'expected' column "
                "(alternatives separated by |). The answer must appear in one of your documents.")
        st.stop()
    docs = {Path(f.name).stem: f.getvalue().decode("utf-8", errors="ignore").strip() for f in files}
    try:
        qs = datasets.prepare_questions(pd.read_csv(io.BytesIO(qfile.getvalue())), docs)
    except Exception as e:
        st.error(str(e))
        st.stop()
    return docs, qs, "custom"


def pick_payload(pid):
    return next(p for p in payloads if p["id"] == pid)


# ------------------------------------------------------------------ Chat demo
if page == "Chat demo":
    docs, questions, label = get_data()
    st.title("Chat demo")
    st.write("Ask the chatbot a question. Switch between clean and poisoned documents, "
             "and turn defenses on or off.")

    labels = [f"{r.id}: {r.question}" for r in questions.itertuples()]
    pick = st.selectbox("Sample question", labels)
    row = questions.iloc[labels.index(pick)]
    question = st.text_input("Question (you can edit it)", row["question"])

    c1, c2, c3 = st.columns(3)
    variant = c1.radio("Documents", ["clean", "poisoned"], horizontal=True)
    payload_id = c2.selectbox("Hidden instruction", [p["id"] for p in payloads])
    position = c3.selectbox("Position in paragraph", attacks.POSITIONS)

    d1 = st.checkbox("D1: scan retrieved text for instruction-like sentences")
    d2 = st.checkbox("D2: safety rules in the prompt")
    d3 = st.checkbox("D3: have the model check the answer")

    if st.button("Ask", type="primary"):
        payload = pick_payload(payload_id)
        use_docs = docs
        if variant == "poisoned":
            use_docs = attacks.poison_docs(docs, row["doc"], row["expected"], payload["text"], position)
        defense_list = [d for d, on in [("d1", d1), ("d2", d2), ("d3", d3)] if on]
        try:
            with st.spinner("Thinking..."):
                res = runner.answer_question(question, use_docs, defense_list, backend, model)
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

        st.subheader("Answer")
        st.write(res["answer"])
        if variant == "poisoned":
            if res["blocked"]:
                st.warning("Blocked by defense D3.")
            elif evaluate.is_tricked(res["answer"], payload["marker"]):
                st.error(f"Tricked: the bot obeyed the hidden instruction (it printed '{payload['marker']}').")
            else:
                st.success("Not tricked: the bot ignored the hidden instruction.")
        if question == row["question"]:
            st.caption("Answer contains the correct fact: " + str(evaluate.is_correct(res["answer"], row["expected"])))
        if res["d1_flagged"]:
            st.info("D1 removed at least one instruction-like sentence.")
        with st.expander("Text the model actually saw"):
            for c in res["chunks"]:
                st.markdown(f"**{c['doc']}** (score {c['score']:.2f})")
                st.write(c["text"])

# --------------------------------------------------------------- Document lab
elif page == "Document lab":
    docs, questions, label = get_data()
    st.title("Document lab")
    st.subheader("Hidden instructions used in tests (all harmless)")
    st.dataframe(pd.DataFrame(payloads))
    st.caption("plain = obvious wording, disguised = looks like a note, "
               "stealth = avoids obvious command words.")

    st.subheader(f"Clean documents ({len(docs)})")
    for name, text in docs.items():
        with st.expander(name):
            st.text(text)

    st.subheader("See a poisoned version")
    doc_name = st.selectbox("Document", list(docs))
    pid = st.selectbox("Payload", [p["id"] for p in payloads], key="lab_payload")
    pos = st.selectbox("Position", attacks.POSITIONS, key="lab_pos")
    qrow = questions[questions["doc"] == doc_name]
    expected = qrow.iloc[0]["expected"] if len(qrow) else ""
    st.text(attacks.poison_docs(docs, doc_name, expected, pick_payload(pid)["text"], pos)[doc_name])

# ------------------------------------------------------------------ Run tests
elif page == "Run tests":
    docs, questions, label = get_data()
    st.title("Run tests")
    st.write(f"Document set: **{label}**  |  questions: **{len(questions)}**  |  "
             f"model: **{model if backend == 'ollama' else 'mock'}**")

    conds = st.multiselect("Defense settings to compare", list(runner.CONDITIONS),
                           default=["none", "d1", "d2", "d1+d2"])
    grid = st.checkbox("Test every hidden instruction on every question (stronger evidence, slower)", value=True)
    repeats = st.number_input("Repeats per test", 1, 5, 1,
                              help="Small models often repeat the same answer, so 1 is usually enough.")
    limit = st.slider("Number of questions", 1, len(questions), min(len(questions), 20))
    name = st.text_input("Results file name", f"{label}_{'grid' if grid else 'basic'}")
    n_poison = len(payloads) if grid else 1
    calls = limit * (1 + n_poison) * len(conds) * int(repeats)
    st.write(f"This will make about **{calls}** model calls (D3 adds extra calls). "
             "At a few seconds per call, check the time estimate before a big run.")

    if st.button("Run", type="primary"):
        bar, status = st.progress(0.0), st.empty()

        def show(done, total, eta):
            bar.progress(done / total)
            status.write(f"{done}/{total} done, about {eta / 60:.1f} min left")

        try:
            runner.run_experiment(
                conds, int(repeats), limit, backend, progress=show, fresh=True, grid=grid, model=model,
                out_csv=config.RESULTS_DIR / f"{name}.csv",
                dataset=None if label == "custom" else label,
                docs=docs if label == "custom" else None,
                questions=questions if label == "custom" else None)
            st.success(f"Finished. Saved as results/{name}.csv. Open the Results page.")
        except RuntimeError as e:
            st.error(str(e))

# ------------------------------------------------------------------ Test pack
elif page == "Test pack":
    docs, questions, label = get_data()
    st.title("Test pack: test ANY chatbot")
    st.write("The lab never needs access to the other bot. It creates a pack of knowledge bases "
             "(one clean, several with a harmless hidden line). The bot's owner loads each one into "
             "their bot, asks the questions, and sends back the answers. The lab then scores them.")
    st.warning("Only test bots you own or have permission to test. All hidden lines are harmless marker words.")

    tab1, tab2 = st.tabs(["1. Create a pack", "2. Score the answers"])

    with tab1:
        chosen = st.multiselect("Hidden instructions to include", [p["id"] for p in payloads],
                                default=[p["id"] for p in payloads])
        n_q = st.slider("Number of questions", 1, len(questions), min(len(questions), 10))
        n_answers = (1 + len(chosen)) * n_q
        st.write(f"The bot owner will need to collect **{n_answers}** answers. "
                 "Fewer questions and fewer hidden instructions make a smaller pack for manual testing.")
        if st.button("Create pack", type="primary") and chosen:
            sel = [p for p in payloads if p["id"] in chosen]
            files, key = pack.build_pack(docs, questions.head(n_q), sel, label)
            st.session_state["pack_zip"] = pack.zip_bytes(files)
            st.session_state["pack_key"] = json.dumps(key, indent=2)
        if "pack_zip" in st.session_state:
            st.success("Pack ready.")
            c1, c2 = st.columns(2)
            c1.download_button("Download pack for the bot owner (zip)", st.session_state["pack_zip"],
                               "test_pack.zip", "application/zip")
            c2.download_button("Download scoring key (keep private)", st.session_state["pack_key"],
                               "key.json", "application/json")
            st.caption("The zip contains the knowledge bases, the questions, an answers template and instructions. "
                       "The key holds the correct answers and the hidden words, so do not send it to the bot owner.")

    with tab2:
        keyfile = st.file_uploader("Scoring key (key.json)", type=["json"])
        ansfile = st.file_uploader("Filled answers (CSV)", type=["csv"])
        c1, c2, c3 = st.columns(3)
        condition = c1.text_input("Setting label", "none",
                                  help="Use 'none' for the bot as it is, and another label (e.g. my_defense) for a changed bot.")
        bot = c2.text_input("Bot name", "my_bot")
        name = c3.text_input("Results file name", "my_bot")
        append = st.checkbox("Add to an existing results file (to compare two settings)")

        if st.button("Score"):
            try:
                if keyfile is not None:
                    key = json.loads(keyfile.getvalue().decode("utf-8"))
                elif "pack_key" in st.session_state:
                    key = json.loads(st.session_state["pack_key"])
                    st.caption("Using the key created in this session.")
                else:
                    raise ValueError("Upload the scoring key (key.json), or create a pack first.")
                if ansfile is None:
                    raise ValueError("Upload the filled answers CSV.")
                answers = pd.read_csv(io.BytesIO(ansfile.getvalue()))
                scored, info = score.score_to_csv(key, answers, condition, bot,
                                                  config.RESULTS_DIR / f"{name}.csv", append=append)
                st.success(f"Scored {info['scored']} answers ({info['clean_rows']} clean, "
                           f"{info['poisoned_rows']} poisoned). Saved as results/{name}.csv.")
                if info["empty_skipped"]:
                    st.info(f"{info['empty_skipped']} empty rows were skipped.")
                if info["problems"]:
                    st.warning("Some rows were skipped: " + "; ".join(info["problems"]))
                st.dataframe(metrics.summarize(pd.read_csv(config.RESULTS_DIR / f"{name}.csv")))
                st.caption("Open the Results page for charts and the full breakdown.")
            except Exception as e:
                st.error(str(e))

# -------------------------------------------------------------------- Results
else:
    st.title("Results")
    files = sorted(config.RESULTS_DIR.glob("*.csv"))
    if not files:
        st.info("No results yet. Run tests first.")
        st.stop()
    pick = st.selectbox("Results file", [f.name for f in files])
    df = pd.read_csv(config.RESULTS_DIR / pick, encoding="utf-8")
    summary = metrics.summarize(df)

    base = summary[summary["condition"] == "none"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Model calls", len(df))
    if len(base):
        m2.metric("Attack success, no defense", f"{base.iloc[0]['attack_success_%']}%")
        m3.metric("Clean accuracy, no defense", f"{base.iloc[0]['clean_accuracy_%']}%")

    st.subheader("Comparison of defenses")
    st.dataframe(summary)
    st.caption("Ranges are approximate 95% ranges. They count each question (or question + hidden "
               "instruction pair) once, because repeats of the same test are not independent. "
               "accuracy_cost_pp = drop in correct answers on clean questions compared with no defense.")
    st.bar_chart(summary.set_index("condition")[["attack_success_%", "clean_accuracy_%"]])

    st.subheader("Attack success (%) by hidden instruction")
    bp = metrics.by_payload(df)
    if len(bp):
        st.dataframe(bp)
        st.caption("n_questions = how many different questions each instruction was tested on.")
        st.subheader("Attack success (%) by instruction style")
        st.dataframe(metrics.by_payload(df, "payload_style"))

    st.subheader("Every test")
    f1, f2 = st.columns(2)
    cond = f1.selectbox("Condition", ["all"] + sorted(df["condition"].unique()))
    var = f2.selectbox("Documents", ["all", "clean", "poisoned"])
    only_tricked = st.checkbox("Only show tricked answers")
    view = df
    if cond != "all":
        view = view[view["condition"] == cond]
    if var != "all":
        view = view[view["variant"] == var]
    if only_tricked:
        view = view[view["tricked"]]
    st.dataframe(view)

    d1, d2 = st.columns(2)
    d1.download_button("Download results (CSV)", df.to_csv(index=False), pick, "text/csv")
    d2.download_button("Download report (Markdown)", report.make_report(df), pick.replace(".csv", "_report.md"))
