%%writefile app.py
import os, io, html, time, inspect
import streamlit as st

st.set_page_config(page_title="Question Paper Blueprint Generator", page_icon="📝", layout="wide")

for _k in ("GEMINI_API_KEY", "GEMINI_MODEL", "USE_EMBEDDINGS"):      # hosting secrets -> environment
    try:
        os.environ.setdefault(_k, str(st.secrets[_k]))
    except Exception:
        pass

import pandas as pd
import plotly.express as px
import nlp_core as nc

DEFAULT_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_question_bank.csv")
CO_COLORS = {"CO1": "#6366f1", "CO2": "#06b6d4", "CO3": "#f59e0b", "CO4": "#ec4899"}

# ------------------------------------------------------------------ navigation (real, clickable)
TABS = ["🧹 Extraction & Vectorization", "🎯 CO Mapping & Info Extraction", "🧠 Topics & Multilingual",
        "🧮 Optimised Blueprint", "🤖 GenAI Assistant", "▶️ Live Pipeline"]
# pipeline pill -> section it opens
PIPE = [("① Extract", 0), ("② Clean", 0), ("③ Vectorize", 0), ("④ Map CO & Bloom", 1),
        ("⑤ Topics", 2), ("⑥ Optimise", 3), ("⑦ Paper + Blueprint", 3)]

if "nav" not in st.session_state:
    st.session_state["nav"] = TABS[0]

def go(tab_index):
    st.session_state["nav"] = TABS[tab_index]          # runs before the rerun, so the radio picks it up

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
html, body, [class*="css"] {font-family: 'Plus Jakarta Sans', sans-serif; -webkit-font-smoothing: antialiased;}
.stApp {background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 40%, #f5f3ff 100%);}
.block-container {padding-top: 1rem; max-width: 1280px;}

/* Hero Banner */
.hero {background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 80%, #6366f1 100%); border-radius: 22px;
       padding: 1.8rem 2.2rem; box-shadow: 0 20px 40px -10px rgba(79,70,229,.35); margin-bottom: 1.2rem; position: relative; overflow: hidden;}
.hero::after {content:''; position: absolute; right: -40px; top: -40px; width: 220px; height: 220px; background: rgba(255,255,255,0.06); border-radius: 50%; pointer-events: none;}
.hero h1 {margin: 0; font-size: 2.1rem; font-weight: 800; color: #fff; letter-spacing: -.6px; display: flex; align-items: center; gap: .6rem;}
.hero p {margin: .45rem 0 0 0; color: #e0e7ff; font-size: .98rem; font-weight: 400; max-width: 820px; line-height: 1.5;}
.hero .status-pill {display: inline-flex; align-items: center; gap: .45rem; background: rgba(255,255,255,.16); color: #fff; padding: .25rem .85rem;
       border-radius: 999px; margin: .9rem .4rem 0 0; font-size: .76rem; font-weight: 600; border: 1px solid rgba(255,255,255,0.22); backdrop-filter: blur(8px);}
.pulse-dot {width: 8px; height: 8px; border-radius: 50%; background: #34d399; box-shadow: 0 0 0 3px rgba(52,211,153,0.35); display: inline-block; animation: pulse 2s infinite;}
@keyframes pulse {0% {box-shadow: 0 0 0 0 rgba(52,211,153,0.6);} 70% {box-shadow: 0 0 0 6px rgba(52,211,153,0);} 100% {box-shadow: 0 0 0 0 rgba(52,211,153,0);}}

/* KPI Cards */
.kpi {background: rgba(255,255,255,0.85); backdrop-filter: blur(12px); border-radius: 18px; padding: 1.1rem 1.3rem;
      border: 1px solid rgba(226,232,240,0.85); box-shadow: 0 10px 25px -5px rgba(15,23,42,.05); border-left: 5px solid var(--c, #6366f1);
      transition: transform .2s ease, box-shadow .2s ease;}
.kpi:hover {transform: translateY(-2px); box-shadow: 0 14px 30px -5px rgba(15,23,42,.09);}
.kpi .v {font-size: 2rem; font-weight: 800; color: #0f172a; line-height: 1.1; letter-spacing: -.5px;}
.kpi .l {font-size: .72rem; color: #64748b; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; margin-top: .2rem;}

.sec-title {font-size: 1.25rem; font-weight: 800; color: #1e1b4b; margin: .6rem 0 .9rem 0; letter-spacing: -.3px; display: flex; align-items: center; gap: .5rem;}

/* Buttons & Controls */
.stButton > button {border-radius: 12px; font-weight: 700; padding: .6rem 1.4rem; transition: all .2s ease; border: 1px solid #e2e8f0;}
.stButton > button:hover {transform: translateY(-1px); box-shadow: 0 4px 12px rgba(15,23,42,.08);}
.stButton > button[kind="primary"] {background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); color: #fff; border: 0;
      box-shadow: 0 8px 20px -4px rgba(79,70,229,.45);}
.stButton > button[kind="primary"]:hover {box-shadow: 0 12px 26px -4px rgba(79,70,229,.6); transform: translateY(-1.5px);}

/* Sidebar */
section[data-testid="stSidebar"] {background: rgba(255,255,255,0.95); backdrop-filter: blur(14px); border-right: 1px solid #e2e8f0;}

/* Section Tabs */
div[role="radiogroup"][aria-label="Section"] {gap: .45rem; background: rgba(255,255,255,0.9); padding: .45rem; border-radius: 16px;
      border: 1px solid #e2e8f0; box-shadow: 0 4px 14px rgba(15,23,42,.04); flex-wrap: wrap;}
div[role="radiogroup"][aria-label="Section"] label {border-radius: 11px; padding: .5rem 1.1rem; margin: 0; cursor: pointer; transition: all .15s ease;}
div[role="radiogroup"][aria-label="Section"] label:hover {background: #f1f5f9;}
div[role="radiogroup"][aria-label="Section"] label > div:first-child {display: none;}
div[role="radiogroup"][aria-label="Section"] label p {font-weight: 700; font-size: .88rem; margin: 0;}
div[role="radiogroup"][aria-label="Section"] label:has(input:checked) {background: linear-gradient(135deg, #4f46e5, #7c3aed); box-shadow: 0 4px 14px rgba(79,70,229,.35);}
div[role="radiogroup"][aria-label="Section"] label:has(input:checked) p {color: #fff !important;}

/* Prestigious University Question Paper Box */
.paper {background: #fff; border: 1px solid #e2e8f0; border-radius: 20px; padding: 2rem 2.4rem;
        box-shadow: 0 14px 35px -8px rgba(15,23,42,.07); border-top: 5px solid #4f46e5;}
.paper .inst-title {text-align: center; font-size: .76rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; color: #4f46e5; margin-bottom: .25rem;}
.paper h3 {text-align: center; margin: 0; font-size: 1.45rem; font-weight: 900; color: #0f172a; letter-spacing: -.4px;}
.paper .course-meta {text-align: center; color: #475569; font-size: .88rem; font-weight: 600; margin-top: .15rem;}
.paper .exam-ribbon {display: flex; justify-content: space-between; align-items: center; background: #f8fafc; border: 1px solid #e2e8f0;
                     border-radius: 12px; padding: .55rem 1.1rem; margin: 1.1rem 0 1.3rem 0; font-size: .84rem; font-weight: 700; color: #334155;}
.paper .instructions {background: #fffbeb; border: 1px solid #fef3c7; border-left: 4px solid #f59e0b; border-radius: 10px;
                      padding: .65rem 1rem; margin-bottom: 1.4rem; font-size: .82rem; color: #92400e; line-height: 1.5;}
.qrow {padding: .95rem 0; border-bottom: 1px dashed #e2e8f0; display: flex; gap: 1rem; align-items: flex-start; transition: background .15s ease;}
.qrow:hover {background: #fafafa; border-radius: 10px; padding-left: .5rem; padding-right: .5rem;}
.qn {font-weight: 900; color: #4f46e5; min-width: 2.6rem; font-size: .95rem;}
.qt {flex: 1; color: #1e293b; font-size: .92rem; line-height: 1.55; font-weight: 500;}
.mk {font-weight: 800; color: #0f172a; background: #f1f5f9; padding: .25rem .65rem; border-radius: 8px; font-size: .82rem; border: 1px solid #e2e8f0; white-space: nowrap;}
.tag {display: inline-flex; align-items: center; padding: .12rem .65rem; border-radius: 999px; font-size: .68rem; font-weight: 800; color: #fff; margin: .35rem .3rem 0 0; letter-spacing: .02em;}
.tag-diff-easy {background: #059669;}
.tag-diff-medium {background: #d97706;}
.tag-diff-hard {background: #e11d48;}
</style>
""", unsafe_allow_html=True)

def kpi(col, label, value, color="#6366f1"):
    col.markdown(f'<div class="kpi" style="--c:{color}"><div class="v">{value}</div><div class="l">{label}</div></div>',
                 unsafe_allow_html=True)

def style(fig, h=340):
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), height=h, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Plus Jakarta Sans, sans-serif", color="#0f172a"))
    return fig

def title(text):
    st.markdown(f'<div class="sec-title">{text}</div>', unsafe_allow_html=True)

def paper_html(sel, total, heading):
    rows = ""
    for k, (_, r) in enumerate(sel.sort_values(["co", "marks"]).iterrows(), 1):
        c = CO_COLORS.get(r["co"], "#64748b")
        diff = str(r["difficulty"]).lower()
        diff_cls = f"tag-diff-{diff}" if diff in ("easy", "medium", "hard") else ""
        rows += (f'<div class="qrow"><div class="qn">Q{k:02d}.</div><div class="qt">{html.escape(str(r["question"]))}<br>'
                 f'<span class="tag" style="background:{c}">{r["co"]}</span>'
                 f'<span class="tag" style="background:#6366f1">{r["bloom"]}</span>'
                 f'<span class="tag {diff_cls}">{r["difficulty"]}</span></div>'
                 f'<div class="mk">[{int(r["marks"])} Marks]</div></div>')
    return (f'<div class="paper">'
            f'<div class="inst-title">MIT Academy of Engineering · Department of Computer Engineering</div>'
            f'<h3>{heading}</h3>'
            f'<div class="course-meta">Natural Language Processing (2311217L) · Autumn Semester 2026</div>'
            f'<div class="exam-ribbon"><span>⏱️ Time: 2.5 Hours</span><span style="color:#4f46e5;">Set A (Optimal Balanced Selection)</span><span>🎯 Maximum Marks: {total}</span></div>'
            f'<div class="instructions"><strong>General Instructions:</strong> Answer all questions. Figures to the right indicate full marks. CO and Bloom level indicators are provided for learning outcome audit.</div>'
            f'{rows}</div>')

# ------------------------------------------------------------------ hero
st.markdown(
    '<div class="hero">'
    '<h1>📝 Intelligent Question Paper Blueprint Generator</h1>'
    '<p>Transform raw question banks into rigorous, fully-balanced, CO-mapped exam papers using NLP feature extraction, Integer Linear Programming (HiGHS), and Generative AI.</p>'
    '<div class="status-pill"><span class="pulse-dot"></span>NLP Engine Online</div>'
    '<div class="status-pill">🎯 100% Balanced Blueprint</div>'
    '<div class="status-pill">✨ Powered by Gemini 3.8 Flash</div>'
    '</div>',
    unsafe_allow_html=True)

# pipeline buttons: each one jumps to the section that performs that step
pcols = st.columns(len(PIPE))
for i, (label, target) in enumerate(PIPE):
    pcols[i].button(label, key=f"pipe_{i}", on_click=go, args=(target,), width="stretch",
                    help=f"Open: {TABS[target]}")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    up = st.file_uploader("Question bank (CSV / XLSX / PDF / DOCX / TXT)", type=["csv", "xlsx", "pdf", "docx", "txt"])
    st.caption("No file? The built-in sample question bank is used.")
    total = st.number_input("Total marks", 10, 100, 50, 5)
    n_topics = st.slider("Number of topics", 2, 8, 4)
    topic_method = st.radio("Topic model", ["nmf", "lda"], horizontal=True)
    auto_k = st.checkbox("Auto-select topic count (coherence)", value=False)
    hot = st.slider("Higher-order share (Analyze / Evaluate / Create)", 0.0, 0.8, 0.3, 0.05)
    st.markdown("**CO weightage (%)**")
    raw_w = {co: st.slider(co, 0, 100, 25) for co in nc.COURSE_OUTCOMES}
    min_q, max_q = st.slider("Number of questions in paper", 4, 20, (8, 14))
    variant = st.number_input("Paper variant (0 = best fit, 1+ = alternative paper)", 0, 50, 0)
    st.divider()
    st.caption(f"LLM: {nc.GEMINI_MODEL} · key {'✅ set' if os.environ.get('GEMINI_API_KEY') else '❌ missing'}")

# ------------------------------------------------------------------ analysis (cached)
@st.cache_data(show_spinner="Running NLP pipeline...")
def analyse(file_bytes, name, n_topics, method, auto_k):
    src = io.BytesIO(file_bytes) if file_bytes else DEFAULT_DATA
    base = nc.load_question_bank(src, name)
    return nc.run_pipeline(base, n_topics=n_topics, method=method, auto_k=auto_k)   # one pass, one topic fit

try:
    df, topics, score = analyse(up.getvalue() if up else None, up.name if up else DEFAULT_DATA, n_topics, topic_method, auto_k)
except Exception as e:
    st.error(f"Could not process the file: {e}")
    st.stop()

df = nc.apply_llm_overrides(df, st.session_state.get("co_overrides", {}))   # LLM labels the user accepted
weights_sum = sum(raw_w.values())
weights = {k: v / weights_sum for k, v in raw_w.items()} if weights_sum else None

st.radio("Section", TABS, key="nav", horizontal=True, label_visibility="collapsed")
section = TABS.index(st.session_state["nav"])

# ================================================================== 1 · Extraction & Vectorization
if section == 0:
    title("Text extraction → cleaning → vectorization")
    bow, Xb, tfidf, Xt = nc.vectorize(df["clean"])
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Questions extracted", len(df))
    kpi(c2, "Vocabulary (1-2 grams)", Xt.shape[1], "#06b6d4")
    kpi(c3, "BoW sparsity", f"{1 - Xb.nnz / (Xb.shape[0] * Xb.shape[1]):.0%}", "#f59e0b")
    kpi(c4, "Marks in bank", int(df.marks.sum()), "#ec4899")
    st.markdown("**Raw question → cleaned and lemmatized tokens**")
    st.dataframe(df[["qid", "question", "marks", "clean"]], width="stretch", hide_index=True)
    st.markdown("**Top 15 TF-IDF terms**")
    top = pd.Series(Xt.sum(axis=0).A1, index=tfidf.get_feature_names_out()).nlargest(15)[::-1]
    fig = px.bar(x=top.values, y=top.index, orientation="h", labels={"x": "TF-IDF weight", "y": ""},
                 color_discrete_sequence=["#6366f1"])
    st.plotly_chart(style(fig, 420), width="stretch")
    with st.expander("Try your own text"):
        txt = st.text_input("Enter a question", "Explain how TF-IDF differs from Bag-of-Words.", key="own_text")
        if txt:
            st.write("**Cleaned:**", nc.preprocess(txt))

# ================================================================== 2 · CO mapping & info extraction
elif section == 1:
    title("CO mapping and information extraction")
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Low-confidence CO maps (<0.40)", int((df.co_conf < 0.4).sum()))
    kpi(c2, "Questions to re-check wording", int((df.tone_flag != "OK").sum()), "#f59e0b")
    if "co_true" in df:
        kpi(c3, "CO-mapping accuracy vs labels", f"{(df.co == df.co_true).mean():.0%}", "#10b981")
    st.dataframe(df[["qid", "question", "marks", "co", "co_conf", "bloom", "difficulty",
                     "keyphrases", "entities", "tone_flag"]], width="stretch", hide_index=True)
    a, b = st.columns(2)
    cnt = df.co.value_counts().sort_index()
    a.plotly_chart(style(px.bar(x=cnt.index, y=cnt.values, color=cnt.index, color_discrete_map=CO_COLORS,
                                labels={"x": "CO", "y": "Questions"}, title="Questions per CO")),
                   width="stretch")
    bl = df.bloom.value_counts().reindex(nc.BLOOM_ORDER, fill_value=0)
    b.plotly_chart(style(px.bar(x=bl.index, y=bl.values, labels={"x": "Bloom level", "y": "Questions"},
                                title="Questions per Bloom level", color_discrete_sequence=["#7c3aed"])),
                   width="stretch")

# ================================================================== 3 · Topics & multilingual
elif section == 2:
    title("Topic modeling")
    k1, k2 = st.columns(2)
    kpi(k1, "Topic coherence (NPMI, higher = better)", f"{nc.topic_coherence(topics, df['clean']):.2f}")
    kpi(k2, "Topic model", f"{topic_method.upper()} · {len(topics)} topics", "#06b6d4")
    st.write("")
    for k, w in topics.items():
        st.markdown(f"**Topic {k + 1}:** " + " ".join(f'`{x}`' for x in w))
    st.dataframe(df[["qid", "question", "topic_label"]], width="stretch", hide_index=True)
    tc = df.topic_label.value_counts()
    st.plotly_chart(style(px.bar(x=tc.values, y=tc.index, orientation="h", labels={"x": "Questions", "y": ""},
                                 color_discrete_sequence=["#f59e0b"])), width="stretch")
    st.divider()
    title("Multilingual NLP: detect and translate")
    ml_text = st.text_area("Enter a question in any language", "टोकनाइजेशन क्या है? उदाहरण सहित समझाइए।", key="ml_text")
    use_aws = st.checkbox("Use Amazon Translate (needs AWS credentials)", key="ml_aws")
    if st.button("Detect & translate", key="ml_btn", type="primary"):
        with st.spinner("Translating..."):
            lang = nc.detect_language(ml_text)
            st.write("**Detected language:**", lang)
            if lang == "en":
                st.write("**English:**", ml_text)
            else:
                eng, ok = nc.translate_text(ml_text, "auto", use_aws)
                if ok:
                    st.write("**English:**", eng)
                else:
                    st.warning("Translation failed (no internet / credentials). The original text was NOT translated.")

# ================================================================== 4 · Optimised blueprint
elif section == 3:
    title("Optimised paper and blueprint")
    if st.button("⚙️ Generate optimal paper", type="primary", key="gen_paper"):
        with st.spinner("Solving the integer program..."):
            st.session_state["result"] = nc.optimise_paper(df, total, weights, hot, min_q, max_q, seed=int(variant))
    res = st.session_state.get("result")
    if not res:
        st.info("Set your constraints in the sidebar, then click **Generate optimal paper**.")
    else:
        sel, report = res
        if sel is None:
            st.error("No feasible paper. Relax the constraints (question count range, marks) or add more questions.")
        else:
            for note in report.attrs.get("notes", []):
                st.warning("Constraints relaxed: " + note)
            bp = nc.build_blueprint(sel)
            m1, m2, m3 = st.columns(3)
            kpi(m1, "Questions selected", len(sel))
            kpi(m2, "Total marks", int(sel.marks.sum()), "#06b6d4")
            kpi(m3, "Topics covered", sel.topic.nunique(), "#f59e0b")
            st.write("")
            st.markdown(paper_html(sel, total, "Question Paper"), unsafe_allow_html=True)
            st.write("")
            l, r = st.columns(2)
            l.markdown("**Target vs achieved marks**")
            l.dataframe(report, width="stretch", hide_index=True)
            heat = bp.iloc[:-1, :-1]
            fig = px.imshow(heat, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                            labels=dict(x="Bloom level", y="Course outcome", color="Marks"))
            r.markdown("**Blueprint: marks by CO × Bloom level**")
            r.plotly_chart(style(fig, 300), width="stretch")
            st.markdown("**Blueprint table**")
            st.dataframe(bp, width="stretch")
            os.makedirs("outputs", exist_ok=True)
            path = nc.export_paper_docx(sel, bp, "outputs/Question_Paper.docx", "Question Paper", total)
            d1, d2 = st.columns(2)
            with open(path, "rb") as f:
                d1.download_button("⬇️ Download paper (DOCX)", f.read(), "Question_Paper.docx",
                                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                   key="dl_docx")
            d2.download_button("⬇️ Download selection (CSV)", sel.to_csv(index=False), "selected_questions.csv",
                               mime="text/csv", key="dl_csv")

# ================================================================== 5 · GenAI assistant
elif section == 4:
    title("Generative AI assistant")
    with st.expander("Prompt-engineering techniques used"):
        st.markdown("- **Role prompting:** system prompt sets the model as a senior examiner\n"
                    "- **Few-shot examples** for CO and Bloom classification\n"
                    "- **Structured output:** JSON-only responses that the app parses\n"
                    "- **Constraints:** CO, Bloom verb, marks, length, topic hint\n"
                    "- **Temperature:** low for classification, higher for question generation\n"
                    "- **Hybrid design:** NLP first; the LLM only handles low-confidence questions")
    if not os.environ.get("GEMINI_API_KEY"):
        st.info("Set GEMINI_API_KEY (environment variable or hosting secret) to enable these features.")
    else:
        st.markdown("**A. Re-classify low-confidence questions with the LLM (one batched call)**")
        thr = st.slider("Confidence threshold", 0.25, 0.80, 0.40, 0.05, key="llm_thr")
        if st.button("Run LLM classification", key="llm_cls"):
            low = df[df.co_conf < thr]
            if low.empty:
                st.success("No question is below the threshold.")
            else:
                try:
                    st.session_state["llm_proposals"] = nc.llm_classify_batch(list(zip(low.qid, low.question)))
                except Exception as e:
                    st.error(str(e))
        props = st.session_state.get("llm_proposals")
        if props:
            tbl = pd.DataFrame.from_dict(props, orient="index").rename_axis("qid").reset_index()
            tbl = tbl.merge(df[["qid", "question", "co", "bloom"]].rename(columns={"co": "nlp_co", "bloom": "nlp_bloom"}), on="qid")
            st.dataframe(tbl[["qid", "question", "nlp_co", "co", "nlp_bloom", "bloom", "reason"]], width="stretch", hide_index=True)
            def _apply_llm():
                st.session_state["co_overrides"] = dict(st.session_state.get("llm_proposals", {}))
            st.button("✅ Apply LLM labels to the question bank", key="llm_apply", on_click=_apply_llm)
        st.markdown("**B. Generate new questions to fill a blueprint gap**")
        g1, g2, g3, g4 = st.columns(4)
        gco = g1.selectbox("CO", list(nc.COURSE_OUTCOMES), key="g_co")
        gbl = g2.selectbox("Bloom level", nc.BLOOM_ORDER, index=2, key="g_bl")
        gmk = g3.selectbox("Marks", [2, 3, 5, 10], index=2, key="g_mk")
        ghint = g4.text_input("Topic hint", "topic modeling", key="g_hint")
        if st.button("Generate questions", key="llm_gen"):
            try:
                st.json(nc.llm_generate_questions(gco, gbl, gmk, ghint, n=2))
            except Exception as e:
                st.error(str(e))
        st.markdown("**C. LLM moderator review of the generated paper**")
        gen = st.session_state.get("result")
        if gen and gen[0] is not None:
            if st.button("Review paper", key="llm_rev"):
                try:
                    st.json(nc.llm_review(gen[0]))
                except Exception as e:
                    st.error(str(e))
        else:
            st.caption("Generate a paper in the Optimised Blueprint section first.")
            st.button("Go to Optimised Blueprint →", on_click=go, args=(3,), key="goto_bp")

# ================================================================== 6 · Live pipeline
else:
    title("Live pipeline: watch each stage run on real data")
    st.caption("Each stage shows the actual source code from nlp_core.py and the output it produces right now.")
    demo_q = st.text_input("Question to trace through the pipeline", df.question.iloc[min(1, len(df) - 1)], key="live_q")
    show_code = st.toggle("Show source code", value=True, key="live_code")

    def stage(name, funcs, runner):
        with st.status(name, expanded=True) as status:
            t0 = time.perf_counter()
            if show_code:
                for fn in funcs:
                    st.code(inspect.getsource(fn), language="python")
            runner()
            status.update(label=f"{name} · done in {time.perf_counter() - t0:.2f}s", state="complete")

    if st.button("▶️ Run pipeline step by step", type="primary", key="live_run"):
        raw = ("Q1. Define tokenization. [2]\n"
               "Q2. Explain the TF-IDF weighting scheme with a suitable example. (5 marks)\n"
               "3) Design a pipeline to extract text from scanned PDF documents. 10 marks\n")

        def r1():
            st.text(raw)
            qs = nc.split_questions(raw)
            st.dataframe(pd.DataFrame({"question": [nc.MARKS_RE.sub("", q).strip() for q in qs],
                                       "marks": [nc.extract_marks(q) for q in qs]}), hide_index=True)

        def r2():
            c1, c2 = st.columns(2)
            c1.markdown("**Input**"); c1.write(demo_q)
            c1.markdown("**After clean_text**"); c1.write(nc.clean_text(demo_q))
            c2.markdown("**After lemmatization + stop-word removal**"); c2.write(nc.preprocess(demo_q))

        def r3():
            bow, Xb, tfidf, Xt = nc.vectorize(df["clean"])
            v = tfidf.transform([nc.preprocess(demo_q)])
            terms = pd.Series(v.toarray()[0], index=tfidf.get_feature_names_out())
            st.write(f"BoW matrix: {Xb.shape} · TF-IDF matrix: {Xt.shape}")
            st.dataframe(terms[terms > 0].sort_values(ascending=False).rename("TF-IDF weight").to_frame())

        def r4():
            tmp = pd.DataFrame({"question": [demo_q]})
            tmp["clean"] = tmp.question.apply(nc.preprocess)
            out = nc.map_to_co(tmp)
            st.success(f"Mapped to **{out.co.iloc[0]}** with confidence **{out.co_conf.iloc[0]:.2f}**")

        def r5():
            tmp = pd.DataFrame({"question": [demo_q], "marks": [5]})
            out = nc.extract_info(tmp)
            st.dataframe(out[["bloom", "bloom_group", "difficulty", "keyphrases", "entities"]], hide_index=True)

        def r6():
            st.write(f"Method: **{topic_method.upper()}** · coherence: {nc.topic_coherence(topics, df['clean']):.2f}")
            for t, w in topics.items():
                st.write(f"Topic {t + 1}: " + ", ".join(w))

        def r7():
            sel_, rep_ = nc.optimise_paper(df, total, weights, hot, min_q, max_q, seed=int(variant))
            if sel_ is None:
                st.error("No feasible paper with the current settings.")
            else:
                st.session_state["live_result"] = (sel_, rep_)      # do not overwrite the user's real paper
                st.success(f"Selected {len(sel_)} questions, total {int(sel_.marks.sum())} marks")
                st.dataframe(rep_, hide_index=True)

        def r8():
            if not os.environ.get("GEMINI_API_KEY"):
                st.info("Set GEMINI_API_KEY to run this stage.")
            else:
                try:
                    st.json(nc.llm_classify(demo_q))
                except Exception as e:
                    st.error(str(e))

        stage("① Text extraction", [nc.split_questions, nc.extract_marks], r1)
        stage("② Preprocessing", [nc.clean_text, nc.preprocess], r2)
        stage("③ Vectorization (BoW and TF-IDF)", [nc.vectorize], r3)
        stage("④ CO mapping", [nc.map_to_co], r4)
        stage("⑤ Information extraction", [nc.detect_bloom, nc.extract_info], r5)
        stage("⑥ Topic modeling", [nc.topic_model], r6)
        stage("⑦ Optimisation (integer program)", [nc.optimise_paper], r7)
        stage("⑧ Generative AI (LLM classification)", [nc.llm_classify], r8)
        st.balloons()

st.divider()
st.caption("Intelligent Question Paper Blueprint Generator · Outcome-Based Education (OBE)")
