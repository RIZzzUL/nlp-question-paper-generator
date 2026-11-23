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
if not os.path.exists(DEFAULT_DATA):
    DEFAULT_DATA = os.path.join(os.getcwd(), "data", "sample_question_bank.csv")

# Dynamic palette generator for any number of Course Outcomes
PALETTE = ["#4f46e5", "#06b6d4", "#f59e0b", "#ec4899", "#10b981", "#8b5cf6", "#f97316", "#0284c7", "#84cc16", "#14b8a6"]

TABS = ["📊 Overview & Ingestion", "🎯 CO Mapping & Info Extraction", "🧠 Topics & Multilingual",
        "🧮 Optimised Blueprint", "🤖 GenAI Assistant", "▶️ Live Pipeline"]

if "nav" not in st.session_state:
    st.session_state["nav"] = TABS[0]

def go(tab_index):
    st.session_state["nav"] = TABS[tab_index]

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
html, body, [class*="css"] {font-family: 'Plus Jakarta Sans', sans-serif; -webkit-font-smoothing: antialiased;}
.stApp {background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 40%, #f5f3ff 100%);}
.block-container {padding-top: 1rem; max-width: 1280px;}

.hero {background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 80%, #6366f1 100%); border-radius: 22px;
       padding: 1.8rem 2.2rem; box-shadow: 0 20px 40px -10px rgba(79,70,229,.35); margin-bottom: 1.2rem; position: relative; overflow: hidden;}
.hero::after {content:''; position: absolute; right: -40px; top: -40px; width: 220px; height: 220px; background: rgba(255,255,255,0.06); border-radius: 50%; pointer-events: none;}
.hero h1 {margin: 0; font-size: 2.1rem; font-weight: 800; color: #fff; letter-spacing: -.6px; display: flex; align-items: center; gap: .75rem;}
.hero p {margin: .45rem 0 0 0; color: #e0e7ff; font-size: .98rem; font-weight: 400; max-width: 820px; line-height: 1.5;}
.hero .status-pill {display: inline-flex; align-items: center; gap: .45rem; background: rgba(255,255,255,.16); color: #fff; padding: .25rem .85rem;
       border-radius: 999px; margin: .9rem .4rem 0 0; font-size: .76rem; font-weight: 600; border: 1px solid rgba(255,255,255,0.22); backdrop-filter: blur(8px);}
.pulse-dot {width: 8px; height: 8px; border-radius: 50%; background: #34d399; box-shadow: 0 0 0 3px rgba(52,211,153,0.35); display: inline-block; animation: pulse 2s infinite;}
@keyframes pulse {0%, 100% {box-shadow: 0 0 0 3px rgba(52,211,153,0.35);} 50% {box-shadow: 0 0 0 6px rgba(52,211,153,0);}}

.kpi {background: rgba(255,255,255,0.85); backdrop-filter: blur(12px); border-radius: 18px; padding: 1.1rem 1.3rem;
      border: 1px solid rgba(226,232,240,0.85); box-shadow: 0 10px 25px -5px rgba(15,23,42,.05); border-left: 5px solid var(--c, #6366f1);
      transition: all .25s ease;}
.kpi:hover {transform: translateY(-2px); box-shadow: 0 14px 30px -5px rgba(15,23,42,.09);}
.kpi .num {font-size: 1.95rem; font-weight: 800; color: #0f172a; line-height: 1.15; font-family: 'JetBrains Mono', monospace;}
.kpi .lbl {font-size: .76rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: #64748b; margin-top: .35rem;}

.sec-title {font-size: 1.25rem; font-weight: 800; color: #1e1b4b; margin: .6rem 0 .9rem 0; letter-spacing: -.3px; display: flex; align-items: center; gap: .5rem;}
.stButton > button {border-radius: 12px; font-weight: 600; font-size: .88rem; padding: .55rem 1.25rem; transition: all .2s ease; border: 1px solid #e2e8f0;}
.stButton > button:hover {transform: translateY(-1px); box-shadow: 0 4px 12px rgba(15,23,42,.08);}
.stButton > button[kind="primary"] {background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%); border: none; box-shadow: 0 8px 20px -4px rgba(79,70,229,.45);}
.stButton > button[kind="primary"]:hover {box-shadow: 0 12px 26px -4px rgba(79,70,229,.6); transform: translateY(-1.5px);}

section[data-testid="stSidebar"] {background: rgba(255,255,255,0.95); backdrop-filter: blur(14px); border-right: 1px solid #e2e8f0;}
.stSelectbox, .stSlider, .stNumberInput {margin-bottom: .2rem;}

div[role="radiogroup"][aria-label="Section"] {gap: .45rem; background: rgba(255,255,255,0.9); padding: .45rem; border-radius: 16px;
    box-shadow: 0 4px 14px -2px rgba(15,23,42,.05); border: 1px solid #e2e8f0; margin-bottom: 1.3rem;}
div[role="radiogroup"][aria-label="Section"] > label {border-radius: 10px; padding: .5rem 1rem !important; font-weight: 600; font-size: .88rem;}

.tag {display: inline-block; padding: .2rem .55rem; border-radius: 6px; font-size: .72rem; font-weight: 700; color: #fff; margin-right: .3rem;}
.tag-diff-easy {background: #dcfce7; color: #15803d; border: 1px solid #bbf7d0;}
.tag-diff-medium {background: #fef3c7; color: #b45309; border: 1px solid #fde68a;}
.tag-diff-hard {background: #fee2e2; color: #b91c1c; border: 1px solid #fecaca;}

.paper {background: #ffffff; border-radius: 18px; padding: 2.2rem; border: 1px solid #e2e8f0; box-shadow: 0 16px 36px -8px rgba(15,23,42,.08); font-family: 'Plus Jakarta Sans', sans-serif;}
.paper .inst-title {text-align: center; font-size: .82rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; color: #4f46e5; margin-bottom: .25rem;}
.paper .exam-title {text-align: center; font-size: 1.45rem; font-weight: 800; color: #0f172a; margin: 0 0 .25rem 0;}
.paper .course-meta {text-align: center; font-size: .85rem; color: #64748b; font-weight: 600; margin-bottom: 1rem; border-bottom: 2px solid #f1f5f9; padding-bottom: .9rem;}
.paper .instructions {background: #fffbeb; border: 1px dashed #f59e0b; border-radius: 10px;
       padding: .65rem 1rem; margin-bottom: 1.4rem; font-size: .82rem; color: #92400e; line-height: 1.5;}
.qrow {display: flex; gap: 1rem; padding: .85rem 0; border-bottom: 1px solid #f1f5f9;}
.qrow:last-child {border-bottom: none;}
.qn {font-weight: 800; color: #4f46e5; width: 44px; font-family: 'JetBrains Mono', monospace; font-size: .92rem;}
.qt {flex: 1; color: #1e293b; font-size: .92rem; line-height: 1.55; font-weight: 500;}
.mk {font-weight: 800; color: #0f172a; background: #f1f5f9; padding: .25rem .65rem; border-radius: 8px; font-size: .82rem; border: 1px solid #e2e8f0; white-space: nowrap;}
</style>
""", unsafe_allow_html=True)

def title(text):
    st.markdown(f'<div class="sec-title">{text}</div>', unsafe_allow_html=True)

def kpi(col, label, val, color="#4f46e5"):
    col.markdown(f'<div class="kpi" style="--c:{color}"><div class="num">{val}</div><div class="lbl">{label}</div></div>',
                 unsafe_allow_html=True)

def style(fig, h=340):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=30, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Plus Jakarta Sans", color="#475569"))
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    return fig

# ------------------------------------------------------------------ Sidebar: Data & Settings
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    side_key = st.text_input("🔑 Gemini API Key", value=os.environ.get("GEMINI_API_KEY", ""), type="password", key="side_api_key", help="Enter key to enable live Google Gemini 2.5 calls")
    if side_key and side_key != os.environ.get("GEMINI_API_KEY", ""):
        os.environ["GEMINI_API_KEY"] = side_key.strip()
        nc._llm_client = None

    up = st.file_uploader("Question bank (CSV / XLSX / PDF / DOCX / TXT)", type=["csv", "xlsx", "pdf", "docx", "txt"])
    st.caption("No file uploaded? The built-in question bank is used.")

    # Topic model settings
    n_topics = st.slider("Number of topics", 2, 8, 4)
    topic_method = st.radio("Topic model", ["nmf", "lda"], horizontal=True)
    auto_k = st.checkbox("Auto-select topic count (coherence)", value=False)

# Cached analysis loader
@st.cache_data(show_spinner="Running NLP pipeline on question bank...")
def analyse(file_bytes, name, n_topics, method, auto_k):
    src = io.BytesIO(file_bytes) if file_bytes else DEFAULT_DATA
    base = nc.load_question_bank(src, name)
    return nc.run_pipeline(base, n_topics=n_topics, method=method, auto_k=auto_k)

try:
    df, topics, score = analyse(up.getvalue() if up else None, up.name if up else DEFAULT_DATA, n_topics, topic_method, auto_k)
except Exception as e:
    st.error(f"Could not process the question bank: {e}")
    st.stop()

# Apply any approved overrides
df = nc.apply_llm_overrides(df, st.session_state.get("co_overrides", {}))

# Dynamically discover Course Outcomes and parameters from the loaded data
active_cos = sorted(df["co"].dropna().unique().tolist()) if ("co" in df and len(df["co"].dropna())) else list(nc.COURSE_OUTCOMES)
CO_COLORS = {co: PALETTE[i % len(PALETTE)] for i, co in enumerate(active_cos)}

total_bank_marks = int(df["marks"].sum()) if ("marks" in df and not df.empty) else 100
min_single_mark = int(df["marks"].min()) if ("marks" in df and not df.empty) else 2
default_total_marks = min(50, total_bank_marks) if total_bank_marks >= 20 else total_bank_marks

bank_size = len(df)
min_q_limit = max(1, min(4, bank_size // 4))
max_q_limit = max(min_q_limit + 2, bank_size)
def_min_q = max(min_q_limit, min(8, bank_size // 3))
def_max_q = max(def_min_q + 1, min(14, bank_size * 2 // 3))

# Dynamic Examination Parameters in Sidebar
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🧮 Blueprint Constraints")
    total = st.number_input("Target Total Marks", min_value=max(2, min_single_mark), max_value=max(10, total_bank_marks), value=default_total_marks, step=5)
    hot = st.slider("Higher-Order Thinking share (Analyze / Evaluate / Create)", 0.0, 0.8, 0.3, 0.05)
    
    st.markdown("**Course Outcome Weightage (%)**")
    eq_w = max(1, round(100 / max(1, len(active_cos))))
    raw_w = {co: st.slider(co, 0, 100, eq_w, key=f"w_slider_{co}") for co in active_cos}
    
    min_q, max_q = st.slider("Number of questions in paper", min_q_limit, max_q_limit, (def_min_q, def_max_q))
    variant = st.number_input("Paper variant (0 = Set A, 1+ = Alternate Sets)", 0, 50, 0)
    
    with st.expander("📋 Exam & Header Details", expanded=False):
        cfg_inst = st.text_input("Institution Name", value="University Examination Board", key="cfg_inst")
        cfg_course = st.text_input("Course Name", value="Natural Language Processing", key="cfg_course")
        cfg_code = st.text_input("Course Code", value="", placeholder="e.g. CS601 / 2311217L", key="cfg_code")
        cfg_term = st.text_input("Session / Term", value="End-Semester Examination", key="cfg_term")
        cfg_instr = st.text_area("General Instructions", value="Answer all questions. Figures to the right indicate full marks. Justify answers with suitable examples where necessary.", key="cfg_instr")

    st.divider()
    st.caption(f"LLM: {nc.GEMINI_MODEL} · {'✅ Live Key' if nc.llm_available() else '⚡ NLP Assistant Active'}")

weights_sum = sum(raw_w.values())
weights = {k: v / weights_sum for k, v in raw_w.items()} if weights_sum else None

# Helper for rendering paper HTML dynamically
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
    
    meta_bits = [b for b in [cfg_course, cfg_code, cfg_term] if b]
    meta_line = " · ".join(meta_bits) if meta_bits else "Course Examination"
    
    return (f'<div class="paper">'
            f'<div class="inst-title">{html.escape(cfg_inst)}</div>'
            f'<div class="exam-title">{html.escape(heading)}</div>'
            f'<div class="course-meta">{html.escape(meta_line)}</div>'
            f'<div class="instructions"><b>General Instructions:</b> {html.escape(cfg_instr)}</div>'
            f'{rows}</div>')

# ------------------------------------------------------------------ Hero Header
st.markdown("""
<div class="hero">
  <h1>📝 Question Paper Blueprint Generator</h1>
  <p>Automate curriculum-aligned question paper assembly: extraction, Course Outcome mapping, Bloom's cognitive taxonomy classification, topic modeling, and mixed-integer linear programming (MILP) optimization.</p>
  <div>
    <span class="status-pill"><span class="pulse-dot"></span> Pipeline Ready</span>
    <span class="status-pill">OBE Aligned (CO1–CO4)</span>
    <span class="status-pill">MILP Optimization</span>
    <span class="status-pill">Bloom's Taxonomy</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.radio("Section", TABS, key="nav", horizontal=True, label_visibility="collapsed")
section = TABS.index(st.session_state["nav"])

# ================================================================== 1 · Overview & ingestion
if section == 0:
    title("Overview and text preprocessing")
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Total questions", len(df))
    kpi(c2, "Total bank marks", int(df.marks.sum()), "#06b6d4")
    kpi(c3, "Identified Course Outcomes", len(active_cos), "#f59e0b")
    kpi(c4, "Average marks / question", f"{df.marks.mean():.1f}", "#ec4899")

    st.markdown("**Question Bank Overview (Ingested & Processed)**")
    st.dataframe(df[["qid", "question", "marks", "clean"]], width="stretch", hide_index=True)
    
    bow, Xb, tfidf, Xt = nc.vectorize(df["clean"])
    st.markdown("**Top Salient TF-IDF Terms**")
    top = pd.Series(Xt.sum(axis=0).A1, index=tfidf.get_feature_names_out()).nlargest(15)[::-1]
    fig = px.bar(x=top.values, y=top.index, orientation="h", labels={"x": "TF-IDF weight", "y": ""},
                 color_discrete_sequence=["#6366f1"])
    st.plotly_chart(style(fig, 420), width="stretch")
    
    with st.expander("🔬 Test Text Preprocessing on Custom Input"):
        def_q_sample = df["question"].iloc[0] if len(df) else "Enter sample question text here..."
        txt = st.text_input("Enter a question to trace preprocessing", value=def_q_sample, key="own_text")
        if txt:
            col_raw, col_cln = st.columns(2)
            col_raw.markdown("**Raw Input:**"); col_raw.write(txt)
            col_cln.markdown("**Lemmatized & Normalized:**"); col_cln.write(nc.preprocess(txt))

# ================================================================== 2 · CO mapping & info extraction
elif section == 1:
    title("Course Outcome (CO) mapping & cognitive analytics")
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Low-confidence CO maps (<0.40)", int((df.co_conf < 0.4).sum()))
    kpi(c2, "Questions with tone warnings", int((df.tone_flag != "OK").sum()), "#f59e0b")
    if "co_true" in df:
        kpi(c3, "CO mapping accuracy vs true labels", f"{(df.co == df.co_true).mean():.0%}", "#10b981")
    else:
        kpi(c3, "Dominant Course Outcome", df["co"].mode().iloc[0] if len(df) else "N/A", "#10b981")
        
    st.dataframe(df[["qid", "question", "marks", "co", "co_conf", "bloom", "difficulty",
                     "keyphrases", "entities", "tone_flag"]], width="stretch", hide_index=True)
    a, b = st.columns(2)
    cnt = df.co.value_counts().sort_index()
    a.plotly_chart(style(px.bar(x=cnt.index, y=cnt.values, color=cnt.index, color_discrete_map=CO_COLORS,
                                labels={"x": "Course Outcome", "y": "Questions"}, title="Questions Distribution per CO")),
                   width="stretch")
    bl = df.bloom.value_counts().reindex(nc.BLOOM_ORDER, fill_value=0)
    b.plotly_chart(style(px.bar(x=bl.index, y=bl.values, labels={"x": "Bloom Cognitive Level", "y": "Questions"},
                                title="Questions Distribution per Bloom Level", color_discrete_sequence=["#7c3aed"])),
                   width="stretch")

# ================================================================== 3 · Topics & multilingual
elif section == 2:
    title("Topic modeling and latent discovery")
    k1, k2 = st.columns(2)
    kpi(k1, "Topic coherence (NPMI, higher = better)", f"{nc.topic_coherence(topics, df['clean']):.2f}")
    kpi(k2, "Active topic model", f"{topic_method.upper()} · {len(topics)} topics", "#06b6d4")
    st.write("")
    for k, w in topics.items():
        st.markdown(f"**Topic {k + 1}:** " + " ".join(f'`{x}`' for x in w))
    st.dataframe(df[["qid", "question", "topic_label"]], width="stretch", hide_index=True)
    tc = df.topic_label.value_counts()
    st.plotly_chart(style(px.bar(x=tc.values, y=tc.index, orientation="h", labels={"x": "Questions", "y": ""},
                                 color_discrete_sequence=["#f59e0b"])), width="stretch")
    
    st.divider()
    title("Multilingual NLP: batch language detection & translation")
    st.caption("Translate single or multiple questions/sentences simultaneously using our layered NLP translation pipeline.")

    # Multi-sentence sample loader
    c_m1, c_m2 = st.columns([3, 1])
    with c_m1:
        st.markdown("**Enter sentences to translate (one sentence per line):**")
    with c_m2:
        if st.button("📋 Load Multi-Sentence Samples", key="btn_sample_ml"):
            st.session_state["ml_text_input"] = (
                "टोकनाइजेशन क्या है? उदाहरण सहित समझाइए।\n"
                "स्टेमिंग और लेमेटाइजेशन में क्या अंतर है?\n"
                "Qu'est-ce que le traitement automatique du langage naturel?\n"
                "¿Cómo funciona la modelización de temas en NLP?\n"
                "Explain the role of stemming in text preprocessing."
            )

    default_ml = st.session_state.get(
        "ml_text_input",
        "टोकनाइजेशन क्या है? उदाहरण सहित समझाइए。\nस्टेमिंग और लेमेटाइजेशन में क्या अंतर है?"
    )

    ml_text = st.text_area("Input text (one sentence per line)", value=default_ml, height=130, key="ml_text",
                           placeholder="Paste or type multiple sentences in Hindi, French, Spanish, German, etc...")
    
    col_btn, col_chk = st.columns([2, 3])
    with col_btn:
        btn_trans = st.button("🌐 Translate All Sentences", key="ml_btn", type="primary")
    with col_chk:
        use_aws = st.checkbox("Use Amazon Translate (requires AWS credentials)", key="ml_aws")

    if btn_trans and ml_text.strip():
        lines = [line.strip() for line in ml_text.splitlines() if line.strip()]
        with st.spinner(f"Translating {len(lines)} sentence(s) in batch..."):
            results = nc.translate_batch(lines, use_aws=use_aws)
            st.session_state["last_batch_res"] = results

    if st.session_state.get("last_batch_res"):
        res_list = st.session_state["last_batch_res"]
        df_trans = pd.DataFrame(res_list)
        df_trans.index = range(1, len(df_trans) + 1)
        df_trans.index.name = "#"

        st.markdown(f"**Batch Translation Results ({len(df_trans)} sentences):**")
        disp_df = df_trans.rename(columns={
            "original": "Original Sentence",
            "detected_lang": "Language",
            "translation": "English Translation",
            "ok": "Success"
        })
        st.dataframe(disp_df, width="stretch")

        csv_data = disp_df.to_csv(index=True)
        st.download_button("⬇️ Download Translations (CSV)", csv_data, "translations.csv", "text/csv", key="dl_trans_csv")

# ================================================================== 4 · Optimised blueprint
elif section == 3:
    title("Optimised paper assembly and blueprint matrix")
    if st.button("⚙️ Generate Optimal Paper (MILP Solver)", type="primary", key="gen_paper"):
        with st.spinner("Solving the mixed-integer linear programming problem..."):
            st.session_state["result"] = nc.optimise_paper(df, total, weights, hot, min_q, max_q, seed=int(variant))
    
    res = st.session_state.get("result")
    if not res:
        st.info("Configure your constraints in the sidebar, then click **Generate Optimal Paper**.")
    else:
        sel, report = res
        if sel is None:
            st.error("No feasible paper found. Try relaxing the question count range, adjusting target marks, or adding more questions.")
        else:
            for note in report.attrs.get("notes", []):
                st.warning("Auto-relaxation note: " + note)
            bp = nc.build_blueprint(sel)
            m1, m2, m3 = st.columns(3)
            kpi(m1, "Questions selected", len(sel))
            kpi(m2, "Total paper marks", int(sel.marks.sum()), "#06b6d4")
            kpi(m3, "Topics covered", sel.topic.nunique(), "#f59e0b")
            
            st.write("")
            paper_heading = f"{cfg_course} Question Paper"
            st.markdown(paper_html(sel, total, paper_heading), unsafe_allow_html=True)
            st.write("")
            
            l, r = st.columns(2)
            l.markdown("**Target vs Achieved Marks Distribution**")
            l.dataframe(report, width="stretch", hide_index=True)
            
            heat = bp.iloc[:-1, :-1]
            fig = px.imshow(heat, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                            labels=dict(x="Bloom level", y="Course outcome", color="Marks"))
            r.markdown("**Blueprint Matrix: Marks by CO × Bloom Level**")
            r.plotly_chart(style(fig, 300), width="stretch")
            
            st.markdown("**Complete Blueprint Summary Table**")
            st.dataframe(bp, width="stretch")
            
            os.makedirs("outputs", exist_ok=True)
            doc_title = f"{cfg_course} ({cfg_code}) - {cfg_term}" if cfg_code else f"{cfg_course} - {cfg_term}"
            path = nc.export_paper_docx(sel, bp, "outputs/Question_Paper.docx", doc_title, total, instructions=cfg_instr)
            
            d1, d2 = st.columns(2)
            with open(path, "rb") as f:
                d1.download_button("⬇️ Download Official Paper (DOCX)", f.read(), "Question_Paper.docx",
                                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                   key="dl_docx")
            d2.download_button("⬇️ Download Selection Matrix (CSV)", sel.to_csv(index=False), "selected_questions.csv",
                               mime="text/csv", key="dl_csv")

# ================================================================== 5 · GenAI assistant
elif section == 4:
    title("Generative AI Assistant")
    with st.expander("ℹ️ Prompt-Engineering & Assistant Architecture"):
        st.markdown("- **Role Prompting:** System prompt establishes persona as a senior university examiner\n"
                    "- **Few-Shot In-Context Learning:** Real-world examples for Course Outcome and Bloom taxonomy classification\n"
                    "- **Structured Output:** Enforced JSON schema for reliable programmatic parsing\n"
                    "- **OBE Constraints:** Enforces Course Outcome, Bloom action verbs, marks, length, and topic hint\n"
                    "- **Dual Operation Mode:** Supports live Google Gemini 2.5 API or offline NLP Assistant mode\n"
                    "- **Hybrid NLP-LLM Design:** Classical NLP first; assistant handles low-confidence questions")

    # API Key Banner
    c_key1, c_key2 = st.columns([3, 1])
    with c_key1:
        user_key = st.text_input("🔑 Gemini API Key (Optional for Live GenAI Mode)",
                                 value=os.environ.get("GEMINI_API_KEY", ""),
                                 type="password",
                                 placeholder="Paste your Gemini API key to activate live Google GenAI...",
                                 key="genai_key_input")
        if user_key and user_key != os.environ.get("GEMINI_API_KEY", ""):
            os.environ["GEMINI_API_KEY"] = user_key.strip()
            nc._llm_client = None
            st.rerun()
    with c_key2:
        if nc.llm_available():
            st.success("✅ Live Gemini 2.5 Active")
        else:
            st.info("⚡ NLP Assistant Active")

    st.markdown("---")

    # A. Re-classify low-confidence questions
    st.markdown("### A. Re-classify Low-Confidence Questions (Hybrid NLP-LLM)")
    st.caption("Identify questions where initial NLP classification confidence is low and apply assistant re-classification.")
    thr = st.slider("Confidence threshold", 0.25, 0.80, 0.40, 0.05, key="llm_thr")
    if st.button("🔍 Run Assistant Classification", key="llm_cls", type="primary"):
        low = df[df.co_conf < thr]
        if low.empty:
            st.success("All questions meet or exceed the confidence threshold!")
        else:
            with st.spinner("Analyzing low-confidence questions..."):
                try:
                    st.session_state["llm_proposals"] = nc.llm_classify_batch(list(zip(low.qid, low.question)))
                    st.success(f"Successfully processed {len(low)} low-confidence question(s)!")
                except Exception as e:
                    st.error(f"Classification notice: {e}")

    props = st.session_state.get("llm_proposals")
    if props:
        tbl = pd.DataFrame.from_dict(props, orient="index").rename_axis("qid").reset_index()
        tbl = tbl.merge(df[["qid", "question", "co", "bloom"]].rename(columns={"co": "nlp_co", "bloom": "nlp_bloom"}), on="qid")
        st.dataframe(tbl[["qid", "question", "nlp_co", "co", "nlp_bloom", "bloom", "reason"]], width="stretch", hide_index=True)
        def _apply_llm():
            st.session_state["co_overrides"] = dict(st.session_state.get("llm_proposals", {}))
        st.button("✅ Apply Proposed Labels to Question Bank", key="llm_apply", on_click=_apply_llm)

    st.markdown("---")

    # B. Generate new questions to fill a blueprint gap
    st.markdown("### B. Generate New Questions for Blueprint Gaps")
    st.caption("Generate curriculum-aligned exam questions dynamically tailored to your selected Course Outcome, Bloom level, and marks.")
    g1, g2, g3, g4 = st.columns(4)
    gco = g1.selectbox("Course Outcome", active_cos, key="g_co")
    gbl = g2.selectbox("Bloom Cognitive Level", nc.BLOOM_ORDER, index=min(2, len(nc.BLOOM_ORDER)-1), key="g_bl")
    
    # Available marks derived dynamically from loaded question bank
    available_marks = sorted(df["marks"].dropna().unique().astype(int).tolist()) if ("marks" in df and len(df["marks"].dropna())) else [2, 3, 5, 10]
    def_mark_idx = available_marks.index(5) if 5 in available_marks else min(len(available_marks)-1, 1)
    gmk = g3.selectbox("Marks", available_marks, index=def_mark_idx, key="g_mk")
    
    # Dynamic topic hint extracted from keywords or question text of that CO
    def get_dynamic_hint(co_key):
        if co_key in nc.CO_KEYWORDS and nc.CO_KEYWORDS[co_key]:
            return ", ".join(nc.CO_KEYWORDS[co_key][:3]).title()
        if "co" in df and "question" in df:
            co_qs = df[df["co"] == co_key]["question"].tolist()
            if co_qs:
                words = [w for w in re.findall(r"\b[A-Za-z]{4,}\b", co_qs[0]) if w.lower() not in nc.QUESTION_STOP]
                if words:
                    return " ".join(words[:3]).title()
        return f"{co_key} Core Topics"

    dynamic_hint_val = get_dynamic_hint(gco)
    ghint = g4.text_input("Topic Hint", value=dynamic_hint_val, key=f"g_hint_{gco}")
    
    if st.button("✨ Generate Questions", key="llm_gen", type="primary"):
        with st.spinner(f"Generating questions for {gco} · {gbl} · {gmk} Marks..."):
            try:
                gen_items = nc.llm_generate_questions(gco, gbl, gmk, ghint, n=2, existing=df.question)
                st.session_state["last_gen_questions"] = gen_items
                st.session_state["last_gen_meta"] = f"{gco} · {gbl} · {gmk} Marks · Topic: '{ghint}'"
            except Exception as e:
                st.error(f"Generation error: {e}")

    if st.session_state.get("last_gen_questions"):
        meta_label = st.session_state.get("last_gen_meta", "")
        st.markdown(f"**Generated Questions ({meta_label}):**")
        for i, q_item in enumerate(st.session_state["last_gen_questions"], 1):
            st.info(f"**Q{i} [{q_item.get('marks', gmk)} Marks]:** {q_item.get('question')}")
        with st.expander("View Raw JSON"):
            st.json(st.session_state["last_gen_questions"])

    st.markdown("---")

    # C. Moderator review of the generated paper
    st.markdown("### C. Question Paper Quality & Balance Moderator Review")
    st.caption("Audits the assembled question paper for Course Outcome coverage, Bloom spread, ambiguity, and question duplication.")
    gen = st.session_state.get("result")
    if gen and gen[0] is not None:
        if st.button("📋 Run Paper Quality Review", key="llm_rev", type="primary"):
            with st.spinner("Auditing question paper against OBE standards..."):
                try:
                    rev_data = nc.llm_review(gen[0])
                    st.session_state["last_paper_review"] = rev_data
                except Exception as e:
                    st.error(f"Review notice: {e}")

        if st.session_state.get("last_paper_review"):
            rev = st.session_state["last_paper_review"]
            v_color = "green" if rev.get("verdict") == "Good" else "orange"
            st.markdown(f"#### Overall Assessment: <span style='color:{v_color}; font-weight:bold;'>{rev.get('verdict', 'Evaluated')}</span>", unsafe_allow_html=True)
            col_iss, col_sug = st.columns(2)
            with col_iss:
                st.markdown("**⚠️ Identified Observations / Gaps:**")
                for iss in rev.get("issues", []):
                    st.markdown(f"- {iss}")
            with col_sug:
                st.markdown("**💡 Recommendations & Suggestions:**")
                for sug in rev.get("suggestions", []):
                    st.markdown(f"- {sug}")
    else:
        st.info("ℹ️ Generate a question paper in the **🧮 Optimised Blueprint** section first to enable moderation review.")
        st.button("Go to Optimised Blueprint →", on_click=go, args=(3,), key="goto_bp")

# ================================================================== 6 · Live pipeline
else:
    title("Live pipeline: trace each processing stage")
    st.caption("Each stage displays the underlying mathematical transformation and the real-time output.")
    
    # Dynamic question selection from bank
    q_opts = list(df.question.unique()) if "question" in df else []
    sel_idx = st.selectbox("Select a question from your bank to trace (or choose custom):", 
                           ["-- Enter Custom Question --"] + q_opts[:50], index=1 if q_opts else 0)
    def_trace = q_opts[0] if (sel_idx == "-- Enter Custom Question --" and q_opts) else (sel_idx if sel_idx != "-- Enter Custom Question --" else "Explain how text preprocessing transforms raw documents.")
    demo_q = st.text_input("Question to trace through pipeline", value=def_trace, key="live_q")
    show_code = st.toggle("Show source code", value=True, key="live_code")

    def stage(name, funcs, runner):
        with st.status(name, expanded=True) as status:
            t0 = time.perf_counter()
            if show_code:
                for fn in funcs:
                    st.code(inspect.getsource(fn), language="python")
            runner()
            status.update(label=f"{name} · done in {time.perf_counter() - t0:.2f}s", state="complete")

    if st.button("▶️ Run Pipeline Step by Step", type="primary", key="live_run"):
        # Dynamically sample from df
        sample_q_rows = []
        for i, row in enumerate(df.head(3).itertuples(), 1):
            sample_q_rows.append(f"Q{i}. {row.question} [{int(row.marks)} marks]")
        raw_stream = "\n".join(sample_q_rows) if sample_q_rows else "Q1. Define tokenization [2 marks]\nQ2. Explain TF-IDF [5 marks]"

        def r1():
            st.text(raw_stream)
            qs = nc.split_questions(raw_stream)
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
            m_val = df[df["question"] == demo_q]["marks"].iloc[0] if (demo_q in df["question"].values and "marks" in df) else 5
            tmp = pd.DataFrame({"question": [demo_q], "marks": [m_val]})
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
                st.success(f"MILP selected {len(sel_)} questions for {int(sel_.marks.sum())} marks.")

        def r8():
            try:
                prop = nc.llm_classify(demo_q)
                st.json(prop)
            except Exception as e:
                st.warning(f"GenAI Stage notice: {e}")
                prop = nc.classify_question_offline(demo_q)
                st.json(prop)

        stage("① Text extraction and marks parsing", [nc.split_questions, nc.extract_marks], r1)
        stage("② Cleaning and lemmatization", [nc.clean_text, nc.preprocess], r2)
        stage("③ Vectorization (BoW + TF-IDF)", [nc.vectorize], r3)
        stage("④ Course outcome mapping", [nc.map_to_co], r4)
        stage("⑤ Information extraction (Bloom + tone)", [nc.detect_bloom, nc.extract_info], r5)
        stage("⑥ Topic modeling", [nc.topic_model], r6)
        stage("⑦ Optimisation (integer program)", [nc.optimise_paper], r7)
        stage("⑧ Generative AI (LLM classification)", [nc.llm_classify], r8)
        st.balloons()

st.divider()
st.caption("Intelligent Question Paper Blueprint Generator · Outcome-Based Education (OBE)")
