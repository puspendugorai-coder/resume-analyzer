import streamlit as st
import os
from dotenv import load_dotenv
from utils import (
    extract_text_from_pdf,
    compute_similarity_score,
    get_ai_analysis,
    extract_keywords,
    find_missing_keywords,
    find_matched_keywords,
)

# ─── Load environment ────────────────────────────────────────────────────────
load_dotenv()

# ─── Pre-warm the embedding model in background (so first analysis is instant) ─
from utils import _get_embedding_model
_get_embedding_model()   # triggers cache load once at startup — silent & fast

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MNC Resume Analyzer",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
  }

  /* ── Dark base ── */
  .stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0f0a 100%);
    color: #e8e8f0;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a1628 100%);
    border-right: 1px solid #1e3a5f40;
  }
  [data-testid="stSidebar"] * { color: #c9d1d9 !important; }

  /* ── Hero header ── */
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00ff88 0%, #00ccff 50%, #7c3aed 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
    letter-spacing: -1px;
    margin-bottom: 0;
  }
  .hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    font-weight: 300;
    color: #6b7a8d;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
  }

  /* ── Cards ── */
  .glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    backdrop-filter: blur(12px);
    margin-bottom: 16px;
  }
  .section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #00ff88;
    margin-bottom: 12px;
  }

  /* ── Score ring ── */
  .score-container {
    text-align: center;
    padding: 32px 16px;
  }
  .score-ring {
    font-family: 'Syne', sans-serif;
    font-size: 5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00ff88, #00ccff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
  }
  .score-label {
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6b7a8d;
    margin-top: 4px;
  }

  /* ── Score tiers ── */
  .tier-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    margin-top: 8px;
  }
  .tier-excellent { background: #00ff8820; color: #00ff88; border: 1px solid #00ff8840; }
  .tier-good      { background: #00ccff20; color: #00ccff; border: 1px solid #00ccff40; }
  .tier-average   { background: #ffa50020; color: #ffa500; border: 1px solid #ffa50040; }
  .tier-low       { background: #ff444420; color: #ff4444; border: 1px solid #ff444440; }

  /* ── Keyword chips ── */
  .keyword-chip {
    display: inline-block;
    background: rgba(255,68,68,0.1);
    border: 1px solid rgba(255,68,68,0.3);
    color: #ff8888;
    border-radius: 8px;
    padding: 4px 12px;
    margin: 4px;
    font-size: 0.8rem;
    font-family: 'DM Sans', sans-serif;
  }
  .keyword-chip-match {
    background: rgba(0,255,136,0.1);
    border: 1px solid rgba(0,255,136,0.3);
    color: #00ff88;
  }

  /* ── AI output ── */
  .ai-block {
    background: rgba(0,204,255,0.04);
    border-left: 3px solid #00ccff;
    border-radius: 0 12px 12px 0;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.95rem;
    line-height: 1.7;
    color: #c9d1d9;
  }

  /* ── Progress bar override ── */
  .stProgress > div > div {
    background: linear-gradient(90deg, #00ff88, #00ccff) !important;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    letter-spacing: 1px;
    color: #6b7a8d;
  }
  .stTabs [aria-selected="true"] {
    color: #00ff88 !important;
    border-bottom: 2px solid #00ff88 !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #00ff88, #00ccff);
    color: #0a0a0f;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: 1px;
    border: none;
    border-radius: 12px;
    padding: 12px 32px;
    font-size: 0.95rem;
    transition: all 0.3s ease;
    width: 100%;
  }
  .stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0,255,136,0.3);
  }

  /* ── Divider ── */
  .neon-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #00ff8840, #00ccff40, transparent);
    margin: 24px 0;
  }

  /* ── Upload zone ── */
  [data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(0,255,136,0.25) !important;
    border-radius: 12px !important;
    transition: all 0.3s;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: rgba(0,255,136,0.6) !important;
    background: rgba(0,255,136,0.04) !important;
  }

  /* Sidebar model badge */
  .model-info {
    background: rgba(0,204,255,0.08);
    border: 1px solid rgba(0,204,255,0.2);
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.8rem;
    color: #6b7a8d;
    margin-top: 12px;
  }

  /* ── Hide sidebar input/select labels completely ── */
  [data-testid="stSidebar"] .stTextInput label,
  [data-testid="stSidebar"] .stSelectbox label {
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
  }
  [data-testid="stSidebar"] .stTextInput,
  [data-testid="stSidebar"] .stSelectbox {
    margin-top: 0 !important;
  }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px;'>
      <div style='font-family:Syne,sans-serif; font-size:1.4rem; font-weight:800;
                  background:linear-gradient(135deg,#00ff88,#00ccff);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        ⚡ MNC OPTIMIZER
      </div>
      <div style='font-size:0.7rem; letter-spacing:3px; color:#3d4f61; margin-top:4px;'>
        RESUME INTELLIGENCE ENGINE
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # ── API Key ── (label hidden)
    st.markdown('<p class="section-label">🔐 API Configuration</p>', unsafe_allow_html=True)
    env_key = os.getenv("GROQ_API_KEY", "")
    if env_key:
        st.success("✅ API key loaded")
        groq_api_key = env_key
    else:
        groq_api_key = st.text_input(
            "API Key",
            type="password",
            placeholder="Paste your Groq key here…",
            help="Get your free key at console.groq.com",
            label_visibility="hidden",
        )
        if groq_api_key:
            st.success("✅ Key entered successfully")
        else:
            st.warning("⚠️ API key required to run analysis")

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # ── Model Selector ── (label & model ID hidden)
    st.markdown('<p class="section-label">🤖 Select AI Model</p>', unsafe_allow_html=True)
    model_options = {
        "Llama 3.1 — 8B  (⚡ Fastest)":   "llama-3.1-8b-instant",
        "Llama 3.3 — 70B (🧠 Smartest)":  "llama-3.3-70b-versatile",
        "Llama 4 Scout   (🔮 Balanced)":   "meta-llama/llama-4-scout-17b-16e-instruct",
    }
    model_label    = st.selectbox("AI Model", list(model_options.keys()), index=0, label_visibility="hidden")
    selected_model = model_options[model_label]

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

    # ── Info ──
    st.markdown("""
    <div style='font-size:0.78rem; color:#3d4f61; line-height:1.8;'>
      <b style='color:#6b7a8d;'>How it works:</b><br>
      1️⃣ Upload your resume PDF<br>
      2️⃣ Paste the job description<br>
      3️⃣ Click Analyze<br>
      4️⃣ Get AI-powered insights
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BODY
# ═══════════════════════════════════════════════════════════════════════════════

# ── Hero Header ──
st.markdown("""
<div style='padding: 32px 0 8px;'>
  <div class='hero-title'>MNC Resume Analyzer</div>
  <div class='hero-sub'>AI-Powered Career Intelligence</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

# ── Two-Column Upload ──
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<p class="section-label">📄 Resume (PDF)</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your resume here",
        type=["pdf"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.markdown(f"""
        <div style='background:rgba(0,255,136,0.06); border:1px solid rgba(0,255,136,0.2);
                    border-radius:10px; padding:10px 14px; margin-top:8px; font-size:0.85rem; color:#00ff88;'>
          ✅ <b>{uploaded_file.name}</b> · {round(uploaded_file.size/1024, 1)} KB
        </div>
        """, unsafe_allow_html=True)

with col_right:
    st.markdown('<p class="section-label">📋 Job Description</p>', unsafe_allow_html=True)
    job_description = st.text_area(
        "Paste JD here",
        height=220,
        placeholder="Paste the full job description here — including required skills, responsibilities, and qualifications…",
        label_visibility="collapsed",
    )
    if job_description:
        word_count = len(job_description.split())
        st.markdown(f"""
        <div style='font-size:0.75rem; color:#3d4f61; margin-top:4px;'>
          {word_count} words · {"✅ Good length" if word_count > 50 else "⚠️ Add more detail for better results"}
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

# ── Analyze Button ──
col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 2])
with col_btn2:
    analyze_btn = st.button("🚀 Analyze Resume", use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
if analyze_btn:
    # ── Validation ──
    if not groq_api_key:
        st.error("❌ Please enter your Groq API key in the sidebar.")
        st.stop()
    if not uploaded_file:
        st.error("❌ Please upload your resume PDF.")
        st.stop()
    if not job_description.strip():
        st.error("❌ Please paste the job description.")
        st.stop()

    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">⚙️ Processing Pipeline</p>', unsafe_allow_html=True)

    progress_bar = st.progress(0)
    status_text  = st.empty()

    # Step 1: Extract PDF
    status_text.markdown("🔍 **Step 1/4** — Ingesting PDF and extracting text…")
    progress_bar.progress(10)
    with st.spinner("Parsing your resume…"):
        resume_text = extract_text_from_pdf(uploaded_file)
    if not resume_text or len(resume_text) < 50:
        st.error("❌ Could not extract text from the PDF. Please ensure it's a text-based PDF (not scanned).")
        st.stop()
    progress_bar.progress(25)

    # Step 2: Similarity Score
    status_text.markdown("🧮 **Step 2/4** — Computing semantic similarity vectors…")
    with st.spinner("Running cosine similarity engine…"):
        score = compute_similarity_score(resume_text, job_description)
    score_pct = round(score * 100, 1)
    progress_bar.progress(55)

    # Step 3: Keywords
    status_text.markdown("🔑 **Step 3/4** — Extracting and matching keywords…")
    with st.spinner("Scanning keywords…"):
        missing_keywords = find_missing_keywords(resume_text, job_description)
        matched_keywords = find_matched_keywords(resume_text, job_description)
    progress_bar.progress(70)

    # Step 4: LLM Analysis
    status_text.markdown("🤖 **Step 4/4** — Calling Groq LLM for AI career coaching…")
    with st.spinner(f"Consulting {model_label.split('(')[0].strip()}…"):
        ai_analysis = get_ai_analysis(
            resume_text=resume_text,
            job_description=job_description,
            score=score_pct,
            missing_keywords=missing_keywords,
            api_key=groq_api_key,
            model=selected_model,
        )
    progress_bar.progress(100)
    status_text.markdown("✅ **Analysis complete!**")

    import time; time.sleep(0.5)
    progress_bar.empty()
    status_text.empty()

    # ─── RESULTS DASHBOARD ─────────────────────────────────────────────────
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    st.markdown('<p class="section-label">📊 Results Dashboard</p>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈  Score & Gaps", "🧠  AI Career Coach"])

    # ── TAB 1: Score & Gaps ──────────────────────────────────────────────────
    with tab1:
        # Score + tier
        if score_pct >= 75:
            tier, tier_cls = "EXCELLENT MATCH", "tier-excellent"
        elif score_pct >= 55:
            tier, tier_cls = "GOOD MATCH", "tier-good"
        elif score_pct >= 35:
            tier, tier_cls = "AVERAGE MATCH", "tier-average"
        else:
            tier, tier_cls = "LOW MATCH", "tier-low"

        col_score, col_metrics = st.columns([1, 2], gap="large")

        with col_score:
            st.markdown(f"""
            <div class="glass-card score-container">
              <div style='font-size:0.7rem; letter-spacing:3px; color:#3d4f61; text-transform:uppercase;
                          font-family:Syne,sans-serif; margin-bottom:8px;'>Match Score</div>
              <div class="score-ring">{score_pct}%</div>
              <div class="score-label">Cosine Similarity</div>
              <div><span class="tier-badge {tier_cls}">{tier}</span></div>
            </div>
            """, unsafe_allow_html=True)

        with col_metrics:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<p class="section-label">📉 Score Breakdown</p>', unsafe_allow_html=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Resume Words",   f"{len(resume_text.split()):,}")
            m2.metric("JD Words",       f"{len(job_description.split()):,}")
            m3.metric("Missing Skills", f"{len(missing_keywords)}")

            st.markdown('<p class="section-label" style="margin-top:16px;">🎯 Score Gauge</p>', unsafe_allow_html=True)
            st.progress(int(score_pct))
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

        # Keywords
        kw_col1, kw_col2 = st.columns(2)

        with kw_col1:
            st.markdown('<p class="section-label">❌ Missing Keywords</p>', unsafe_allow_html=True)
            if missing_keywords:
                chips = "".join(
                    f'<span class="keyword-chip">{kw}</span>'
                    for kw in missing_keywords[:20]
                )
                st.markdown(f'<div class="glass-card">{chips}</div>', unsafe_allow_html=True)
            else:
                st.success("🎉 No significant missing keywords found!")

        with kw_col2:
            st.markdown('<p class="section-label">✅ Matched Keywords</p>', unsafe_allow_html=True)
            if matched_keywords:
                chips = "".join(
                    f'<span class="keyword-chip keyword-chip-match">{kw}</span>'
                    for kw in matched_keywords
                )
                st.markdown(f'<div class="glass-card">{chips}</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Few keyword matches found. Consider tailoring your resume.")

    # ── TAB 2: AI Career Coach ───────────────────────────────────────────────
    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p class="section-label">🤖 AI Career Coach — Detailed Analysis</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="ai-block">{ai_analysis}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Download
        st.download_button(
            label="⬇️ Download AI Report (.txt)",
            data=f"MNC RESUME ANALYSIS REPORT\n{'='*50}\n\nMatch Score: {score_pct}%\nTier: {tier}\n\nMissing Keywords:\n{chr(10).join(f'- {k}' for k in missing_keywords)}\n\nAI CAREER COACH FEEDBACK:\n{ai_analysis}",
            file_name="resume_analysis_report.txt",
            mime="text/plain",
        )

# ── Empty state ──
elif not analyze_btn:
    st.markdown("""
    <div style='text-align:center; padding:48px 0; color:#2a3a4a;'>
      <div style='font-size:4rem; margin-bottom:16px;'>🚀</div>
      <div style='font-family:Syne,sans-serif; font-size:1.1rem; font-weight:700; color:#3d4f61;'>
        Upload your resume and paste a job description to begin
      </div>
      <div style='font-size:0.85rem; color:#2a3a4a; margin-top:8px;'>
        Powered by Groq · LangChain · FAISS · HuggingFace Embeddings
      </div>
    </div>
    """, unsafe_allow_html=True)