"""
utils.py — Backend Engine for MNC Resume Optimizer 2028
Handles: PDF parsing · Embeddings · FAISS · Cosine Similarity · Groq LLM
"""

from __future__ import annotations

import io
import re
import string
import numpy as np
from typing import List

import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq


# ─── Stop-words ──────────────────────────────────────────────────────────────
_STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","need","this","that","these","those","i","we","you","he",
    "she","they","it","its","our","your","their","my","me","him","her","us",
    "them","what","which","who","how","when","where","why","not","no","so",
    "as","if","than","then","each","every","all","both","few","more","most",
    "other","some","such","into","through","during","before","after","above",
    "below","up","down","out","off","over","under","again","further","once",
    "about","against","between","across","also","just","any","very","own",
    "same","too","s","t","re","ve","ll","d","am","per","etc","eg","ie",
    "well","get","got","make","made","use","used",
}


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDING MODEL — cached permanently for the process lifetime
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def _get_embedding_model() -> SentenceTransformer:
    """
    Loads all-MiniLM-L6-v2 exactly ONCE per server process.
    st.cache_resource keeps it in RAM — every subsequent call returns instantly.
    First cold start: ~3-5 s.  All later runs: 0 s.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PDF INGESTION
# ═══════════════════════════════════════════════════════════════════════════════
def extract_text_from_pdf(file_obj) -> str:
    """
    Extracts and cleans text from a PDF file object.
    Works with single-column, double-column, and complex MNC resume layouts.
    """
    raw_chunks: List[str] = []
    with pdfplumber.open(io.BytesIO(file_obj.read())) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=3, y_tolerance=3)
            if text:
                raw_chunks.append(text)

    full_text = "\n".join(raw_chunks)
    return _clean_text(full_text)


def _clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = "".join(ch for ch in text if ch.isprintable() or ch == "\n")
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. KEYWORD EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
def extract_keywords(text: str, top_n: int = 40) -> List[str]:
    text_lower = text.lower()
    translator = str.maketrans("", "", string.punctuation)
    text_clean = text_lower.translate(translator)
    tokens = [t for t in text_clean.split() if t not in _STOP_WORDS and len(t) > 2]
    freq: dict[str, int] = {}
    for tok in tokens:
        freq[tok] = freq.get(tok, 0) + 1
    return sorted(freq, key=freq.get, reverse=True)[:top_n]


def find_missing_keywords(resume_text: str, jd_text: str, max_missing: int = 15) -> List[str]:
    jd_keywords  = extract_keywords(jd_text, top_n=50)
    translator   = str.maketrans("", "", string.punctuation)
    resume_words = set(resume_text.lower().translate(translator).split())
    return [kw for kw in jd_keywords if kw.lower() not in resume_words][:max_missing]


def find_matched_keywords(resume_text: str, jd_text: str, max_matched: int = 20) -> List[str]:
    jd_keywords  = extract_keywords(jd_text, top_n=50)
    translator   = str.maketrans("", "", string.punctuation)
    resume_words = set(resume_text.lower().translate(translator).split())
    return [kw for kw in jd_keywords if kw.lower() in resume_words][:max_matched]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. VECTOR EMBEDDING & COSINE SIMILARITY
# ═══════════════════════════════════════════════════════════════════════════════
def compute_similarity_score(resume_text: str, jd_text: str) -> float:
    """
    Embeds both texts using cached all-MiniLM-L6-v2 and computes cosine similarity.
    Returns a float in [0, 1].
    """
    model          = _get_embedding_model()          # instant after first load
    resume_snippet = _truncate_words(resume_text, 350)
    jd_snippet     = _truncate_words(jd_text, 350)
    embeddings     = model.encode([resume_snippet, jd_snippet])
    sim_matrix     = cosine_similarity([embeddings[0]], [embeddings[1]])
    return float(max(0.0, min(1.0, sim_matrix[0][0])))


def _truncate_words(text: str, max_words: int) -> str:
    return " ".join(text.split()[:max_words])


# ═══════════════════════════════════════════════════════════════════════════════
# 4. GROQ LLM — AI CAREER COACHING
# ═══════════════════════════════════════════════════════════════════════════════
def get_ai_analysis(
    resume_text: str,
    job_description: str,
    score: float,
    missing_keywords: List[str],
    api_key: str,
    model: str = "llama-3.1-8b-instant",
    max_tokens: int = 1500,
) -> str:
    client = Groq(api_key=api_key)

    missing_kw_str = ", ".join(missing_keywords) if missing_keywords else "None identified"

    system_prompt = (
        "You are a Senior HR Director and Career Coach at a Fortune 500 MNC with 20+ years "
        "of recruiting experience. You specialize in helping candidates land roles at top companies. "
        "Provide structured, actionable, and honest feedback. Format your response with clear "
        "sections using bold headers. Be specific, not generic."
    )

    user_prompt = f"""
## Resume Analysis Request

**Semantic Match Score:** {score}% (calculated via cosine similarity of embeddings)
**Missing Keywords from JD:** {missing_kw_str}

---

### Resume Content:
{_truncate_words(resume_text, 500)}

---

### Job Description:
{_truncate_words(job_description, 400)}

---

## Your Task:
Please provide a **structured career coaching report** with the following EXACT sections:

**1. 🎯 Match Analysis**
Explain what the {score}% score means in practical terms. Is it good enough to pass ATS screening?

**2. 🚨 Critical Skill Gaps**
Based on the missing keywords and JD requirements, list the top 5 skills/technologies the candidate MUST add or acquire. Be specific.

**3. ✍️ Bullet Point Rewrites (3 Examples)**
Take 3 implied weak areas from the resume and rewrite them as powerful, quantified, STAR-format bullet points that would impress an MNC recruiter.

**4. 🔑 ATS Optimization Tips**
Give 3 specific keyword injection tips — which exact phrases from the JD should be added and where in the resume.

**5. 📈 Overall Verdict**
Give an honest assessment: Is this resume ready to apply? What is the single most impactful change the candidate should make TODAY?
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.65,
    )
    return response.choices[0].message.content