# 🚀 MNC Resume Optimizer 2028
### AI-Powered Resume Analysis Engine · Groq + LangChain + FAISS + HuggingFace

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit)
![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Llama%203-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📌 What This Does

A production-grade AI pipeline that:
1. **Extracts** raw text from your PDF resume (handles complex layouts)
2. **Embeds** both resume and job description into 384-dimensional vectors
3. **Calculates** cosine similarity for a semantic match score
4. **Identifies** missing keywords from the JD
5. **Generates** a detailed AI career coaching report via Groq (Llama 3)

---

## 🏗️ Architecture

```
User Upload (PDF)
     │
     ▼
pdfplumber ──► Raw Text
     │
     ▼
SentenceTransformer (all-MiniLM-L6-v2)
     │
     ├──► Resume Vector (384-dim)
     └──► JD Vector     (384-dim)
               │
               ▼
        Cosine Similarity ──► Match Score (0–100%)
               │
               ▼
        Groq API (Llama 3.1 / 3.3 / 4)
               │
               ▼
        AI Career Coach Report ──► Streamlit Dashboard
```

---

## 🛠️ Tech Stack

| Component    | Library / API                    | Why?                                  |
|-------------|----------------------------------|---------------------------------------|
| Frontend     | `streamlit`                      | Fast, beautiful Python UI             |
| PDF Parsing  | `pdfplumber`                     | Best for complex resume layouts       |
| Embeddings   | `sentence-transformers`          | Free, local, high-quality vectors     |
| Vector Math  | `scikit-learn` (cosine_sim)      | Semantic similarity calculation       |
| LLM Brain    | Groq API (Llama 3.1/3.3/4)      | Ultra-fast inference, free tier       |
| Orchestration| `langchain-groq`                 | Clean LLM integration                 |
| Security     | `python-dotenv`                  | Secure API key management             |

---

## ⚡ Quick Start

### Step 1: Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/resume-analyzer.git
cd resume-analyzer
```

### Step 2: Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Get your FREE Groq API Key

1. Go to 👉 **[console.groq.com](https://console.groq.com)**
2. Sign up (it's completely free)
3. Navigate to **API Keys** in the left sidebar
4. Click **"Create API Key"**
5. Copy the key (starts with `gsk_...`)

### Step 5: Set up your API key securely

```bash
# Copy the example file
cp .env.example .env
```

Now open `.env` and replace the placeholder:
```
GROQ_API_KEY=gsk_your_actual_key_here
```

> ✅ The `.env` file is in `.gitignore` — it will NEVER be uploaded to GitHub.

### Step 6: Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** 🎉

---

## 🔐 API Key — 3 Ways to Use It

| Method               | When to Use                         | How                                    |
|---------------------|-------------------------------------|----------------------------------------|
| `.env` file         | Local development (recommended)     | Add key to `.env`, auto-loaded by app  |
| Sidebar input       | Quick testing / sharing with others | Type/paste key in the sidebar field    |
| Streamlit Cloud     | Cloud deployment                    | Add via Streamlit dashboard → Secrets  |

The app checks for the key in this order:
1. `.env` file (via `python-dotenv`)
2. Sidebar manual input (fallback)

---

## 📁 File Structure

```
resume-analyzer/
│
├── app.py                    # Main Streamlit UI
├── utils.py                  # Backend: PDF → Embed → LLM pipeline
├── requirements.txt          # All dependencies
├── .env.example              # Template — copy to .env and add your key
├── .env                      # ← YOUR SECRET KEY (never committed)
├── .gitignore                # Ignores .env, __pycache__, etc.
├── README.md                 # This file
│
└── .streamlit/
    ├── config.toml           # Dark theme + server settings
    └── secrets.toml          # For Streamlit Cloud deployment
```

---

## 🚀 Deploy to Streamlit Cloud (Free)

1. Push your code to GitHub *(`.env` is ignored — safe!)*
2. Go to **[share.streamlit.io](https://share.streamlit.io)**
3. Connect your GitHub repo
4. In **Advanced Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   ```
5. Deploy! Your app gets a public URL instantly.

---

## 📊 How the Score Works

| Score Range | Meaning                                              |
|------------|------------------------------------------------------|
| 75% – 100% | 🟢 Excellent Match — Apply with confidence           |
| 55% – 74%  | 🔵 Good Match — Minor tweaks recommended             |
| 35% – 54%  | 🟡 Average Match — Significant improvements needed   |
| 0% – 34%   | 🔴 Low Match — Resume needs major tailoring          |

The score uses **cosine similarity** between 384-dimensional sentence embeddings:

```
score = (A · B) / (||A|| × ||B||)
```

This is **semantic** similarity — not just keyword matching. It understands meaning.

---

## 🤖 Available AI Models

| Model                    | Speed   | Quality | Best For             |
|--------------------------|---------|---------|----------------------|
| Llama 3.1 — 8B Instant  | ⚡⚡⚡  | ⭐⭐⭐  | Quick feedback        |
| Llama 3.3 — 70B          | ⚡⚡    | ⭐⭐⭐⭐⭐ | Deep analysis       |
| Llama 4 Scout            | ⚡⚡⚡  | ⭐⭐⭐⭐ | Balanced             |

---

## 🧑‍💻 Author

Built with ❤️ as an AI portfolio project demonstrating:
- Vector embeddings & semantic search
- LLM prompt engineering
- Secure API key management
- Production-ready Streamlit UI

---

## 📄 License

MIT License — free to use, modify, and distribute.