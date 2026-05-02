# DocuMind AI — Intelligent Document Assistant

An AI-powered document assistant that answers user queries using a custom knowledge base via **Retrieval-Augmented Generation (RAG)** architecture. Built with modern AI engineering best practices including hybrid search, cross-encoder reranking, and automatic LLM fallback.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)

---

## Architecture

```
User Query
    │
    ▼
┌──────────────────────────────────────────────────┐
│                 GUARDRAILS                        │
│     Prompt injection detection & sanitization     │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│              SEMANTIC CACHE CHECK                 │
│   Cosine similarity > 0.95 → return cached        │
└──────────────────────┬───────────────────────────┘
                       │ (cache miss)
                       ▼
┌──────────────────────────────────────────────────┐
│           QUERY EXPANSION (Optional)              │
│     LLM generates 2 alternative phrasings         │
└──────────────────────┬───────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
┌──────────────────┐     ┌──────────────────┐
│   FAISS Dense    │     │   BM25 Sparse    │
│  (Semantic)      │     │  (Keyword)       │
└────────┬─────────┘     └────────┬─────────┘
         │                        │
         └───────────┬────────────┘
                     ▼
┌──────────────────────────────────────────────────┐
│       RECIPROCAL RANK FUSION (RRF)               │
│    Merge dense + sparse results (k=60)            │
└──────────────────────┬───────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────┐
│         CROSS-ENCODER RERANKING                   │
│   ms-marco-MiniLM-L-6-v2 precision scoring        │
└──────────────────────┬───────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────┐
│           LLM GENERATION (with Fallback)          │
│                                                    │
│   Primary:  Gemini 1.5 Flash (Google API)         │
│   Fallback: Llama 3.3 70B (OpenRouter)            │
│   Fallback: Qwen3 80B A3B (OpenRouter)            │
│                                                    │
│   Strict system prompt for grounding              │
│   Temperature 0.1 for factual accuracy            │
└──────────────────────┬───────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────┐
│         EVALUATION (Optional)                     │
│   Faithfulness · Answer Relevancy · Context       │
└──────────────────────────────────────────────────┘
```

---

## Features

### Core
| Feature | Description |
|:---|:---|
| **Multi-format Ingestion** | PDF (PyMuPDF), HTML (BeautifulSoup), TXT with encoding detection |
| **Smart Chunking** | Recursive character splitter with configurable size & overlap |
| **Local Embeddings** | `all-MiniLM-L6-v2` — fast, free, no API cost |
| **Vector Storage** | FAISS with disk persistence |
| **Semantic Search** | Dense vector similarity via FAISS |
| **Top-K Retrieval** | Configurable retrieval depth |
| **LLM Integration** | Gemini + OpenRouter with automatic fallback |
| **Context-Aware Generation** | Strict grounding prompts, source citation |
| **Prompt Engineering** | XML-delimited context, hallucination reduction |
| **Chat UI** | Premium Streamlit interface with dark theme |

### Bonus
| Feature | Description |
|:---|:---|
| **Hybrid Search** | FAISS dense + BM25 sparse with Reciprocal Rank Fusion |
| **Cross-Encoder Reranking** | `ms-marco-MiniLM-L-6-v2` for precision scoring |
| **Query Expansion** | LLM-generated alternative phrasings for better recall |
| **Semantic Caching** | Cosine similarity cache to skip LLM calls |
| **Guardrails** | Regex-based prompt injection detection + output sanitization |
| **Evaluation Metrics** | RAGAS-style: Faithfulness, Answer Relevancy, Context Relevancy |
| **API Fallback** | Automatic failover: Gemini → Llama 3.3 → Qwen3 |
| **Error Handling** | Comprehensive logging and graceful degradation |
| **Config Management** | Environment-based configuration via `.env` |

---

## Quick Start

### Prerequisites
- Python 3.10+
- Gemini API key ([Get free key](https://aistudio.google.com/apikey))
- OpenRouter API key ([Get free key](https://openrouter.ai/keys))

### Installation

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd RAG-Project

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Configure environment
# Create a .env file with your API keys:
# GEMINI_API_KEY=your_gemini_key
# OPENROUTER_API_KEY=your_openrouter_key

# 6. Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🗂️ Project Structure

```
RAG-Project/
├── app.py                    # Main Streamlit application
├── .streamlit/
│   └── config.toml           # UI theme configuration
├── src/
│   ├── config.py             # Environment & settings
│   ├── document_loader.py    # PDF/HTML/TXT loaders
│   ├── chunker.py            # Recursive text splitting
│   ├── embedder.py           # SentenceTransformer embeddings
│   ├── vector_store.py       # FAISS + BM25 indices
│   ├── retriever.py          # Hybrid search + RRF + reranking
│   ├── generator.py          # LLM generation with fallback
│   ├── query_expander.py     # Multi-query expansion
│   ├── cache.py              # Semantic caching
│   ├── guardrails.py         # Prompt injection protection
│   └── evaluator.py          # RAGAS-style evaluation
├── data/                     # Sample documents
├── storage/                  # Persisted indices (auto-generated)
├── requirements.txt
├── .env                      # API keys (not committed)
└── .gitignore
```

---

## ⚙️ Configuration

All settings are configurable via the `.env` file:

| Variable | Default | Description |
|:---|:---|:---|
| `GEMINI_API_KEY` | — | Google AI Studio API key |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `PRIMARY_MODEL` | `gemini-1.5-flash` | Primary LLM model |
| `FALLBACK_MODEL_1` | `meta-llama/llama-3.3-70b-instruct:free` | First fallback |
| `FALLBACK_MODEL_2` | `qwen/qwen3-next-80b-a3b-instruct:free` | Second fallback |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `10` | Candidates before reranking |
| `RERANK_TOP_N` | `5` | Final results after reranking |

---

## 🧪 Technical Details

### Hybrid Search with RRF
Dense (FAISS) captures semantic similarity while BM25 ensures keyword fidelity. Reciprocal Rank Fusion merges both ranked lists using `score = Σ 1/(k + rank)`, avoiding score normalization issues.

### Cross-Encoder Reranking
Unlike bi-encoders, cross-encoders process query-document pairs jointly, providing significantly more accurate relevance scores. Applied to top-K fusion results to select the best N chunks for generation.

### Hallucination Reduction
- Temperature set to 0.1 for deterministic outputs
- System prompt enforces strict grounding in provided context
- XML delimiters separate context from instructions
- Model instructed to refuse when information is insufficient

### API Fallback Chain
If the primary Gemini API hits rate limits, the system automatically falls back to OpenRouter free models (Llama 3.3 70B → Qwen3 80B) without any user intervention or error.

