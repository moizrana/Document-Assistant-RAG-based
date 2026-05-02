"""
Intelligent Document Assistant — Main Streamlit Application
A RAG-powered AI assistant with hybrid search, reranking, and premium UI.
"""

import streamlit as st
import logging
import time
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

from src.config import settings
from src.document_loader import load_document, SUPPORTED_EXTENSIONS
from src.chunker import chunk_documents
from src.embedder import embed_texts, embed_query, get_embedding_dimension
from src.vector_store import FAISSIndex, BM25Index
from src.retriever import hybrid_retrieve
from src.generator import generate_answer
from src.query_expander import expand_query
from src.cache import SemanticCache
from src.guardrails import check_input, check_output
from src.evaluator import run_evaluation

# =====================================================================
# Page Configuration
# =====================================================================
st.set_page_config(
    page_title="Intelligent Document Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# Custom CSS for Premium UI
# =====================================================================
st.markdown("""
<style>
    /* ---- Import Google Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ---- Global ---- */
    html, body, .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ---- Header ---- */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #6C63FF 0%, #A855F7 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #9CA3AF;
        font-size: 0.95rem;
    }

    /* ---- Chat Messages ---- */
    .stChatMessage {
        border-radius: 16px !important;
        margin-bottom: 0.5rem !important;
        border: 1px solid rgba(108, 99, 255, 0.1) !important;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(108, 99, 255, 0.15);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #6C63FF;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* ---- Source Cards ---- */
    .source-card {
        background: rgba(108, 99, 255, 0.08);
        border: 1px solid rgba(108, 99, 255, 0.2);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.82rem;
        transition: all 0.2s ease;
    }
    .source-card:hover {
        border-color: #6C63FF;
        background: rgba(108, 99, 255, 0.14);
    }
    .source-card .source-title {
        color: #A78BFA;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .source-card .source-text {
        color: #D1D5DB;
        font-size: 0.78rem;
        margin-top: 0.3rem;
        line-height: 1.4;
    }

    /* ---- Metric Cards ---- */
    .metric-row {
        display: flex;
        gap: 0.75rem;
        margin: 0.5rem 0;
    }
    .metric-card {
        flex: 1;
        background: rgba(108, 99, 255, 0.06);
        border: 1px solid rgba(108, 99, 255, 0.15);
        border-radius: 10px;
        padding: 0.6rem 0.8rem;
        text-align: center;
    }
    .metric-card .metric-label {
        color: #9CA3AF;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .metric-card .metric-value {
        color: #6C63FF;
        font-size: 1.2rem;
        font-weight: 700;
    }

    /* ---- Upload Zone ---- */
    .upload-zone {
        background: rgba(108, 99, 255, 0.04);
        border: 2px dashed rgba(108, 99, 255, 0.25);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
    }
    .upload-zone:hover {
        border-color: #6C63FF;
        background: rgba(108, 99, 255, 0.08);
    }

    /* ---- Status Badge ---- */
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
    }
    .status-ready {
        background: rgba(34, 197, 94, 0.15);
        color: #22C55E;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    .status-empty {
        background: rgba(234, 179, 8, 0.15);
        color: #EAB308;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }

    /* ---- Animated loading ---- */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .thinking {
        animation: pulse 1.5s ease-in-out infinite;
        color: #A78BFA;
    }

    /* ---- Divider ---- */
    hr {
        border: none;
        border-top: 1px solid rgba(108, 99, 255, 0.12);
        margin: 1rem 0;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =====================================================================
# Session State Initialization
# =====================================================================
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        "messages": [],
        "faiss_index": None,
        "bm25_index": None,
        "semantic_cache": SemanticCache(),
        "documents_loaded": 0,
        "chunks_count": 0,
        "uploaded_files": [],
        "use_query_expansion": False,
        "use_evaluation": False,
        "use_guardrails": True,
        "use_cache": True,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# =====================================================================
# Sidebar
# =====================================================================
def render_sidebar():
    """Render the sidebar with upload, settings, and status."""
    with st.sidebar:
        # Logo / Title
        st.markdown("""
        <div style="text-align:center; padding: 0.5rem 0 1rem 0;">
            <span style="font-size: 2.5rem;">🧠</span>
            <h2 style="margin: 0.2rem 0; background: linear-gradient(135deg, #6C63FF, #A855F7);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            font-size: 1.3rem;">DocuMind AI</h2>
            <p style="color: #6B7280; font-size: 0.75rem; margin:0;">Intelligent Document Assistant</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # --- Document Upload ---
        st.markdown("### 📄 Upload Documents")
        uploaded_files = st.file_uploader(
            "Drop your files here",
            type=["pdf", "html", "htm", "txt", "md"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="file_uploader"
        )

        if uploaded_files:
            # Check for any new files that haven't been processed yet
            new_files = [f for f in uploaded_files if f.name not in st.session_state.uploaded_files]
            if new_files:
                process_documents(new_files)



        # --- Settings ---
        st.markdown("### ⚙️ Pipeline Settings")
        st.session_state.use_guardrails = True  # Always ON in backend, hidden from UI
        st.session_state.use_evaluation = st.toggle("📈 Evaluation Metrics", value=True, help="Score answer quality")
        st.session_state.use_cache = st.toggle("⚡ Semantic Cache", value=True, help="Cache similar queries")
        st.session_state.use_query_expansion = st.toggle("🔄 Query Expansion", value=False, help="Generate alternative phrasings")

        st.markdown("---")

        # --- Clear Chat ---
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# =====================================================================
# Document Processing
# =====================================================================
def process_documents(uploaded_files):
    """Process uploaded files: load → chunk → embed → index."""
    progress = st.sidebar.progress(0, text="Processing documents...")

    all_chunks = []
    total_docs = 0

    for i, uploaded_file in enumerate(uploaded_files):
        try:
            progress.progress(
                (i) / len(uploaded_files),
                text=f"Loading {uploaded_file.name}..."
            )

            # Load document
            file_bytes = uploaded_file.read()
            documents = load_document(file_bytes, uploaded_file.name)
            total_docs += len(documents)

            # Chunk
            chunks = chunk_documents(
                documents,
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP
            )
            all_chunks.extend(chunks)

        except Exception as e:
            st.sidebar.error(f"❌ Error processing {uploaded_file.name}: {e}")
            logger.error(f"Error processing {uploaded_file.name}: {e}")

    if not all_chunks:
        progress.empty()
        st.sidebar.error("No content could be extracted from the uploaded files.")
        return

    # Embed chunks
    progress.progress(0.7, text="Uploading document...")
    texts = [chunk.content for chunk in all_chunks]
    embeddings = embed_texts(texts, settings.EMBEDDING_MODEL)

    # Build FAISS index
    progress.progress(0.85, text="Building search index...")
    dim = embeddings.shape[1]

    if st.session_state.faiss_index is None:
        st.session_state.faiss_index = FAISSIndex(dim)
    st.session_state.faiss_index.add(embeddings, all_chunks)

    # Build BM25 index
    if st.session_state.bm25_index is None:
        st.session_state.bm25_index = BM25Index()
    st.session_state.bm25_index.add(all_chunks)

    # Save indices
    st.session_state.faiss_index.save(settings.STORAGE_DIR)
    st.session_state.bm25_index.save(settings.STORAGE_DIR)

    # Update counts
    st.session_state.documents_loaded += total_docs
    st.session_state.chunks_count = st.session_state.faiss_index.size
    st.session_state.uploaded_files.extend([f.name for f in uploaded_files])

    progress.progress(1.0, text="✅ Done!")
    time.sleep(0.5)
    progress.empty()

    st.sidebar.success(f"Processed {total_docs} document(s) → {len(all_chunks)} chunks")
    st.toast(f"📚 {len(all_chunks)} chunks indexed!", icon="✅")
    st.rerun()


# =====================================================================
# RAG Pipeline
# =====================================================================
def run_rag_pipeline(query: str, stream_callback=None) -> dict:
    """
    Execute the full RAG pipeline:
    Guardrails → Cache → Query Expansion → Hybrid Retrieval → Reranking → Generation → Evaluation
    """
    pipeline_log = {"steps": []}

    # Step 1: Guardrails
    if st.session_state.use_guardrails:
        guard_result = check_input(query)
        if not guard_result.is_safe:
            return {
                "answer": f"⚠️ {guard_result.reason}",
                "sources": [],
                "model_used": "guardrails",
                "eval_scores": None,
                "from_cache": False,
                "pipeline_log": pipeline_log
            }
        query = guard_result.sanitized_query
        pipeline_log["steps"].append("✅ Guardrails passed")

    # Step 2: Generate query embedding
    query_emb = embed_query(query, settings.EMBEDDING_MODEL)

    # Step 3: Semantic Cache check
    if st.session_state.use_cache:
        cached = st.session_state.semantic_cache.get(query, query_emb)
        if cached:
            pipeline_log["steps"].append("⚡ Cache hit!")
            cached["from_cache"] = True
            cached["pipeline_log"] = pipeline_log
            return cached

    # Step 4: Query Expansion (optional)
    queries = [query]
    if st.session_state.use_query_expansion:
        queries = expand_query(query)
        pipeline_log["steps"].append(f"🔄 Expanded to {len(queries)} queries")

    # Step 5: Hybrid Retrieval (for each query variant)
    all_results = []
    for q in queries:
        q_emb = embed_query(q, settings.EMBEDDING_MODEL) if q != query else query_emb
        results = hybrid_retrieve(
            query=q,
            query_embedding=q_emb,
            faiss_index=st.session_state.faiss_index,
            bm25_index=st.session_state.bm25_index,
            top_k=settings.TOP_K_RETRIEVAL,
            rerank_top_n=settings.RERANK_TOP_N,
            reranker_model=settings.RERANKER_MODEL
        )
        all_results.extend(results)
    pipeline_log["steps"].append(f"🔍 Retrieved {len(all_results)} chunks")

    # Deduplicate by chunk_id
    seen = set()
    unique_results = []
    for r in all_results:
        if r.chunk_id not in seen:
            seen.add(r.chunk_id)
            unique_results.append(r)

    # Take top N after dedup
    final_results = unique_results[:settings.RERANK_TOP_N]

    if not final_results:
        return {
            "answer": "I couldn't find any relevant information in the uploaded documents for your question. Please try rephrasing or upload more relevant documents.",
            "sources": [],
            "model_used": "none",
            "eval_scores": None,
            "from_cache": False,
            "pipeline_log": pipeline_log
        }

    # Step 6: LLM Generation
    pipeline_log["steps"].append("🤖 Generating answer...")
    gen_result = generate_answer(query, final_results, stream_callback=stream_callback)

    # Sanitize output
    if st.session_state.use_guardrails:
        gen_result.answer = check_output(gen_result.answer)

    pipeline_log["steps"].append(f"✅ Generated by {gen_result.model_used}")

    # Step 7: Evaluation (optional)
    eval_scores = None
    if st.session_state.use_evaluation and gen_result.answer:
        context_texts = [r.content for r in final_results]
        eval_scores = run_evaluation(query, gen_result.answer, context_texts)
        pipeline_log["steps"].append("📈 Evaluation complete")

    response = {
        "answer": gen_result.answer,
        "sources": gen_result.sources,
        "model_used": gen_result.model_used,
        "latency_ms": gen_result.latency_ms,
        "eval_scores": eval_scores,
        "from_cache": False,
        "pipeline_log": pipeline_log,
    }

    # Store in cache
    if st.session_state.use_cache:
        st.session_state.semantic_cache.put(query, query_emb, response)

    return response


# =====================================================================
# Main Chat Interface
# =====================================================================

# Render sidebar now that all functions are defined
render_sidebar()

# Header
st.markdown("""
<div class="main-header">
    <h1>🧠 Intelligent Document Assistant</h1>
    <p>Upload documents and ask questions — powered by RAG with hybrid search & reranking</p>
</div>
""", unsafe_allow_html=True)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑" if message["role"] == "user" else "🧠"):
        st.markdown(message["content"])

        # Show sources for assistant messages
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("📚 Sources", expanded=False):
                for src in message["sources"]:
                    source_name = src.get("source", "Unknown")
                    page = src.get("page", "")
                    page_str = f" | Page {page}" if page else ""
                    content_preview = src.get("content", "")[:200] + "..."
                    st.markdown(
                        f'<div class="source-card">'
                        f'<div class="source-title">📄 {source_name}{page_str}</div>'
                        f'<div class="source-text">{content_preview}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        # Show evaluation scores
        if message["role"] == "assistant" and message.get("eval_scores"):
            scores = message["eval_scores"]
            with st.expander("📈 Evaluation Metrics", expanded=False):
                cols = st.columns(3)
                for col, (name, val) in zip(cols, scores.items()):
                    label = name.replace("_", " ").title()
                    color = "#22C55E" if val >= 0.7 else "#EAB308" if val >= 0.4 else "#EF4444"
                    col.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">{label}</div>'
                        f'<div class="metric-value" style="color:{color}">{val:.2f}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        # Show model info
        if message["role"] == "assistant" and message.get("model_used"):
            cache_badge = " ⚡ cached" if message.get("from_cache") else ""
            st.caption(f"🤖 {message['model_used']}{cache_badge}")

# Chat input
if prompt := st.chat_input("Ask a question about your documents...", key="chat_input"):
    # Check if documents are loaded
    if not st.session_state.faiss_index or st.session_state.faiss_index.size == 0:
        st.warning("📄 Please upload and process documents first using the sidebar.")
    else:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant", avatar="🧠"):
            # Check for simple greetings and identity questions
            cleaned_prompt = prompt.strip().lower()
            is_small_talk = False
            
            if cleaned_prompt.rstrip("!?.,") in ["hey", "hello", "hi", "how are you", "good morning", "good evening", "hi there", "hello there", "sup", "what's up"]:
                is_small_talk = True
            if "who are you" in cleaned_prompt or "what are you" in cleaned_prompt or "your name" in cleaned_prompt:
                is_small_talk = True
            
            if is_small_talk:
                response = {
                    "answer": "Hello! I am DocuMind AI, your Intelligent Document Assistant. I am here to help you extract and understand information from your uploaded documents. How can I help you today?",
                    "sources": [],
                    "model_used": "system",
                    "eval_scores": None,
                    "from_cache": False,
                    "pipeline_log": {"steps": ["✅ Greeting/Identity detected"]}
                }
            else:
                placeholder = st.empty()
                streamed_text = {"text": ""}
                
                def stream_callback(chunk):
                    streamed_text["text"] += chunk
                    placeholder.markdown(streamed_text["text"] + "▌")
                    
                with st.spinner("Thinking..."):
                    response = run_rag_pipeline(prompt, stream_callback=stream_callback)

                placeholder.markdown(response["answer"])

            # Show sources
            if response.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    for src in response["sources"]:
                        source_name = src.get("source", "Unknown")
                        page = src.get("page", "")
                        page_str = f" | Page {page}" if page else ""
                        content_preview = src.get("content", "")[:200] + "..."
                        st.markdown(
                            f'<div class="source-card">'
                            f'<div class="source-title">📄 {source_name}{page_str}</div>'
                            f'<div class="source-text">{content_preview}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            # Show eval scores
            if response.get("eval_scores"):
                scores = response["eval_scores"]
                with st.expander("📈 Evaluation Metrics", expanded=False):
                    cols = st.columns(3)
                    for col, (name, val) in zip(cols, scores.items()):
                        label = name.replace("_", " ").title()
                        color = "#22C55E" if val >= 0.7 else "#EAB308" if val >= 0.4 else "#EF4444"
                        col.markdown(
                            f'<div class="metric-card">'
                            f'<div class="metric-label">{label}</div>'
                            f'<div class="metric-value" style="color:{color}">{val:.2f}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

            # Show model badge
            cache_badge = " ⚡ cached" if response.get("from_cache") else ""
            st.caption(f"🤖 {response.get('model_used', 'unknown')}{cache_badge}")

        # Save to chat history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "sources": response.get("sources"),
            "model_used": response.get("model_used"),
            "eval_scores": response.get("eval_scores"),
            "from_cache": response.get("from_cache"),
        })


# =====================================================================
# Welcome State (when no messages)
# =====================================================================
if not st.session_state.messages:
    st.markdown("---")
    cols = st.columns(3)

    features = [
        ("🔍", "Hybrid Search", "FAISS dense + BM25 sparse search with Reciprocal Rank Fusion"),
        ("🎯", "Cross-Encoder Reranking", "Precision reranking using transformer cross-encoders"),
        ("🛡️", "Guardrails", "Prompt injection protection and output validation"),
    ]

    for col, (icon, title, desc) in zip(cols, features):
        col.markdown(
            f"""<div style="text-align:center; padding: 1.2rem; background: rgba(108,99,255,0.05);
            border: 1px solid rgba(108,99,255,0.12); border-radius: 14px;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
            <div style="color: #E0E0E0; font-weight: 600; font-size: 0.95rem;">{title}</div>
            <div style="color: #9CA3AF; font-size: 0.78rem; margin-top: 0.3rem;">{desc}</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.markdown("")
    cols2 = st.columns(3)
    features2 = [
        ("⚡", "Semantic Caching", "Instant responses for similar questions"),
        ("🔄", "Query Expansion", "Multi-query retrieval for better recall"),
        ("📈", "Evaluation Metrics", "Faithfulness, relevancy, and context scoring"),
    ]
    for col, (icon, title, desc) in zip(cols2, features2):
        col.markdown(
            f"""<div style="text-align:center; padding: 1.2rem; background: rgba(108,99,255,0.05);
            border: 1px solid rgba(108,99,255,0.12); border-radius: 14px;">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
            <div style="color: #E0E0E0; font-weight: 600; font-size: 0.95rem;">{title}</div>
            <div style="color: #9CA3AF; font-size: 0.78rem; margin-top: 0.3rem;">{desc}</div>
            </div>""",
            unsafe_allow_html=True
        )
