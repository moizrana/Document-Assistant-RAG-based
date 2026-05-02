"""
Configuration module for the RAG Intelligent Document Assistant.
Loads settings from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""

    # --- LLM API Keys ---
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")

    # --- Model Configuration ---
    PRIMARY_MODEL: str = os.getenv("PRIMARY_MODEL", "gemini-3.1-flash-lite-preview")
    FALLBACK_MODEL_1: str = os.getenv("FALLBACK_MODEL_1", "meta-llama/llama-3.3-70b-instruct:free")
    FALLBACK_MODEL_2: str = os.getenv("FALLBACK_MODEL_2", "qwen/qwen3-next-80b-a3b-instruct:free")
    FALLBACK_MODEL_3: str = os.getenv("FALLBACK_MODEL_3", "google/gemini-3.1-flash-lite-preview")
    FALLBACK_MODEL_4: str = os.getenv("FALLBACK_MODEL_4", "google/gemma-3-12b-it:free")

    # --- Embedding & Reranking (local models) ---
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

    # --- RAG Pipeline ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "10"))
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", "5"))

    # --- Semantic Cache ---
    CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))
    CACHE_MAX_SIZE: int = int(os.getenv("CACHE_MAX_SIZE", "100"))

    # --- Paths ---
    STORAGE_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "storage")
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    @classmethod
    def ensure_dirs(cls):
        """Create required directories if they don't exist."""
        os.makedirs(cls.STORAGE_DIR, exist_ok=True)
        os.makedirs(cls.DATA_DIR, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
