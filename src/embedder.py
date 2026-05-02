"""
Embedding module.
Uses SentenceTransformers to generate dense vector embeddings locally.
The model is loaded lazily as a singleton to avoid redundant loading.
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level singleton
_model = None
_model_name = None


def _get_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load the SentenceTransformer model as a singleton."""
    global _model, _model_name
    if _model is None or _model_name != model_name:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading embedding model: {model_name}")
        _model = SentenceTransformer(model_name)
        _model_name = model_name
        logger.info(f"Embedding model loaded. Dimension: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_texts(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generate embeddings for a list of texts.
    
    Args:
        texts: List of text strings to embed.
        model_name: Name of the SentenceTransformer model.
    
    Returns:
        NumPy array of shape (len(texts), embedding_dim).
    """
    model = _get_model(model_name)
    embeddings = model.encode(
        texts,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,  # Normalize for cosine similarity via inner product
        batch_size=32
    )
    logger.info(f"Generated embeddings for {len(texts)} texts, shape: {embeddings.shape}")
    return np.array(embeddings, dtype=np.float32)


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Generate embedding for a single query string.
    
    Args:
        query: The query text to embed.
        model_name: Name of the SentenceTransformer model.
    
    Returns:
        NumPy array of shape (1, embedding_dim).
    """
    model = _get_model(model_name)
    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )
    return np.array(embedding, dtype=np.float32)


def get_embedding_dimension(model_name: str = "all-MiniLM-L6-v2") -> int:
    """Return the embedding dimension for the given model."""
    model = _get_model(model_name)
    return model.get_sentence_embedding_dimension()
