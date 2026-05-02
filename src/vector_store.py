"""
Vector Store module.
Manages FAISS dense index and BM25 sparse index for hybrid search.
Supports saving/loading indices to/from disk for persistence.
"""

import logging
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single search result from the vector store."""
    chunk_id: str
    content: str
    metadata: dict
    score: float


class FAISSIndex:
    """FAISS-based dense vector index using inner product (for normalized embeddings)."""

    def __init__(self, dimension: int):
        import faiss
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.chunks = []  # Parallel list of chunk data
        logger.info(f"Initialized FAISS index with dimension {dimension}")

    def add(self, embeddings: np.ndarray, chunks: list):
        """
        Add embeddings and their corresponding chunks to the index.
        
        Args:
            embeddings: NumPy array of shape (n, dimension).
            chunks: List of Chunk objects (parallel to embeddings).
        """
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} vectors to FAISS. Total: {self.index.ntotal}")

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> list[SearchResult]:
        """
        Search for the top-k most similar vectors.
        
        Args:
            query_embedding: NumPy array of shape (1, dimension).
            top_k: Number of results to return.
        
        Returns:
            List of SearchResult objects sorted by score (descending).
        """
        if self.index.ntotal == 0:
            return []

        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks) and idx >= 0:
                chunk = self.chunks[idx]
                results.append(SearchResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    score=float(score)
                ))
        return results

    @property
    def size(self) -> int:
        return self.index.ntotal

    def save(self, directory: str):
        """Save FAISS index and chunk metadata to disk."""
        import faiss
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "faiss_index.bin"))
        with open(os.path.join(directory, "faiss_chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(f"Saved FAISS index ({self.index.ntotal} vectors) to {directory}")

    @classmethod
    def load(cls, directory: str) -> Optional["FAISSIndex"]:
        """Load FAISS index and chunk metadata from disk."""
        import faiss
        index_path = os.path.join(directory, "faiss_index.bin")
        chunks_path = os.path.join(directory, "faiss_chunks.pkl")

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            return None

        try:
            index = faiss.read_index(index_path)
            with open(chunks_path, "rb") as f:
                chunks = pickle.load(f)

            instance = cls.__new__(cls)
            instance.dimension = index.d
            instance.index = index
            instance.chunks = chunks
            logger.info(f"Loaded FAISS index ({index.ntotal} vectors) from {directory}")
            return instance
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return None


class BM25Index:
    """BM25-based sparse keyword index using rank_bm25."""

    def __init__(self):
        self.bm25 = None
        self.chunks = []
        self.tokenized_corpus = []

    def add(self, chunks: list):
        """
        Build/rebuild the BM25 index from chunks.
        
        Args:
            chunks: List of Chunk objects.
        """
        from rank_bm25 import BM25Okapi

        self.chunks.extend(chunks)
        # Re-tokenize the entire corpus (BM25 requires full rebuild)
        self.tokenized_corpus = [
            chunk.content.lower().split() for chunk in self.chunks
        ]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        logger.info(f"Built BM25 index with {len(self.chunks)} documents")

    def search(self, query: str, top_k: int = 20) -> list[SearchResult]:
        """
        Search using BM25 keyword matching.
        
        Args:
            query: Search query string.
            top_k: Number of results to return.
        
        Returns:
            List of SearchResult objects sorted by BM25 score (descending).
        """
        if not self.bm25 or not self.chunks:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices by score
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include results with positive scores
                chunk = self.chunks[idx]
                results.append(SearchResult(
                    chunk_id=chunk.chunk_id,
                    content=chunk.content,
                    metadata=chunk.metadata,
                    score=float(scores[idx])
                ))
        return results

    @property
    def size(self) -> int:
        return len(self.chunks)

    def save(self, directory: str):
        """Save BM25 index data to disk."""
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "bm25_data.pkl"), "wb") as f:
            pickle.dump({
                "chunks": self.chunks,
                "tokenized_corpus": self.tokenized_corpus
            }, f)
        logger.info(f"Saved BM25 index ({len(self.chunks)} documents) to {directory}")

    @classmethod
    def load(cls, directory: str) -> Optional["BM25Index"]:
        """Load BM25 index data from disk."""
        from rank_bm25 import BM25Okapi

        data_path = os.path.join(directory, "bm25_data.pkl")
        if not os.path.exists(data_path):
            return None

        try:
            with open(data_path, "rb") as f:
                data = pickle.load(f)

            instance = cls()
            instance.chunks = data["chunks"]
            instance.tokenized_corpus = data["tokenized_corpus"]
            if instance.tokenized_corpus:
                instance.bm25 = BM25Okapi(instance.tokenized_corpus)
            logger.info(f"Loaded BM25 index ({len(instance.chunks)} documents) from {directory}")
            return instance
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return None
