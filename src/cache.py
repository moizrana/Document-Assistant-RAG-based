"""
Semantic Cache module.
Caches (query_embedding, response) pairs and returns cached responses
for semantically similar queries (cosine similarity > threshold).
Reduces LLM API calls and latency for repeated or similar questions.
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single entry in the semantic cache."""
    query: str
    query_embedding: np.ndarray
    response: dict  # Full GenerationResult as dict
    timestamp: float
    hit_count: int = 0


class SemanticCache:
    """
    In-memory semantic cache using cosine similarity.
    
    When a new query arrives, its embedding is compared against all cached
    query embeddings. If the similarity exceeds the threshold, the cached
    response is returned immediately, skipping the entire retrieval + generation pipeline.
    """

    def __init__(
        self,
        similarity_threshold: float = None,
        max_size: int = None
    ):
        self.threshold = similarity_threshold or settings.CACHE_SIMILARITY_THRESHOLD
        self.max_size = max_size or settings.CACHE_MAX_SIZE
        self.cache: list[CacheEntry] = []

    def get(self, query: str, query_embedding: np.ndarray) -> Optional[dict]:
        """
        Look up a cached response for a semantically similar query.
        
        Args:
            query: The user's query text.
            query_embedding: The query's embedding vector.
        
        Returns:
            Cached response dict if a similar query is found, None otherwise.
        """
        if not self.cache:
            return None

        # Compute cosine similarity with all cached embeddings
        cached_embeddings = np.vstack([entry.query_embedding for entry in self.cache])
        similarities = np.dot(cached_embeddings, query_embedding.T).flatten()

        max_idx = int(np.argmax(similarities))
        max_sim = float(similarities[max_idx])

        if max_sim >= self.threshold:
            entry = self.cache[max_idx]
            entry.hit_count += 1
            logger.info(
                f"Cache HIT (similarity={max_sim:.4f}, hits={entry.hit_count}): "
                f"'{query}' matched '{entry.query}'"
            )
            return entry.response

        logger.debug(f"Cache MISS (best similarity={max_sim:.4f})")
        return None

    def put(self, query: str, query_embedding: np.ndarray, response: dict):
        """
        Store a query-response pair in the cache.
        
        Args:
            query: The user's query text.
            query_embedding: The query's embedding vector.
            response: The full response dict to cache.
        """
        # Evict oldest entry if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.pop(0)
            logger.info("Cache evicted oldest entry (max size reached)")

        self.cache.append(CacheEntry(
            query=query,
            query_embedding=query_embedding,
            response=response,
            timestamp=time.time()
        ))
        logger.info(f"Cached response for: '{query}' (cache size: {len(self.cache)})")

    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()
        logger.info("Cache cleared")

    @property
    def size(self) -> int:
        return len(self.cache)

    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        total_hits = sum(entry.hit_count for entry in self.cache)
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "threshold": self.threshold,
            "total_hits": total_hits,
        }
