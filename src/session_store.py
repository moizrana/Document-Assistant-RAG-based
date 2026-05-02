import os
import pickle
import time
from typing import Optional, Dict

from src.cache import SemanticCache
from src.embedder import get_embedding_dimension
from src.vector_store import FAISSIndex, BM25Index
from src.config import settings


class UserSession:
    """
    Serializable session object.

    Note: we persist sessions to Redis on upload/clear only to avoid slowing
    down token streaming on /api/ask.
    """

    def __init__(self):
        self.faiss_index = FAISSIndex(get_embedding_dimension(settings.EMBEDDING_MODEL))
        self.bm25_index = BM25Index()
        self.semantic_cache = SemanticCache()
        self.documents_loaded = 0
        self.uploaded_files = []
        self.last_accessed = time.time()

    def to_bytes(self) -> bytes:
        payload = {
            "faiss": self.faiss_index.to_state(),
            "bm25": self.bm25_index.to_state(),
            "semantic_cache": self.semantic_cache,
            "documents_loaded": self.documents_loaded,
            "uploaded_files": self.uploaded_files,
            "last_accessed": self.last_accessed,
        }
        return pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_bytes(cls, data: bytes) -> "UserSession":
        payload = pickle.loads(data)
        obj = cls()
        obj.faiss_index = FAISSIndex.from_state(payload["faiss"])
        obj.bm25_index = BM25Index.from_state(payload["bm25"])
        obj.semantic_cache = payload.get("semantic_cache") or SemanticCache()
        obj.documents_loaded = int(payload.get("documents_loaded", 0))
        obj.uploaded_files = list(payload.get("uploaded_files", []))
        obj.last_accessed = float(payload.get("last_accessed", time.time()))
        return obj


class RedisSessionStore:
    def __init__(self, redis_client, namespace: str = "documind:sess", ttl_seconds: int = 60 * 60 * 24):
        self.redis = redis_client
        self.namespace = namespace
        self.ttl_seconds = ttl_seconds
        self.local_cache: Dict[str, UserSession] = {}

    def _key(self, session_id: str) -> str:
        return f"{self.namespace}:{session_id}"

    async def get(self, session_id: str) -> UserSession:
        if session_id in self.local_cache:
            sess = self.local_cache[session_id]
            sess.last_accessed = time.time()
            return sess

        raw = await self.redis.get(self._key(session_id))
        if raw:
            sess = UserSession.from_bytes(raw)
        else:
            sess = UserSession()

        sess.last_accessed = time.time()
        self.local_cache[session_id] = sess
        return sess

    async def save(self, session_id: str) -> None:
        sess = self.local_cache.get(session_id)
        if not sess:
            return
        sess.last_accessed = time.time()
        await self.redis.set(self._key(session_id), sess.to_bytes(), ex=self.ttl_seconds)

    async def delete(self, session_id: str) -> None:
        self.local_cache.pop(session_id, None)
        await self.redis.delete(self._key(session_id))


class InMemorySessionStore:
    """
    Lightweight fallback when Redis isn't available (dev-safe).
    Keeps behavior identical to the old in-memory dict.
    """

    def __init__(self):
        self.local_cache: Dict[str, UserSession] = {}

    async def get(self, session_id: str) -> UserSession:
        if session_id not in self.local_cache:
            self.local_cache[session_id] = UserSession()
        sess = self.local_cache[session_id]
        sess.last_accessed = time.time()
        return sess

    async def save(self, session_id: str) -> None:
        # No persistence in fallback mode.
        return None

    async def delete(self, session_id: str) -> None:
        self.local_cache.pop(session_id, None)

