"""
Retriever module.
Implements hybrid search (FAISS dense + BM25 sparse) with Reciprocal Rank Fusion (RRF)
and cross-encoder reranking for maximum retrieval precision.
"""

import logging
from typing import Optional

import numpy as np

from src.vector_store import FAISSIndex, BM25Index, SearchResult

logger = logging.getLogger(__name__)

# Module-level singleton for the reranker
_reranker = None
_reranker_name = None


def _get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Lazy-load the cross-encoder reranker model."""
    global _reranker, _reranker_name
    if _reranker is None or _reranker_name != model_name:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading reranker model: {model_name}")
        _reranker = CrossEncoder(model_name)
        _reranker_name = model_name
        logger.info("Reranker model loaded")
    return _reranker


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]],
    k: int = 60
) -> list[SearchResult]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).
    
    RRF score = sum(1 / (k + rank_i)) across all lists where the document appears.
    This is the industry-standard method for combining results from different
    retrieval systems without needing to normalize scores.
    
    Args:
        result_lists: List of ranked SearchResult lists from different retrievers.
        k: Constant to prevent high-ranked documents from dominating (default 60).
    
    Returns:
        Merged list of SearchResult objects sorted by fused score.
    """
    fused_scores = {}  # chunk_id -> cumulative RRF score
    chunk_map = {}     # chunk_id -> SearchResult object

    for result_list in result_lists:
        for rank, result in enumerate(result_list):
            if result.chunk_id not in fused_scores:
                fused_scores[result.chunk_id] = 0.0
                chunk_map[result.chunk_id] = result
            fused_scores[result.chunk_id] += 1.0 / (k + rank + 1)

    # Sort by fused score descending
    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    results = []
    for chunk_id in sorted_ids:
        result = chunk_map[chunk_id]
        results.append(SearchResult(
            chunk_id=result.chunk_id,
            content=result.content,
            metadata=result.metadata,
            score=fused_scores[chunk_id]
        ))

    return results


def rerank(
    query: str,
    candidates: list[SearchResult],
    top_n: int = 5,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> list[SearchResult]:
    """
    Rerank candidates using a cross-encoder model for precise relevance scoring.
    
    Unlike bi-encoders (used in FAISS), cross-encoders process the query and
    document together, enabling much more accurate relevance scoring at the
    cost of being slower (hence used only on a small candidate set).
    
    Args:
        query: The user's search query.
        candidates: List of SearchResult candidates from initial retrieval.
        top_n: Number of top results to return after reranking.
        model_name: Cross-encoder model to use.
    
    Returns:
        Reranked list of top-N SearchResult objects.
    """
    if not candidates:
        return []

    reranker = _get_reranker(model_name)

    # Create query-document pairs for the cross-encoder
    pairs = [[query, result.content] for result in candidates]
    scores = reranker.predict(pairs)

    # Attach scores and sort
    scored_results = list(zip(candidates, scores))
    scored_results.sort(key=lambda x: x[1], reverse=True)

    reranked = []
    for result, score in scored_results[:top_n]:
        reranked.append(SearchResult(
            chunk_id=result.chunk_id,
            content=result.content,
            metadata=result.metadata,
            score=float(score)
        ))

    logger.info(
        f"Reranked {len(candidates)} candidates → top {len(reranked)}. "
        f"Score range: {reranked[-1].score:.4f} to {reranked[0].score:.4f}"
    )
    return reranked


def hybrid_retrieve(
    query: str,
    query_embedding: np.ndarray,
    faiss_index: FAISSIndex,
    bm25_index: BM25Index,
    top_k: int = 20,
    rerank_top_n: int = 5,
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
) -> list[SearchResult]:
    """
    Full hybrid retrieval pipeline:
    1. Dense search via FAISS
    2. Sparse search via BM25
    3. Reciprocal Rank Fusion to merge results
    4. Cross-encoder reranking for final precision
    
    Args:
        query: The user's search query text.
        query_embedding: Pre-computed query embedding for FAISS.
        faiss_index: The FAISS dense index.
        bm25_index: The BM25 sparse index.
        top_k: Number of candidates from each retriever before fusion.
        rerank_top_n: Number of final results after reranking.
        reranker_model: Cross-encoder model name.
    
    Returns:
        Final list of top-N reranked SearchResult objects.
    """
    # Step 1: Dense search (FAISS)
    dense_results = faiss_index.search(query_embedding, top_k=top_k)
    logger.info(f"Dense search returned {len(dense_results)} results")

    # Step 2: Sparse search (BM25)
    sparse_results = bm25_index.search(query, top_k=top_k)
    logger.info(f"Sparse search returned {len(sparse_results)} results")

    # Step 3: Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion([dense_results, sparse_results])
    logger.info(f"RRF fusion produced {len(fused_results)} unique candidates")

    # Step 4: Cross-encoder reranking
    if fused_results:
        final_results = rerank(
            query=query,
            candidates=fused_results[:top_k],  # Rerank top-K fused candidates
            top_n=rerank_top_n,
            model_name=reranker_model
        )
    else:
        final_results = []

    return final_results
