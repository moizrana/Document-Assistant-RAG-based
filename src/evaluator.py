"""
Evaluator module.
Implements RAGAS-style evaluation metrics using LLM-as-a-judge.
Measures Faithfulness and Answer Relevancy for RAG quality assessment.
"""

import logging
import re
from src.generator import generate_simple

logger = logging.getLogger(__name__)


def evaluate_faithfulness(answer: str, context_chunks: list[str]) -> float:
    """
    Evaluate if the answer is grounded in the provided context (no hallucination).
    Score 0.0 to 1.0 where 1.0 means fully faithful.
    """
    context = "\n---\n".join(context_chunks)
    prompt = f"""Evaluate faithfulness: Is the following answer fully supported by the context?

Context:
{context}

Answer:
{answer}

Rate faithfulness from 0.0 to 1.0:
- 1.0 = Every claim is supported by the context
- 0.5 = Some claims are supported, some are not
- 0.0 = The answer contradicts or adds unsupported information

Return ONLY a single number between 0.0 and 1.0. Nothing else."""

    try:
        result = generate_simple(prompt, "You are an evaluation judge. Return only a number.")
        score = float(re.search(r"(\d+\.?\d*)", result).group(1))
        return min(max(score, 0.0), 1.0)
    except Exception as e:
        logger.warning(f"Faithfulness evaluation failed: {e}")
        return -1.0


def evaluate_relevancy(query: str, answer: str) -> float:
    """
    Evaluate how well the answer addresses the user's question.
    Score 0.0 to 1.0 where 1.0 means perfectly relevant.
    """
    prompt = f"""Evaluate answer relevancy: Does this answer address the question?

Question: {query}
Answer: {answer}

Rate relevancy from 0.0 to 1.0:
- 1.0 = Directly and completely answers the question
- 0.5 = Partially answers the question
- 0.0 = Does not address the question at all

Return ONLY a single number between 0.0 and 1.0. Nothing else."""

    try:
        result = generate_simple(prompt, "You are an evaluation judge. Return only a number.")
        score = float(re.search(r"(\d+\.?\d*)", result).group(1))
        return min(max(score, 0.0), 1.0)
    except Exception as e:
        logger.warning(f"Relevancy evaluation failed: {e}")
        return -1.0


def evaluate_context_relevancy(query: str, context_chunks: list[str]) -> float:
    """
    Evaluate the signal-to-noise ratio of retrieved context.
    Score 0.0 to 1.0 where 1.0 means all context is relevant.
    """
    context = "\n---\n".join(context_chunks)
    prompt = f"""Evaluate context relevancy: How much of this context is useful for answering the question?

Question: {query}
Retrieved Context:
{context}

Rate from 0.0 to 1.0:
- 1.0 = All retrieved content is relevant
- 0.5 = About half is relevant
- 0.0 = None of the content is relevant

Return ONLY a single number between 0.0 and 1.0. Nothing else."""

    try:
        result = generate_simple(prompt, "You are an evaluation judge. Return only a number.")
        score = float(re.search(r"(\d+\.?\d*)", result).group(1))
        return min(max(score, 0.0), 1.0)
    except Exception as e:
        logger.warning(f"Context relevancy evaluation failed: {e}")
        return -1.0


def run_evaluation(query: str, answer: str, context_chunks: list[str]) -> dict:
    """Run all evaluation metrics and return scores."""
    scores = {
        "faithfulness": evaluate_faithfulness(answer, context_chunks),
        "answer_relevancy": evaluate_relevancy(query, answer),
        "context_relevancy": evaluate_context_relevancy(query, context_chunks),
    }
    logger.info(f"Evaluation scores: {scores}")
    return scores
