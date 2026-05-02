"""
Query Expansion module.
Uses LLM to generate alternative phrasings of a query to improve retrieval recall.
This catches relevant documents that might be missed by a single query phrasing.
"""

import logging
import re

from src.generator import generate_simple

logger = logging.getLogger(__name__)


EXPANSION_PROMPT = """Given the following user question, generate 2 alternative phrasings that capture the same intent but use different words or perspectives. The alternative phrasings should help find relevant documents that the original question might miss.

Original question: {query}

Return ONLY the alternative phrasings, one per line, numbered 1 and 2. Do not include any other text.

1.
2."""


def expand_query(query: str) -> list[str]:
    """
    Generate alternative phrasings of a query using an LLM.
    
    Args:
        query: The original user query.
    
    Returns:
        List of query variants (original + alternatives). 
        Returns just [query] if expansion fails.
    """
    try:
        result = generate_simple(
            EXPANSION_PROMPT.format(query=query),
            system_prompt="You are a search query expansion assistant. Generate alternative search queries."
        )

        if not result:
            return [query]

        # Parse numbered alternatives
        alternatives = []
        for line in result.strip().split("\n"):
            # Remove numbering like "1.", "2.", "1)", "2)"
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if cleaned and cleaned != query and len(cleaned) > 5:
                alternatives.append(cleaned)

        expanded = [query] + alternatives[:2]  # Original + up to 2 alternatives
        logger.info(f"Expanded query into {len(expanded)} variants: {expanded}")
        return expanded

    except Exception as e:
        logger.warning(f"Query expansion failed: {e}. Using original query only.")
        return [query]
