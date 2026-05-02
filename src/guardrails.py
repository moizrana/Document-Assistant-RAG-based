"""
Guardrails module.
Implements prompt injection protection and input/output validation.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Result of guardrail checks."""
    is_safe: bool
    reason: str = ""
    sanitized_query: str = ""


INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above\s+instructions",
    r"disregard\s+(all\s+)?previous",
    r"forget\s+(all\s+)?previous",
    r"override\s+(all\s+)?system\s+prompt",
    r"you\s+are\s+now\s+(?:a|an)\s+",
    r"pretend\s+(?:you\s+are|to\s+be)",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*",
    r"reveal\s+(?:your\s+)?(?:system\s+)?prompt",
    r"show\s+(?:your\s+)?(?:system\s+)?prompt",
    r"what\s+(?:is|are)\s+your\s+(?:system\s+)?(?:instructions|prompt)",
    r"repeat\s+(?:your\s+)?(?:system\s+)?(?:instructions|prompt)",
    r"jailbreak",
    r"DAN\s+mode",
    r"developer\s+mode",
]

_compiled_patterns = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
MAX_QUERY_LENGTH = 2000


def check_input(query: str) -> GuardrailResult:
    """Validate user input against prompt injection patterns."""
    if not query or not query.strip():
        return GuardrailResult(is_safe=False, reason="Empty query provided.")

    query = query.strip()

    if len(query) > MAX_QUERY_LENGTH:
        return GuardrailResult(is_safe=False, reason=f"Query exceeds max length of {MAX_QUERY_LENGTH} chars.")

    for pattern in _compiled_patterns:
        if pattern.search(query):
            logger.warning(f"Prompt injection detected: '{pattern.pattern}'")
            return GuardrailResult(is_safe=False, reason="Your query contains patterns that look like prompt manipulation. Please rephrase.")

    special_ratio = sum(1 for c in query if not c.isalnum() and c not in " .,?!'-()") / max(len(query), 1)
    if special_ratio > 0.4:
        return GuardrailResult(is_safe=False, reason="Too many special characters. Please use natural language.")

    return GuardrailResult(is_safe=True, sanitized_query=query)


def check_output(response: str) -> str:
    """Sanitize LLM output, removing system prompt leakage."""
    if not response:
        return response
    leakage = [r"<\s*system\s*>.*?<\s*/\s*system\s*>", r"system\s*instruction\s*:", r"my\s+instructions?\s+are\s*:"]
    for p in leakage:
        response = re.sub(p, "[REDACTED]", response, flags=re.IGNORECASE | re.DOTALL)
    return response
