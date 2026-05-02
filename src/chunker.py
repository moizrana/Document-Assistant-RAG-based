"""
Text Chunking module.
Implements a recursive character text splitter that preserves document structure
and maintains metadata through the chunking process.
"""

import logging
import uuid
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a text chunk with metadata and unique identifier."""
    chunk_id: str
    content: str
    metadata: dict = field(default_factory=dict)


class RecursiveCharacterSplitter:
    """
    Recursively splits text using a hierarchy of separators:
    paragraphs -> newlines -> sentences -> words.
    
    This ensures chunks break at natural language boundaries
    rather than cutting mid-sentence.
    """

    SEPARATORS = ["\n\n", "\n", ". ", " "]

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of overlapping characters between consecutive chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks using recursive separator strategy."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        return self._recursive_split(text, 0)

    def _recursive_split(self, text: str, separator_idx: int) -> list[str]:
        """Recursively split text, trying larger separators first."""
        if separator_idx >= len(self.SEPARATORS):
            # Last resort: hard-split by character count
            return self._hard_split(text)

        separator = self.SEPARATORS[separator_idx]
        parts = text.split(separator)

        chunks = []
        current_chunk = ""

        for part in parts:
            # Would adding this part exceed chunk_size?
            candidate = (current_chunk + separator + part) if current_chunk else part

            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                # Save the current chunk if it has content
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # If the part itself is too large, recursively split it
                if len(part) > self.chunk_size:
                    sub_chunks = self._recursive_split(part, separator_idx + 1)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
                else:
                    current_chunk = part

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Apply overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """Split text by fixed character count as a last resort."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.chunk_overlap
        return chunks

    def _apply_overlap(self, chunks: list[str]) -> list[str]:
        """Apply overlap by prepending the tail of the previous chunk."""
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            overlap_text = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
            # Only add overlap if it doesn't start mid-word
            space_idx = overlap_text.find(" ")
            if space_idx != -1:
                overlap_text = overlap_text[space_idx + 1:]
            overlapped.append(overlap_text + " " + chunks[i])
        return overlapped


def chunk_documents(documents: list, chunk_size: int = 500, chunk_overlap: int = 50) -> list[Chunk]:
    """
    Split a list of Document objects into smaller Chunks while preserving metadata.
    
    Args:
        documents: List of Document objects from the loader.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap between consecutive chunks.
    
    Returns:
        List of Chunk objects with unique IDs and inherited metadata.
    """
    splitter = RecursiveCharacterSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []

    for doc in documents:
        text_chunks = splitter.split_text(doc.content)
        for idx, text in enumerate(text_chunks):
            chunk = Chunk(
                chunk_id=str(uuid.uuid4()),
                content=text,
                metadata={
                    **doc.metadata,
                    "chunk_index": idx,
                    "total_chunks": len(text_chunks),
                }
            )
            chunks.append(chunk)

    logger.info(
        f"Chunked {len(documents)} document(s) into {len(chunks)} chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks
