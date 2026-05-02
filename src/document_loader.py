"""
Document Loader module.
Supports loading and extracting text from PDF, HTML, and TXT files.
Uses a factory pattern to auto-detect file format by extension.
"""

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Represents a loaded document with its content and metadata."""
    content: str
    metadata: dict = field(default_factory=dict)


class PDFLoader:
    """Load text content from PDF files using PyMuPDF."""

    @staticmethod
    def load(file_bytes: bytes, filename: str) -> list[Document]:
        """Extract text from a PDF file, page by page."""
        import fitz  # PyMuPDF

        documents = []
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text")
                if text.strip():
                    documents.append(Document(
                        content=text.strip(),
                        metadata={
                            "source": filename,
                            "page": page_num + 1,
                            "total_pages": len(doc),
                            "type": "pdf"
                        }
                    ))
            doc.close()
            logger.info(f"Loaded PDF '{filename}': {len(documents)} pages with content")
        except Exception as e:
            logger.error(f"Failed to load PDF '{filename}': {e}")
            raise ValueError(f"Failed to process PDF file: {e}")
        return documents


class HTMLLoader:
    """Load text content from HTML files using BeautifulSoup."""

    @staticmethod
    def load(file_bytes: bytes, filename: str) -> list[Document]:
        """Extract clean text from an HTML file."""
        from bs4 import BeautifulSoup

        documents = []
        try:
            soup = BeautifulSoup(file_bytes, "html.parser")

            # Remove script and style elements
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()

            # Extract title
            title = soup.title.string.strip() if soup.title and soup.title.string else filename

            text = soup.get_text(separator="\n", strip=True)
            if text.strip():
                documents.append(Document(
                    content=text.strip(),
                    metadata={
                        "source": filename,
                        "title": title,
                        "type": "html"
                    }
                ))
            logger.info(f"Loaded HTML '{filename}': {len(text)} characters")
        except Exception as e:
            logger.error(f"Failed to load HTML '{filename}': {e}")
            raise ValueError(f"Failed to process HTML file: {e}")
        return documents


class TextLoader:
    """Load text content from plain text files."""

    @staticmethod
    def load(file_bytes: bytes, filename: str) -> list[Document]:
        """Read a plain text file with encoding detection."""
        documents = []
        try:
            # Try UTF-8 first, then fall back to latin-1
            try:
                text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = file_bytes.decode("latin-1")

            if text.strip():
                documents.append(Document(
                    content=text.strip(),
                    metadata={
                        "source": filename,
                        "type": "text"
                    }
                ))
            logger.info(f"Loaded TXT '{filename}': {len(text)} characters")
        except Exception as e:
            logger.error(f"Failed to load TXT '{filename}': {e}")
            raise ValueError(f"Failed to process text file: {e}")
        return documents


# --- Supported file extensions ---
SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".txt", ".md"}

LOADER_MAP = {
    ".pdf": PDFLoader,
    ".html": HTMLLoader,
    ".htm": HTMLLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def load_document(file_bytes: bytes, filename: str) -> list[Document]:
    """
    Factory function: auto-detect file format and load documents.
    
    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename (used for extension detection and metadata).
    
    Returns:
        List of Document objects extracted from the file.
    
    Raises:
        ValueError: If the file format is not supported.
    """
    ext = Path(filename).suffix.lower()
    if ext not in LOADER_MAP:
        raise ValueError(
            f"Unsupported file format: '{ext}'. "
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    loader_class = LOADER_MAP[ext]
    documents = loader_class.load(file_bytes, filename)

    if not documents:
        raise ValueError(f"No text content could be extracted from '{filename}'.")

    return documents
