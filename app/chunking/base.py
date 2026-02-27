"""Base class for all chunking strategies."""

from abc import ABC, abstractmethod
from typing import List, Dict


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 75):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, doc_id: str, content: str) -> List[Dict]:
        """
        Split a document into chunks.

        Args:
            doc_id: Document identifier
            content: Full document text

        Returns:
            List of dicts with 'doc_id', 'chunk_id', 'text'
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__
