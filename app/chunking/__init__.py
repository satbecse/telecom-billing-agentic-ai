"""Pluggable chunking strategies for the RAG pipeline."""

from app.chunking.base import BaseChunker
from app.chunking.fixed_size import FixedSizeChunker
from app.chunking.recursive import RecursiveChunker
from app.chunking.semantic import SemanticChunker

CHUNKING_STRATEGIES = {
    "fixed_size": FixedSizeChunker,
    "recursive": RecursiveChunker,
    "semantic": SemanticChunker,
}

def get_chunker(strategy: str = "fixed_size", **kwargs) -> BaseChunker:
    """Get a chunker by strategy name."""
    if strategy not in CHUNKING_STRATEGIES:
        raise ValueError(f"Unknown chunking strategy: {strategy}. Options: {list(CHUNKING_STRATEGIES.keys())}")
    return CHUNKING_STRATEGIES[strategy](**kwargs)
