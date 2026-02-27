"""
Recursive Character Chunking Strategy.

Uses LangChain's RecursiveCharacterTextSplitter which tries to split on
paragraph breaks first, then newlines, then sentences, then words.
"""

from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.chunking.base import BaseChunker


class RecursiveChunker(BaseChunker):
    """Split documents using LangChain's recursive character splitter."""

    def chunk(self, doc_id: str, content: str) -> List[Dict]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 4,       # Convert tokens to chars (~4 chars/token)
            chunk_overlap=self.chunk_overlap * 4,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        texts = splitter.split_text(content)

        return [
            {
                "doc_id": doc_id,
                "chunk_id": i,
                "text": text.strip()
            }
            for i, text in enumerate(texts) if text.strip()
        ]
