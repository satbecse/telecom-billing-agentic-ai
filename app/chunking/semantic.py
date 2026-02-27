"""
Semantic Chunking Strategy.

Uses OpenAI embeddings to detect topic boundaries. Sentences with similar
embeddings are grouped together; a new chunk starts when the topic shifts.
"""

from typing import List, Dict
from openai import OpenAI
from app.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL
from app.chunking.base import BaseChunker

import re


class SemanticChunker(BaseChunker):
    """Split documents by detecting semantic (topic) boundaries via embeddings."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.similarity_threshold = 0.78  # Below this = topic shift

    def chunk(self, doc_id: str, content: str) -> List[Dict]:
        # Split into sentences
        sentences = self._split_sentences(content)
        if len(sentences) <= 1:
            return [{"doc_id": doc_id, "chunk_id": 0, "text": content.strip()}]

        # Get embeddings for all sentences
        embeddings = self._get_embeddings(sentences)

        # Find topic boundaries by comparing consecutive sentence embeddings
        chunks = []
        current_chunk_sentences = [sentences[0]]
        chunk_id = 0

        for i in range(1, len(sentences)):
            similarity = self._cosine_similarity(embeddings[i - 1], embeddings[i])

            if similarity < self.similarity_threshold:
                # Topic shift detected -- save current chunk and start new one
                chunk_text = " ".join(current_chunk_sentences).strip()
                if chunk_text:
                    chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "text": chunk_text
                    })
                    chunk_id += 1
                current_chunk_sentences = [sentences[i]]
            else:
                current_chunk_sentences.append(sentences[i])

                # Also split if chunk gets too long
                current_text = " ".join(current_chunk_sentences)
                if len(current_text) // 4 > self.chunk_size:
                    chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "text": current_text.strip()
                    })
                    chunk_id += 1
                    current_chunk_sentences = []

        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences).strip()
            if chunk_text:
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": chunk_text
                })

        return chunks if chunks else [{"doc_id": doc_id, "chunk_id": 0, "text": content.strip()}]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        sentences = re.split(r'(?<=[.!?])\s+|\n\s*\n|\n', text)
        return [s.strip() for s in sentences if s.strip()]

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts, batching to avoid API limits."""
        all_embeddings = []
        batch_size = 50
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = self.client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=batch
            )
            all_embeddings.extend([d.embedding for d in response.data])
        return all_embeddings

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
