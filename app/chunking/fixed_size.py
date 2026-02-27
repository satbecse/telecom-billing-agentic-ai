"""
Fixed-Size Chunking Strategy.

Splits text into chunks of a fixed token count with overlap.
This is the original chunking method from the project.
"""

import re
from typing import List, Dict
from app.chunking.base import BaseChunker


class FixedSizeChunker(BaseChunker):
    """Split documents by fixed token windows with overlap."""

    def chunk(self, doc_id: str, content: str) -> List[Dict]:
        chunks = []
        paragraphs = re.split(r'\n\s*\n', content)

        current_chunk = ""
        chunk_id = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            combined = current_chunk + "\n\n" + para if current_chunk else para
            combined_tokens = len(combined) // 4  # ~4 chars per token

            if combined_tokens <= self.chunk_size:
                current_chunk = combined
            else:
                if current_chunk:
                    chunks.append({
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "text": current_chunk.strip()
                    })
                    chunk_id += 1

                if current_chunk and self.chunk_overlap > 0:
                    words = current_chunk.split()
                    overlap_words = int(self.chunk_overlap / 4)
                    overlap_text = " ".join(words[-overlap_words:]) if len(words) > overlap_words else ""
                    current_chunk = overlap_text + "\n\n" + para
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "text": current_chunk.strip()
            })

        return chunks
