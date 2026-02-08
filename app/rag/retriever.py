"""
Telecom Document Retriever module.

CONCEPT: The Retriever's Job
=============================
When a user asks a question, the retriever:
1. Converts the question to an embedding (vector)
2. Searches Pinecone for similar document chunks
3. Returns the most relevant chunks with similarity scores
4. Formats them for use by the BillingAgent

This is the "R" in RAG - Retrieval!
"""

from typing import List, Dict, Tuple
from openai import OpenAI

from app.config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
)
from app.rag.pinecone_store import PineconeStore
from app.utils.logging import get_logger

logger = get_logger("retriever")


class TelecomRetriever:
    """
    Retrieves relevant document chunks for billing queries.
    
    Workflow:
    1. User question → embedding (via OpenAI)
    2. Embedding → Pinecone query
    3. Results → formatted context for LLM
    """
    
    def __init__(self):
        """Initialize the retriever with OpenAI and Pinecone clients."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set. Check your .env file.")
        
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        self.store = PineconeStore()
        self.embedding_model = OPENAI_EMBEDDING_MODEL
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Convert text to an embedding vector.
        
        CONCEPT: Embeddings
        ===================
        An embedding is a list of ~1536 numbers that represents
        the *meaning* of the text. Similar texts have similar embeddings.
        
        Args:
            text: The text to embed
        
        Returns:
            List of floats representing the embedding
        """
        response = self.openai.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K
    ) -> Tuple[List[Dict], float]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: The user's question
            top_k: Number of chunks to retrieve
        
        Returns:
            Tuple of (list of chunks, top similarity score)
            
            Each chunk dict contains:
            - doc_id: Document identifier (e.g., "DOC_4_INVOICE...")
            - chunk_id: Chunk number within document
            - text: The actual chunk text
            - score: Similarity score (0-1, higher = more similar)
        """
        logger.info(f"Retrieving for query: '{query[:50]}...'")
        
        # Step 1: Convert query to embedding
        query_embedding = self.get_embedding(query)
        
        # Step 2: Search Pinecone
        matches = self.store.query(
            query_embedding=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        
        if not matches:
            logger.warning("No matches found in vector store.")
            return [], 0.0
        
        # Step 3: Format results
        chunks = []
        for match in matches:
            metadata = match.get("metadata", {})
            chunks.append({
                "doc_id": metadata.get("doc_id", "unknown"),
                "chunk_id": metadata.get("chunk_id", 0),
                "text": metadata.get("text", ""),
                "score": match.get("score", 0.0)
            })
        
        # Get the top score (highest similarity)
        top_score = matches[0]["score"] if matches else 0.0
        
        logger.info(f"Retrieved {len(chunks)} chunks. Top score: {top_score:.3f}")
        
        return chunks, top_score
    
    def format_context_for_llm(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks as context for the LLM.
        
        This creates a structured text that the LLM can use
        to answer questions with citations.
        
        Args:
            chunks: List of retrieved chunk dictionaries
        
        Returns:
            Formatted string with source information
        """
        if not chunks:
            return "No relevant documents found."
        
        context_parts = ["## Retrieved Documents\n"]
        
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(f"""
### Source {i}
- **Document ID**: {chunk['doc_id']}
- **Chunk ID**: {chunk['chunk_id']}
- **Relevance Score**: {chunk['score']:.3f}

**Content**:
{chunk['text']}

---
""")
        
        return "\n".join(context_parts)
    
    def create_citations_from_chunks(
        self,
        chunks: List[Dict],
        answer: str
    ) -> List[Dict]:
        """
        Create citation objects from retrieved chunks.
        
        This is a helper to format citations for the BillingAgent response.
        
        Args:
            chunks: Retrieved chunks
            answer: The generated answer (to find relevant quotes)
        
        Returns:
            List of citation dicts with doc_id, chunk_id, quote
        """
        citations = []
        
        for chunk in chunks:
            # Extract a relevant quote (first 20 words or full text if shorter)
            text = chunk.get("text", "")
            words = text.split()
            quote = " ".join(words[:20]) + ("..." if len(words) > 20 else "")
            
            citations.append({
                "doc_id": chunk["doc_id"],
                "chunk_id": chunk["chunk_id"],
                "quote": quote
            })
        
        return citations
