"""
Pinecone Vector Store module.

CONCEPT: What is a Vector Database?
====================================
Unlike traditional databases that search by exact text matches,
vector databases search by *semantic meaning*.

How it works:
1. Documents are converted to "embeddings" (lists of ~1536 numbers)
2. These numbers capture the *meaning* of the text
3. Similar meanings → similar numbers → close in vector space
4. We can find relevant documents by searching for nearby vectors

Example:
- "What's my January bill?" → embedding → [0.12, -0.45, 0.78, ...]
- Search finds DOC_4 (January invoice) because its embedding is "close"
"""

from typing import List, Dict, Optional, Tuple
from pinecone import Pinecone, ServerlessSpec

from app.config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_DIMENSION,
)
from app.utils.logging import get_logger

logger = get_logger("pinecone_store")


class PineconeStore:
    """
    Manages the Pinecone vector database connection and operations.
    
    Key operations:
    - create_index(): Create the serverless index if it doesn't exist
    - upsert_vectors(): Add document chunks to the index
    - query(): Search for similar documents
    - delete_all(): Clear the index (for re-ingestion)
    """
    
    def __init__(self):
        """Initialize the Pinecone client."""
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY is not set. Check your .env file.")
        
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index_name = PINECONE_INDEX_NAME
        self.namespace = PINECONE_NAMESPACE
        self._index = None
    
    @property
    def index(self):
        """Get or initialize the Pinecone index."""
        if self._index is None:
            self._index = self.pc.Index(self.index_name)
        return self._index
    
    def index_exists(self) -> bool:
        """Check if the index already exists."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        return self.index_name in existing_indexes
    
    def create_index(self, dimension: int = EMBEDDING_DIMENSION) -> bool:
        """
        Create the Pinecone serverless index if it doesn't exist.
        
        Args:
            dimension: Vector dimension (1536 for text-embedding-3-small)
        
        Returns:
            True if index was created, False if it already existed
        """
        if self.index_exists():
            logger.info(f"Index '{self.index_name}' already exists.")
            return False
        
        logger.info(f"Creating serverless index '{self.index_name}'...")
        
        self.pc.create_index(
            name=self.index_name,
            dimension=dimension,
            metric="cosine",  # Cosine similarity: higher = more similar
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION
            )
        )
        
        # Wait for index to be ready
        logger.info("Waiting for index to be ready...")
        import time
        while not self.pc.describe_index(self.index_name).status.get("ready", False):
            time.sleep(1)
        
        logger.info(f"Index '{self.index_name}' is ready!")
        return True
    
    def upsert_vectors(
        self,
        vectors: List[Dict],
        batch_size: int = 100,
        namespace: Optional[str] = None
    ) -> int:
        """
        Upsert (insert or update) vectors into the index.
        
        CONCEPT: "Upsert" = Update if exists, Insert if new
        
        Args:
            vectors: List of dicts with 'id', 'values' (embedding), 'metadata'
            batch_size: Number of vectors to upsert at once
            namespace: Optional namespace to upsert into, defaults to target namespace
        
        Returns:
            Number of vectors upserted
        """
        total_upserted = 0
        
        target_namespace = namespace if namespace else self.namespace
        
        # Process in batches (Pinecone has limits on batch size)
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=target_namespace)
            total_upserted += len(batch)
            logger.debug(f"Upserted batch {i // batch_size + 1}: {len(batch)} vectors")
        
        logger.info(f"Total vectors upserted: {total_upserted}")
        return total_upserted
    
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 4,
        include_metadata: bool = True,
        namespace: Optional[str] = None
    ) -> List[Dict]:
        """
        Query the index for similar vectors.
        
        Args:
            query_embedding: The embedding of the search query
            top_k: Number of results to return
            include_metadata: Whether to include document metadata
            namespace: Optional namespace to query (defaults to self.namespace)
        
        Returns:
            List of matching documents with scores and metadata
        """
        target_namespace = namespace if namespace else self.namespace
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=include_metadata,
            namespace=target_namespace
        )
        
        # Convert Pinecone results to our format
        matches = []
        for match in results.get("matches", []):
            matches.append({
                "id": match["id"],
                "score": match["score"],
                "metadata": match.get("metadata", {})
            })
        
        return matches
    
    def delete_namespace(self) -> None:
        """Delete all vectors in the namespace (for clean re-ingestion)."""
        logger.info(f"Deleting all vectors in namespace '{self.namespace}'...")
        self.index.delete(delete_all=True, namespace=self.namespace)
        logger.info("Namespace cleared.")
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.get("total_vector_count", 0),
            "namespaces": stats.get("namespaces", {})
        }
