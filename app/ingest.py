"""
Document Ingestion Script for Telecom Billing RAG System.

CONCEPT: What is "Ingestion"?
=============================
Before we can search documents, we need to:
1. READ the documents from disk
2. CHUNK them into smaller pieces (for better retrieval)
3. EMBED each chunk (convert to vectors)
4. UPSERT into Pinecone (store for searching)

This script handles all four steps.

Usage:
    python -m app.ingest

This will:
1. Create the Pinecone index if it doesn't exist
2. Read all 5 documents from data/docs/
3. Chunk them (300-500 tokens per chunk)
4. Embed using OpenAI
5. Upload to Pinecone
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
from openai import OpenAI

from app.config import (
    validate_config,
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
)
from app.rag.pinecone_store import PineconeStore
from app.utils.logging import setup_logging, get_logger, Colors
from app.rag.pdf_loader import extract_text_from_pdf

logger = get_logger("ingest")


# =============================================================================
# DOCUMENT LOADING
# =============================================================================

def load_documents() -> List[Dict[str, str]]:
    """
    Load all documents from the data/docs/customer_pdfs directory.
    
    Returns:
        List of dicts with 'doc_id', 'filename', and 'content'
    """
    documents = []
    
    pdf_dir = DATA_DIR / "customer_pdfs"
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    
    # Find all .pdf files in the docs directory
    doc_files = sorted(pdf_dir.glob("*.pdf"))
    
    if not doc_files:
        raise FileNotFoundError(f"No .pdf files found in {pdf_dir}")
    
    for doc_path in doc_files:
        content = extract_text_from_pdf(str(doc_path))
        
        if not content:
            logger.warning(f"Could not extract text from {doc_path.name}")
            continue
            
        # Extract doc_id from the first line or filename
        doc_id = doc_path.stem  # Filename without extension
        
        documents.append({
            "doc_id": doc_id,
            "filename": doc_path.name,
            "content": content
        })
        
        logger.info(f"Loaded: {doc_path.name} ({len(content)} characters)")
    
    return documents


# =============================================================================
# CHUNKING
# =============================================================================

def estimate_tokens(text: str) -> int:
    """
    Estimate the number of tokens in text.
    
    A rough estimate: ~4 characters per token for English text.
    This avoids the overhead of loading a tokenizer.
    
    Args:
        text: The text to estimate
    
    Returns:
        Estimated token count
    """
    return len(text) // 4


def chunk_document(
    doc_id: str,
    content: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict]:
    """
    Split a document into overlapping chunks.
    
    CONCEPT: Why Chunk?
    ===================
    - LLMs have context limits (can't process huge documents at once)
    - Smaller chunks = more precise retrieval
    - Overlap ensures we don't miss information at boundaries
    
    Args:
        doc_id: Document identifier
        content: Full document text
        chunk_size: Target tokens per chunk (300-500)
        chunk_overlap: Tokens to overlap between chunks (50-100)
    
    Returns:
        List of chunk dicts with doc_id, chunk_id, text
    """
    chunks = []
    
    # Split by paragraphs first (preserve meaning)
    paragraphs = re.split(r'\n\s*\n', content)
    
    current_chunk = ""
    chunk_id = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Estimate tokens with this paragraph added
        combined = current_chunk + "\n\n" + para if current_chunk else para
        combined_tokens = estimate_tokens(combined)
        
        if combined_tokens <= chunk_size:
            # Add to current chunk
            current_chunk = combined
        else:
            # Save current chunk and start new one
            if current_chunk:
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "text": current_chunk.strip()
                })
                chunk_id += 1
            
            # Start new chunk, keeping overlap from previous
            if current_chunk and chunk_overlap > 0:
                # Take last ~overlap tokens from previous chunk
                words = current_chunk.split()
                overlap_words = int(chunk_overlap / 4)  # Estimate words
                overlap_text = " ".join(words[-overlap_words:]) if len(words) > overlap_words else ""
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = para
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "text": current_chunk.strip()
        })
    
    return chunks


def chunk_all_documents(documents: List[Dict]) -> List[Dict]:
    """
    Chunk all documents.
    
    Args:
        documents: List of document dicts
    
    Returns:
        List of all chunks from all documents
    """
    all_chunks = []
    
    for doc in documents:
        chunks = chunk_document(doc["doc_id"], doc["content"])
        all_chunks.extend(chunks)
        logger.info(f"  {doc['doc_id']}: {len(chunks)} chunks")
    
    return all_chunks


# =============================================================================
# EMBEDDING
# =============================================================================

def embed_chunks(
    chunks: List[Dict],
    batch_size: int = 20
) -> List[Dict]:
    """
    Generate embeddings for all chunks using OpenAI.
    
    Args:
        chunks: List of chunk dicts
        batch_size: Number of chunks to embed at once
    
    Returns:
        List of dicts ready for Pinecone upsert
    """
    client = OpenAI(api_key=OPENAI_API_KEY)
    vectors = []
    
    # Process in batches
    total = len(chunks)
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk["text"] for chunk in batch]
        
        # Get embeddings from OpenAI
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=texts
        )
        
        # Create Pinecone vectors
        for j, embedding_data in enumerate(response.data):
            chunk = batch[j]
            vector_id = f"{chunk['doc_id']}__chunk_{chunk['chunk_id']}"
            
            vectors.append({
                "id": vector_id,
                "values": embedding_data.embedding,
                "metadata": {
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"][:4000]  # Pinecone metadata limit
                }
            })
        
        logger.info(f"  Embedded batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size}")
    
    return vectors


# =============================================================================
# MAIN INGESTION FLOW
# =============================================================================

def run_ingestion():
    """
    Run the full ingestion pipeline.
    
    Steps:
    1. Validate configuration
    2. Load documents
    3. Chunk documents
    4. Create embeddings
    5. Upsert to Pinecone
    """
    print(f"\n{Colors.BOLD}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{'TELECOM BILLING RAG - DOCUMENT INGESTION':^60}{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 60}{Colors.RESET}\n")
    
    # Step 0: Validate config
    print("[1/5] Validating configuration...")
    validate_config()
    print(f"      {Colors.SUCCESS}OK - Configuration valid{Colors.RESET}\n")
    
    # Step 1: Initialize Pinecone
    print("[2/5] Initializing Pinecone...")
    store = PineconeStore()
    created = store.create_index()
    if created:
        print(f"      {Colors.SUCCESS}OK - Created new index{Colors.RESET}\n")
    else:
        print(f"      {Colors.INFO}(i) Index already exists{Colors.RESET}")
        # Clear existing data for fresh ingestion
        print("      Clearing existing vectors...")
        store.delete_namespace()
        print(f"      {Colors.SUCCESS}OK - Namespace cleared{Colors.RESET}\n")
    
    # Step 2: Load documents
    print(f"[3/5] Loading documents from {DATA_DIR}...")
    documents = load_documents()
    print(f"      {Colors.SUCCESS}OK - Loaded {len(documents)} documents{Colors.RESET}\n")
    
    # Step 3: Chunk documents
    print(f"[4/5] Chunking documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = chunk_all_documents(documents)
    print(f"      {Colors.SUCCESS}OK - Created {len(chunks)} chunks{Colors.RESET}\n")
    
    # Step 4: Generate embeddings
    print(f"[5/5] Generating embeddings with {OPENAI_EMBEDDING_MODEL}...")
    vectors = embed_chunks(chunks)
    print(f"      {Colors.SUCCESS}OK - Generated {len(vectors)} embeddings{Colors.RESET}\n")
    
    # Step 5: Upsert to Pinecone
    print("[*] Uploading to Pinecone...")
    count = store.upsert_vectors(vectors)
    print(f"      {Colors.SUCCESS}OK - Uploaded {count} vectors{Colors.RESET}\n")
    
    # Done!
    print("=" * 60)
    print(f"{Colors.SUCCESS}INGESTION COMPLETE!{Colors.RESET}")
    print("=" * 60 + "\n")
    
    # Print stats
    stats = store.get_stats()
    print("Index Stats:")
    print(f"   Total vectors: {stats['total_vectors']}")
    print(f"   Namespace: {store.namespace}")
    print()


if __name__ == "__main__":
    setup_logging()
    run_ingestion()
