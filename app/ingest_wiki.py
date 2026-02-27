"""
Downloads Wikipedia articles and ingests them into the telecom-wiki namespace.
"""
import wikipedia
import re
from pathlib import Path
from typing import List, Dict

from app.config import DATA_DIR
from app.rag.pinecone_store import PineconeStore
from app.ingest import chunk_document, embed_chunks
from app.utils.logging import get_logger

logger = get_logger("ingest_wiki")

WIKIPEDIA_ARTICLES = [
    "AT&T",
    "Telephone billing",       # Replaces "Mobile phone billing" which doesn't exist
    "Telephone call",
    "Roaming",
    "Mobile data",
    "5G NR",
    "Late fee",
    "Subscriber identity module",
    "Internet service provider",
    "Data cap"
]

def fetch_wikipedia_articles() -> List[Dict[str, str]]:
    documents = []
    for title in WIKIPEDIA_ARTICLES:
        try:
            logger.info(f"Fetching Wikipedia article: {title}")
            # Try to fetch, if disambiguation, take first
            try:
                page = wikipedia.page(title, auto_suggest=False)
            except wikipedia.exceptions.DisambiguationError as e:
                logger.warning(f"Disambiguation for {title}, taking first option: {e.options[0]}")
                page = wikipedia.page(e.options[0], auto_suggest=False)
            
            # Save for inspection
            out_dir = DATA_DIR / "wikipedia"
            out_dir.mkdir(exist_ok=True, parents=True)
            safe_title = re.sub(r'[^a-zA-Z0-9]', '_', title)
            out_path = out_dir / f"{safe_title}.txt"
            
            # Some wikipedia pages can be huge. We'll take the first 15000 chars to avoid massive ingest time
            content = page.content[:15000] 
            
            out_path.write_text(content, encoding="utf-8")
            
            documents.append({
                "doc_id": title,
                "filename": f"wikipedia/{safe_title}.txt",
                "content": content
            })
        except Exception as e:
            logger.error(f"Failed to fetch {title}: {e}")
            
    return documents

def run_wiki_ingestion():
    print("\n============================================================")
    print("          TELECOM WIKI - WIKIPEDIA INGESTION                ")
    print("============================================================\n")
    
    print("[1/4] Fetching Wikipedia articles...")
    documents = fetch_wikipedia_articles()
    print(f"OK - Fetched {len(documents)} articles\n")
    
    print("[2/4] Chunking articles...")
    all_chunks = []
    for doc in documents:
        # Wikipedia has lots of \n, so chunk_document should work fine
        chunks = chunk_document(doc["doc_id"], doc["content"])
        all_chunks.extend(chunks)
        logger.info(f"Chunked {doc['doc_id']}: {len(chunks)} chunks")
    print(f"OK - Created {len(all_chunks)} chunks\n")
    
    print("[3/4] Embedding chunks...")
    vectors = embed_chunks(all_chunks)
    print(f"OK - Generated {len(vectors)} embeddings\n")
    
    print("[4/4] Uploading to Pinecone namespace: telecom-wiki...")
    store = PineconeStore()
    
    logger.info("Clearing existing wiki namespace...")
    try:
        store.index.delete(delete_all=True, namespace="telecom-wiki")
    except Exception as e: # Pinecone might throw if namespace doesn't exist
        logger.warning(f"Could not clear namespace: {e}")
        
    store.upsert_vectors(vectors, namespace="telecom-wiki")
    print(f"OK - Uploaded {len(vectors)} vectors to telecom-wiki\n")
    print("INGESTION COMPLETE!")

if __name__ == "__main__":
    run_wiki_ingestion()
