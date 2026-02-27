"""
Evaluation runner for the Telecom Billing RAG Comparison Framework.

Runs every combination of RAG strategy x Chunking method against all
10 evaluation queries, scores with LLM-based evaluator, and produces a report.

Usage:
    python -m app.evaluation.eval_runner
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
from openai import OpenAI

from app.config import (
    DATA_DIR, OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, validate_config
)
from app.rag.pinecone_store import PineconeStore
from app.rag.pdf_loader import extract_text_from_pdf
from app.chunking import get_chunker

# --------------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------------

EVAL_QUERIES_FILE = DATA_DIR.parent / "eval_queries.txt"

WIKI_NAMESPACE = "telecom-wiki"
RETRIEVAL_TOP_K = 4

CHUNKING_METHODS = ["fixed_size", "recursive", "semantic"]

# --------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------

openai_client = OpenAI(api_key=OPENAI_API_KEY)


def load_eval_queries() -> List[Dict[str, str]]:
    """Load query | ground_truth pairs from eval_queries.txt."""
    queries = []
    with open(EVAL_QUERIES_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 1)
            query = parts[0].strip()
            ground_truth = parts[1].strip() if len(parts) > 1 else ""
            queries.append({"query": query, "ground_truth": ground_truth})
    return queries


def load_customer_documents() -> List[Dict[str, str]]:
    """Load customer PDF documents."""
    pdf_dir = DATA_DIR / "customer_pdfs"
    documents = []
    for doc_path in sorted(pdf_dir.glob("*.pdf")):
        content = extract_text_from_pdf(str(doc_path))
        if content:
            documents.append({
                "doc_id": doc_path.stem,
                "filename": doc_path.name,
                "content": content
            })
    return documents


def embed_query(query: str) -> List[float]:
    response = openai_client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL, input=query
    )
    return response.data[0].embedding


def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    """Embed chunks and return Pinecone-ready vectors."""
    vectors = []
    batch_size = 20
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c["text"] for c in batch]
        response = openai_client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL, input=texts
        )
        for j, emb_data in enumerate(response.data):
            chunk = batch[j]
            vectors.append({
                "id": f"{chunk['doc_id']}__chunk_{chunk['chunk_id']}",
                "values": emb_data.embedding,
                "metadata": {
                    "doc_id": chunk["doc_id"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"][:4000]
                }
            })
    return vectors


def retrieve_chunks(query_embedding: List[float], namespace: str, store: PineconeStore) -> List[Dict]:
    """Retrieve top-k chunks from a specific namespace."""
    results = store.index.query(
        vector=query_embedding, top_k=RETRIEVAL_TOP_K,
        include_metadata=True, namespace=namespace
    )
    return [
        {
            "text": m.get("metadata", {}).get("text", ""),
            "doc_id": m.get("metadata", {}).get("doc_id", ""),
            "score": m.get("score", 0)
        }
        for m in results.get("matches", [])
    ]


def ingest_with_chunker(chunker_name: str, documents: List[Dict], store: PineconeStore) -> str:
    """Re-ingest customer docs using a specific chunking strategy. Returns namespace name."""
    namespace = f"eval-{chunker_name}"
    chunker = get_chunker(chunker_name)

    # Chunk all documents
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk(doc["doc_id"], doc["content"])
        all_chunks.extend(chunks)

    print(f"    Chunked: {len(all_chunks)} chunks with {chunker_name}")

    # Embed
    vectors = embed_chunks(all_chunks)
    print(f"    Embedded: {len(vectors)} vectors")

    # Clear and upload
    try:
        store.index.delete(delete_all=True, namespace=namespace)
    except Exception:
        pass
    store.upsert_vectors(vectors, namespace=namespace)
    print(f"    Uploaded to namespace: {namespace}")

    # Small delay for Pinecone indexing
    time.sleep(2)
    return namespace


# --------------------------------------------------------------------------
# RAG STRATEGIES
# --------------------------------------------------------------------------

def rag_naive(query: str, doc_ns: str, store: PineconeStore) -> Tuple[str, List[str]]:
    """Naive RAG: embed query directly, search both namespaces."""
    emb = embed_query(query)
    chunks_docs = retrieve_chunks(emb, doc_ns, store)
    chunks_wiki = retrieve_chunks(emb, WIKI_NAMESPACE, store)
    all_chunks = chunks_docs + chunks_wiki
    return _generate(query, all_chunks), _get_contexts(all_chunks)


def rag_hyde(query: str, doc_ns: str, store: PineconeStore) -> Tuple[str, List[str]]:
    """HyDE RAG: generate hypothetical answer, embed that, then retrieve."""
    hypothesis_resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a telecom billing expert. Write a concise hypothetical answer to the following question. Maximum 3 sentences."},
            {"role": "user", "content": query}
        ],
        temperature=0.3, max_tokens=200
    )
    hypothesis = hypothesis_resp.choices[0].message.content
    emb = embed_query(hypothesis)
    chunks_docs = retrieve_chunks(emb, doc_ns, store)
    chunks_wiki = retrieve_chunks(emb, WIKI_NAMESPACE, store)
    all_chunks = chunks_docs + chunks_wiki
    return _generate(query, all_chunks), _get_contexts(all_chunks)


def rag_multi_query(query: str, doc_ns: str, store: PineconeStore) -> Tuple[str, List[str]]:
    """Multi-Query RAG: generate 3 query variations, retrieve for all, merge."""
    var_resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Generate 3 different phrasings of the following user query for a telecom billing system. Return only the 3 queries, one per line, nothing else."},
            {"role": "user", "content": query}
        ],
        temperature=0.5, max_tokens=200
    )
    variations = [v.strip() for v in var_resp.choices[0].message.content.splitlines() if v.strip()]
    all_queries = [query] + variations[:3]

    seen_texts = set()
    all_chunks = []
    for q in all_queries:
        emb = embed_query(q)
        for ns in [doc_ns, WIKI_NAMESPACE]:
            for chunk in retrieve_chunks(emb, ns, store):
                if chunk["text"] not in seen_texts:
                    seen_texts.add(chunk["text"])
                    all_chunks.append(chunk)

    return _generate(query, all_chunks[:8]), _get_contexts(all_chunks[:8])


RAG_STRATEGIES = {
    "Naive": rag_naive,
    "HyDE": rag_hyde,
    "Multi-Query": rag_multi_query,
}


def _get_contexts(chunks: List[Dict]) -> List[str]:
    return [c["text"] for c in chunks if c["text"]]


def _generate(query: str, chunks: List[Dict]) -> str:
    """Generate an answer from the retrieved chunks."""
    context = "\n\n---\n\n".join(
        f"[{c['doc_id']}] {c['text']}" for c in chunks if c["text"]
    )
    if not context:
        return "I could not find relevant information to answer this question."

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a telecom billing assistant. Answer the user's question "
                    "using ONLY the provided context. Be concise and factual. "
                    "If the context doesn't have enough information, say so.\n\n"
                    f"Context:\n{context}"
                )
            },
            {"role": "user", "content": query}
        ],
        temperature=0.0, max_tokens=400
    )
    return resp.choices[0].message.content.strip()


# --------------------------------------------------------------------------
# LLM-BASED EVALUATOR
# --------------------------------------------------------------------------

def evaluate_with_llm(query: str, answer: str, contexts: List[str], ground_truth: str) -> Dict[str, float]:
    """Evaluate a RAG response using GPT-4o-mini as the judge."""
    context_str = "\n\n".join(contexts[:4]) if contexts else "No context was retrieved."

    prompt = f"""You are evaluating a RAG system response. Score the following on 3 metrics (0.0 to 1.0):

QUESTION: {query}

RETRIEVED CONTEXT:
{context_str}

ANSWER GIVEN:
{answer}

GROUND TRUTH:
{ground_truth}

Evaluate and respond with ONLY a JSON object:
{{
  "faithfulness": <0.0-1.0, is the answer grounded in the retrieved context only?>,
  "relevancy": <0.0-1.0, does the answer directly address the question?>,
  "correctness": <0.0-1.0, how well does the answer match the ground truth?>
}}"""

    resp = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=100,
        response_format={"type": "json_object"}
    )

    try:
        scores = json.loads(resp.choices[0].message.content)
        return {
            "faithfulness": float(scores.get("faithfulness", 0)),
            "relevancy": float(scores.get("relevancy", 0)),
            "correctness": float(scores.get("correctness", 0)),
        }
    except Exception:
        return {"faithfulness": 0.0, "relevancy": 0.0, "correctness": 0.0}


# --------------------------------------------------------------------------
# MAIN RUNNER
# --------------------------------------------------------------------------

def run_evaluation():
    """Run all RAG x Chunking evaluations against all 10 queries."""
    validate_config()

    print("\n" + "=" * 70)
    print("  TELECOM BILLING - RAG EVALUATION FRAMEWORK")
    print("  3 RAG Strategies x 3 Chunking Methods x 10 Queries = 90 Runs")
    print("=" * 70)

    queries = load_eval_queries()
    store = PineconeStore()
    store.create_index()

    # Load customer documents once
    print("\n[SETUP] Loading customer documents...")
    documents = load_customer_documents()
    print(f"  Loaded {len(documents)} documents")

    all_results = []

    for chunker_name in CHUNKING_METHODS:
        print(f"\n{'='*70}")
        print(f"  CHUNKING: {chunker_name.upper()}")
        print(f"{'='*70}")

        # Re-ingest with this chunking strategy
        print(f"\n  [INGEST] Re-ingesting with {chunker_name} chunker...")
        doc_namespace = ingest_with_chunker(chunker_name, documents, store)

        for rag_name, rag_fn in RAG_STRATEGIES.items():
            print(f"\n  {'-'*60}")
            print(f"  RAG: {rag_name} + Chunking: {chunker_name}")
            print(f"  {'-'*60}")

            combo_scores = {"faithfulness": [], "relevancy": [], "correctness": []}

            for i, q in enumerate(queries, 1):
                print(f"    [{i:02d}/{len(queries)}] {q['query'][:55]}...", end=" ", flush=True)

                t0 = time.time()
                try:
                    answer, contexts = rag_fn(q["query"], doc_namespace, store)
                    scores = evaluate_with_llm(q["query"], answer, contexts, q["ground_truth"])
                except Exception as e:
                    print(f"ERROR: {e}")
                    scores = {"faithfulness": 0.0, "relevancy": 0.0, "correctness": 0.0}
                    answer, contexts = "", []

                elapsed = time.time() - t0
                avg = sum(scores.values()) / 3
                print(f"F:{scores['faithfulness']:.2f} R:{scores['relevancy']:.2f} C:{scores['correctness']:.2f} -> {avg:.2f} ({elapsed:.1f}s)")

                for k in combo_scores:
                    combo_scores[k].append(scores[k])

                all_results.append({
                    "rag_strategy": rag_name,
                    "chunking": chunker_name,
                    "query": q["query"],
                    "ground_truth": q["ground_truth"],
                    "answer": answer,
                    **scores,
                    "avg_score": avg
                })

            avg_f = sum(combo_scores["faithfulness"]) / len(queries)
            avg_r = sum(combo_scores["relevancy"]) / len(queries)
            avg_c = sum(combo_scores["correctness"]) / len(queries)
            total_avg = (avg_f + avg_r + avg_c) / 3
            print(f"\n  AVG -> F:{avg_f:.2f}  R:{avg_r:.2f}  C:{avg_c:.2f}  Total:{total_avg:.2f}")

    # Save raw results
    out_dir = DATA_DIR.parent / "evaluation_results"
    out_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = out_dir / f"results_full_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nRaw results saved to: {results_path}")

    # Auto-generate the report
    from app.evaluation.report import generate_report
    report_text = generate_report(all_results)
    report_path = out_dir / f"comparison_report_{timestamp}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(report_text)
    print(f"\nReport saved to: {report_path}")

    return all_results, str(results_path)


if __name__ == "__main__":
    run_evaluation()
