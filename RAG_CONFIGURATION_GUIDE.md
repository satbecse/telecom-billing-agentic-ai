# Telecom Agentic AI: RAG Configuration Guide

This guide explains the key hyperparameters and configurations used across the Telecom Billing Agentic AI system. Tuning these parameters directly impacts the performance, accuracy, and cost of the Retrieval-Augmented Generation (RAG) system.

---

## 1. LLM Generation Parameters

These parameters control how the AI formulates its responses. They are primarily used when calling `openai.chat.completions.create()`.

### `temperature`
- **What it does:** Controls the randomness/creativity of the LLM's output. A value of `0.0` makes the model completely deterministic (always chooses the most probable next word). Higher values (e.g., `0.8` or `1.0`) make the output more creative and varied.
- **Where it's used & How it affects RAG:**
  - **BillingAgent (`temperature=0.3`):** Low temperature is used because billing queries require strict factual accuracy. We want the model to rely entirely on the retrieved context and not "hallucinate" creative numbers.
  - **SalesAgent (`temperature=0.7`):** Higher temperature is used to make the agent sound friendly, conversational, and natural when answering general policy or plan questions.
  - **Router (`temperature=0.0`):** Strict `0.0` is used for intent classification because we need absolute consistency. Given the same query, it should always route to the exact same agent.
  - **ManagerAgent (`temperature=0.1`):** Very low temperature for validation to ensure strict, logical evaluation of the BillingAgent's output without making creative leaps.

### `max_tokens`
- **What it does:** Limits the maximum length of the generated response (in tokens; ~4 chars per token).
- **Where it's used & How it affects RAG:**
  - **SalesAgent/BillingAgent (`max_tokens=500` / `1000`):** Prevents the agent from rambling incessantly or burning API credits unnecessarily. It forces the response to be concise.
  - **Router (`max_tokens=50`):** Since the router only needs to output a single word (e.g., `billing_account_specific`), we cap it tightly to prevent it from accidentally generating explanations and wasting tokens.

### `model` (e.g., `gpt-4o-mini`)
- **What it does:** The specific LLM used for text generation.
- **Where it's used & How it affects RAG:** Defined in `app/config.py` (`OPENAI_MODEL`). We use `gpt-4o-mini` because it is fast and cheap, making it excellent for multi-agent workflows where multiple LLM calls happen in a single user interaction (Routing → Retrieval generation → Manager validation). If greater complex reasoning is needed for complex billing disputes, it could be swapped to `gpt-4o`.

---

## 2. Retrieval & Embedding Parameters

These control how the system converts text to numbers and finds relevant documents.

### `embedding_model` (e.g., `text-embedding-3-small`)
- **What it does:** The model used to convert text (both the documents and the user's query) into a dense vector (a list of 1536 numbers).
- **Where it's used & How it affects RAG:** Used in `app/rag/retriever.py` and `app/ingest.py` (via `OPENAI_EMBEDDING_MODEL` in config.py). `text-embedding-3-small` is the latest standard. If swapped, the entire Pinecone database must be wiped and re-ingested because vectors from different models are mathematically incompatible.

### `RETRIEVAL_TOP_K` (e.g., `3` or `4`)
- **What it does:** The number of document chunks retrieved from Pinecone to feed to the LLM.
- **Where it's used & How it affects RAG:** Defined in `app/config.py` and used in `app/rag/retriever.py`. 
  - **If too low (e.g., 1):** The LLM might miss the answer if it spans multiple pages or if the top chunk is a false positive.
  - **If too high (e.g., 20):** You risk "Lost in the Middle" syndrome (where the LLM forgets context buried in the middle of a massive prompt), and it significantly increases API costs and latency.

---

## 3. Chunking Parameters

These parameters define how the raw PDFs are sliced into smaller pieces before being embedded.

### `chunk_size`
- **What it does:** The target number of tokens/characters per chunk.
- **Where it's used & How it affects RAG:** Within `app/chunking/`.
  - **If too small (e.g., 50 tokens):** The embedding captures very specific semantic meaning, but the chunk lacks the surrounding context needed by the LLM to formulate an answer (e.g., an invoice line item severed from the customer's name).
  - **If too large (e.g., 2000 tokens):** The embedding's semantic meaning becomes diluted because it covers too many topics. The retriever struggles to match specific user queries to the massive chunk.
  - *Note:* In our evaluation, `chunk_size=500` proved optimal for structured telecom PDFs.

### `chunk_overlap`
- **What it does:** The number of tokens/characters that overlap between adjacent chunks.
- **Where it's used & How it affects RAG:** Within `app/chunking/` (e.g., `chunk_overlap=50`). It prevents "cutoff" errors where a critical sentence or invoice table spans the artificial boundary between two chunks. It ensures continuity of context.

### The 3 Chunking Strategies Tested
1. **Fixed-Size (Token-based):** Slices blindly by length. Surprising winner in our telecom evaluation because invoices are highly structured tabular data where semantic boundaries don't apply well.
2. **Recursive Character:** Tries to slice at paragraphs (`\n\n`), then lines (`\n`), then words. Great for prose like Wikipedia, but struggling with PDF invoices.
3. **Semantic:** Tries to use embeddings to slice when the "topic" changes. Performed worst for invoices because every line item has a slightly different semantic meaning, resulting in massive over-fragmentation.

---

## 4. RAG Retrieval Strategies

These alter the user's search query *before* hitting Pinecone.

### Naive RAG
- **How it works:** `query -> embed -> search Pinecone`.
- **Effect:** Fast and cheap, but fails on "vocabulary mismatch" (when the user asks "How much did I talk overseas?" but the document says "International Voice Usage Overage").

### HyDE (Hypothetical Document Embeddings)
- **How it works:** `query -> LLM generates fake answer -> embed fake answer -> search Pinecone`.
- **Effect:** Slower (requires an extra LLM call), but incredibly accurate. Our evaluation proved this was the best performer. It bridges the vocabulary gap because the generated "fake answer" naturally uses the dense, corporate terminology found in the actual billing PDFs.

### Multi-Query RAG
- **How it works:** `query -> LLM generates 3 variations -> retrieve for all -> merge`.
- **Effect:** Broadens the search net. Good for vague user queries, but retrieving 3x the documents can dilute the final context window with slightly irrelevant data.

---

## 5. System Guardrails

### `CONFIDENCE_THRESHOLD` (e.g., `0.75`)
- **What it does:** The minimum retrieval score allowed, or the minimum confidence score the ManagerAgent requires to approve a response.
- **Where it's used & How it affects RAG:** In `app/agents/manager.py` or defined in configs. If set too high (e.g., `0.95`), the system will constantly reject answers and ask the user for clarification, resulting in a frustrating UX. If set too low (e.g., `0.30`), the system will confidently hallucinate answers based on completely irrelevant retrieved documents.
