# Telecom Billing Agentic AI

A multi-agent AI system for telecom billing inquiries, demonstrating enterprise-grade patterns with:
- **LangChain + LangGraph** for agent orchestration
- **Pinecone** for vector-based RAG (Retrieval-Augmented Generation)
- **OpenAI GPT** for language understanding and generation
- **Guardrails** for safe, verified responses
- **Session Memory** for context-aware conversations
- **LangSmith** for tracing and debugging

> **Demo Project**: This uses mock data for demonstration purposes. Not for production use.

---

## What Does This System Do?

This system simulates a telecom customer service experience with three specialized AI agents:

```
┌─────────────────────────────────────────────────────────────────┐
│                     USER ASKS A QUESTION                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│  🔀 ROUTER: Classifies the query                                │
│     • billing_account_specific → BillingAgent                   │
│     • billing_general → BillingAgent                            │
│     • sales_general → SalesAgent (direct answer)                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                                       ▼
┌───────────────┐                     ┌───────────────┐
│  💼 SALES     │                     │  📊 BILLING   │
│  AGENT        │                     │  AGENT        │
│               │                     │               │
│  Handles:     │                     │  Uses RAG to: │
│  • Plan info  │                     │  • Search docs│
│  • Pricing    │                     │  • Get amounts│
│  • Policies   │                     │  • Cite source│
│               │                     │               │
│  CANNOT say:  │                     │  MUST provide:│
│  "Your bill   │                     │  • Citations  │
│  is $X"       │                     │  • Confidence │
└───────────────┘                     └───────┬───────┘
                                              │
                                              ▼
                                    ┌───────────────┐
                                    │  ✅ MANAGER   │
                                    │  AGENT        │
                                    │               │
                                    │  Validates:   │
                                    │  • Citations ✓│
                                    │  • Confidence │
                                    │  • $ amounts  │
                                    │               │
                                    │  Approves or  │
                                    │  rejects      │
                                    └───────┬───────┘
                                            │
                                            ▼
                              ┌─────────────────────────┐
                              │  📋 FINAL RESPONSE      │
                              │  (with citations)       │
                              └─────────────────────────┘
```

---

## Quick Start (5 minutes)

### Prerequisites
- Python 3.11+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))
- Pinecone API key ([get one here](https://app.pinecone.io/))
- LangSmith API key (optional, [get one here](https://smith.langchain.com/))

### Step 1: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment

```bash
# Copy the example environment file
# Windows:
copy .env.example .env
# Mac/Linux:
cp .env.example .env

# Edit .env with your API keys:
# OPENAI_API_KEY=sk-your-key-here
# PINECONE_API_KEY=your-pinecone-key-here
# LANGCHAIN_TRACING_V2=true (optional, for LangSmith)
# LANGCHAIN_API_KEY=your-langsmith-key (optional)
```

### Step 4: Ingest Documents

```bash
python -m app.ingest
```

This will:
1. Create a Pinecone serverless index (if needed)
2. Load the 5 demo documents
3. Chunk them into smaller pieces
4. Generate embeddings
5. Upload to Pinecone

### Step 5: Run the System

```bash
# Ask a single question
python -m app.cli "What is my bill for January 2026?"

# Or use interactive mode (with session memory!)
python -m app.cli --interactive
```

---

## Project Structure

```
telecom-billing-agentic-ai/
|
+-- app/                          # Main application code
|   +-- __init__.py
|   +-- config.py                 # Configuration & environment
|   +-- ingest.py                 # Document ingestion script
|   +-- cli.py                    # Command-line interface
|   +-- graph.py                  # LangGraph workflow definition
|   |
|   +-- agents/                   # Agent implementations
|   |   +-- sales.py              # SalesAgent (front desk)
|   |   +-- billing.py            # BillingAgent (RAG-powered)
|   |   +-- manager.py            # ManagerAgent (validator)
|   |
|   +-- memory/                   # NEW: Session & Entity Memory
|   |   +-- session_store.py      # In-memory session storage
|   |   +-- entity_extractor.py   # Extract account, name, etc.
|   |
|   +-- rag/                      # RAG components
|   |   +-- pinecone_store.py     # Pinecone vector DB wrapper
|   |   +-- retriever.py          # Document retrieval logic
|   |
|   +-- utils/                    # Utilities
|       +-- guardrails.py         # Safety validation rules
|       +-- logging.py            # Colored console output
|
+-- data/
|   +-- docs/                     # 5 demo documents
|       +-- DOC_1_PLANS_AND_PRICING.txt
|       +-- DOC_2_BILLING_FAQ.txt
|       +-- DOC_3_CUSTOMER_PROFILE_DILEEP_DEMO_001.txt
|       +-- DOC_4_INVOICE_JAN_2026_DILEEP_DEMO_001.txt
|       +-- DOC_5_POLICIES_DISPUTE_LATEFEES.txt
|
+-- .env.example                  # Environment template
+-- requirements.txt              # Python dependencies
+-- pyproject.toml                # Project configuration
+-- README.md                     # This file
```

---

## Demo Questions

These three questions showcase the system's capabilities:

### 1. "How much is my bill for January 2026?"
- **Router**: Classifies as `billing_account_specific`
- **BillingAgent**: Retrieves DOC_4 (invoice), finds $137.14
- **ManagerAgent**: Validates citations, approves
- **Expected Answer**: "$137.14 with line-item breakdown"

### 2. "Why is my bill higher this month?"
- **Router**: Classifies as `billing_account_specific`
- **BillingAgent**: Retrieves invoice + customer profile, identifies international call overage
- **ManagerAgent**: Validates the $8.75 overage is cited
- **Expected Answer**: "Higher due to 35 minutes of international calls to India ($8.75)"

### 3. "What is the due date and what happens if I pay late?"
- **Router**: Classifies as `billing_account_specific`
- **BillingAgent**: Retrieves invoice (due date) + policy doc (late fees)
- **ManagerAgent**: Validates both are cited
- **Expected Answer**: "January 30, 2026. Late fees: $10 after 15 days, $25 total after 30 days..."

---

## Guardrails Explained

### 1. SalesAgent Guardrail
**Rule**: Cannot output dollar amounts for account-specific queries

```python
# Blocked: "Your bill is $137.14"
# Allowed: "The Pro plan costs $49.99/month" (general pricing)
```

### 2. BillingAgent Guardrail
**Rule**: Must output structured response with citations

```json
{
  "answer": "Your bill is $137.14",
  "citations": [
    {"doc_id": "DOC_4_INVOICE...", "chunk_id": 1, "quote": "TOTAL AMOUNT DUE: $137.14"}
  ],
  "top_score": 0.89
}
```

### 3. ManagerAgent Guardrail
**Three checks**:
1. ✅ Citations must not be empty
2. ✅ Confidence score ≥ 0.75
3. ✅ Dollar amounts in answer must appear in quotes

If any check fails, the response is rejected with clarifying questions.

---

## Configuration Options

### Required API Keys

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `OPENAI_API_KEY` | (required) | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model to use |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `PINECONE_API_KEY` | (required) | Your Pinecone API key |
| `PINECONE_INDEX_NAME` | `telecom-billing-demo` | Index name |
| `PINECONE_NAMESPACE` | `telecom-docs` | Namespace for docs |

### LangSmith Tracing (Optional)

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `LANGCHAIN_TRACING_V2` | `false` | Set to `true` to enable tracing |
| `LANGCHAIN_API_KEY` | (optional) | Your LangSmith API key |
| `LANGCHAIN_PROJECT` | `telecom-billing-demo` | Project name in LangSmith |

LangSmith provides full observability into your agent runs - see every LLM call, token usage, and latency.

---

## Session Memory (Interactive Mode)

The system includes **entity memory** that remembers context across queries in interactive mode:

### How It Works

```
You: "My account is ACC-789456123"
     -> System extracts and stores: account_id = "ACC-789456123"

You: "What's my bill?"
     -> System remembers your account, uses it for better RAG search
     -> Answer: "$137.14 for account ACC-789456123"

You: "Why is it higher?"
     -> System maintains context, knows you're asking about the same bill
```

### Using Interactive Mode

```bash
python -m app.cli --interactive
```

Special commands:
- `session` - View current session context
- `exit` or `quit` - End session (shows summary)

### Entities Extracted

| Entity | Pattern | Example |
|--------|---------|---------|
| Account ID | `ACC-XXXXXXXXX` | ACC-789456123 |
| Customer Name | "I'm [Name]" | "Dileep" |
| Billing Period | "January 2026" | "January 2026" |
| Topic | Keywords detected | "billing", "dispute" |

### Storage

Currently uses **in-memory storage** (data lost on restart). For production, you would use:
- Redis for distributed sessions
- SQLite/PostgreSQL for persistence
- LangGraph MemorySaver for checkpointing

---

## Architecture Decisions

### Why LangGraph (not just LangChain)?
- LangChain = building blocks for LLM apps
- LangGraph = **stateful workflows** with conditional routing
- Our flow requires: Router → Agent → Manager → Response (with state)

### Why Pinecone?
- Serverless = no infrastructure to manage
- Fast semantic search for RAG
- Metadata filtering for precise retrieval

### Why Guardrails are Rule-Based (not AI)?
- Deterministic = predictable, auditable
- Faster = no LLM call needed
- Reliable = no hallucination in validation

---

## For Developers

### Key Concepts to Understand

1. **Embeddings**: Text → Vector of numbers (meaning representation)
2. **Vector Search**: Find similar vectors = find similar meanings
3. **RAG**: Retrieve docs → Add to prompt → Generate grounded response
4. **Agents**: LLM + Role + Rules + Tools
5. **Guardrails**: Deterministic checks for safety

### Extending the System

**Add a new document type:**
1. Add `.txt` file to `data/docs/`
2. Re-run `python -m app.ingest`

**Adjust confidence threshold:**
- Edit `CONFIDENCE_THRESHOLD` in `app/config.py`

**Add a new agent:**
1. Create `app/agents/new_agent.py`
2. Add node to `app/graph.py`
3. Wire up edges

---

## License

MIT License - This is a demo/educational project.

---

## Acknowledgments

Built with:
- [LangChain](https://langchain.com/) - LLM application framework
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent orchestration
- [Pinecone](https://pinecone.io/) - Vector database
- [OpenAI](https://openai.com/) - LLM and embeddings

---

*Happy coding!*
