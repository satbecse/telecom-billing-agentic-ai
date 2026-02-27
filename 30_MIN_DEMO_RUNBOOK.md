# Telecom Billing Agentic AI - 30 Minute Demo Runbook

**Objective**: Showcase the enterprise-grade features of the AI system, including LangGraph orchestration, RAG techniques, persistent memory, and LLM-as-a-judge evaluation.

---

## Pre-Demo Checklist (5 Mins Before)
1. **Clear Evaluation Namespaces**: Go to [Pinecone Dashboard](https://app.pinecone.io/) and delete `eval-fixed`, `eval-recursive`, and `eval-semantic` namespaces if they exist. (Leave `telecom-docs` and `telecom-wiki` alone).
2. **Activate Environment**:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
3. **Open DB Browser** and have `data/sessions.db` ready to view.
4. **Open LangSmith** dashboard in your browser.

---

## ⏱️ Step 1: Architecture Overview (5 mins)
*Display the `README.md` diagrams.*

* Explain the **Non-Technical View** first: "The Sales Agent acts as our Front Desk receptionist. It handles general public knowledge questions itself, but transfers private account questions to the Billing department."
* Explain the **Technical View**: "Notice we separate our vector databases (Pinecone). Sales only reads Wikipedia. Billing only reads PDF chunks. The Manager Agent protects the system by verifying the Billing Agent's math and citations."

## ⏱️ Step 2: Code Walkthrough (5 mins)
Briefly show the structure in Visual Studio Code:
* `app/graph.py` - Show the LangGraph node setup
* `app/chunking/` - Mention we have 3 pluggable chunkers
* `data/docs/customer_pdfs/` - Show what a raw bill looks like.

## ⏱️ Step 3: Interactive Demo & Memory (10 mins)
*Run the application in interactive mode, using the new `--session-id` flag and `hyde` strategy (best for PDFs).*

```powershell
python -m app.cli --interactive --session-id demo-run --rag-strategy hyde
```

**Interaction 1: Sales Agent (Wikipedia RAG)**
> **You:** `When was AT&T founded?`
* Explain: "The Sales Agent recognizes this as a general knowledge question about AT&T and answers it directly by querying the Wikipedia namespace."

**Interaction 2: Billing Agent (Customer PDF RAG)**
> **You:** `What is the bill for customer ACC-DEMO-001?`
* Explain: "The Sales Agent sees this requires private account data, so it extracts the account ID and transfers the question to the Billing Agent. The Manager Agent verifies the $137.14 amount against the PDF citation, and returns it safely."

**Interaction 3: Proving SQLite Memory Persistence**
> **You:** `exit`

*Run the exact same command to rejoin the session:*
```powershell
python -m app.cli --interactive --session-id demo-run --rag-strategy hyde
```
> **You:** `What did I just ask you about?`
* Explain: "The system remembers us! It recalled the previous context from our SQLite database."

**Show Behind the Scenes:**
* Show `data/sessions.db` in SQLite DB Browser (show the `conversation_turns` table).
* Show LangSmith trace for the query.

## ⏱️ Step 4: The Evaluation Framework (8 mins)
*Run the intensive 90-step evaluator.*

```powershell
python -m app.evaluation.eval_runner
```

*While it runs (takes ~2.5 mins), explain the code:*
1. "The script is running 10 questions across 3 chunkers and 3 RAG strategies (90 total LLM calls)."
2. "It's using **LLM-as-a-judge** to grade Faithfulness, Relevancy, and Correctness against a ground truth."
3. Open `app/agents/billing.py` (Lines 164-180) to explain how **HyDE** works (generating a fake answer to find a real one).
4. Open `app/utils/guardrails.py` to explain how we use Regex to protect the customer from hallucinated dollar amounts.

## ⏱️ Step 5: Evaluation Results & Winner (2 mins)
*Once the evaluation finishes:*
1. Open the new `data/evaluation_results/comparison_report_YYYYMMDD.txt` file.
2. Scroll to the "Top 3 Configurations" and declare the winner to the audience. Explain *why* that specific configuration won (e.g., Fixed-Size chunking preserved telecom tables better than semantic splitting).
