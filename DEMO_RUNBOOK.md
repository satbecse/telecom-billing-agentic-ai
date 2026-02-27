# Telecom AI - Demo Runbook

This guide provides a step-by-step script for demonstrating the Agentic AI telecom billing assistant to your manager.

## Preparation Before Demo
Ensure your terminal is open, you have activated your virtual environment, and your `.env` file is loaded with your OpenAI and Pinecone API keys.
```powershell
# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

> **Troubleshooting:** If you get a "running scripts is disabled" error, run this command first to temporarily allow scripts in your current session:
> ```powershell
> Set-ExecutionPolicy Bypass -Scope Process -Force
> ```

---

## 🎭 Mode 1: Interactive Conversational AI
**Goal:** Show off multi-agent routing (Sales vs Billing), Session Memory (Entity extraction), and RAG retrieving from customer PDFs.

**Script:**
1. **Start the interactive mode with our best RAG strategy (HyDE):**
   ```powershell
   python -m app.cli --interactive --rag-strategy hyde
   ```
2. **Set context (Entity Memory):**
   > **You type:** `Hi, I am Dileep.`
   > **System does:** Extracts the name `Dileep` and stores it in session memory.

3. **Ask a billing question (BillingAgent + RAG):**
   > **You type:** `What is my bill for January 2026?`
   > **System does:** Router sends to BillingAgent -> Enhances query with "Dileep" -> RAG retrieves DOC_4 -> Manager validates $ amounts -> Answers `$137.14`.

4. **Ask a follow-up question (Session Memory):**
   > **You type:** `Why is it higher than normal?`
   > **System does:** Remembers we are taking about Dileep's January 2026 bill -> BillingAgent finds the $8.75 international calling overage.

5. **Type `exit`** to leave interactive mode.

---

## 🔍 Mode 2: RAG Strategy Comparison & Wikipedia Knowledge
**Goal:** Show your manager how the CLI can switch RAG strategies on the fly, and demonstrate the SalesAgent using the Wikipedia knowledge base for AT&T queries.

**Script:**
1. **Explain Naive RAG:** "First, let's ask a complex AT&T question using the standard Naive RAG approach."
   ```powershell
   python -m app.cli "When was AT&T founded?" --rag-strategy naive
   ```
   *Watch the trace: Router sends to SalesAgent -> SalesAgent searches the `telecom-wiki` namespace -> Retrieves chunks -> Answers March 3, 1885.*

2. **Explain HyDE RAG:** "Now let's ask a complex billing question using HyDE (Hypothetical Document Embeddings), where the LLM guesses the answer first to improve search."
   ```powershell
   python -m app.cli "How does the late fee work for AT&T bills?" --rag-strategy hyde
   ```
   *Watch the trace: See the BillingAgent generate a "HyDE hypothesis" first, then retrieve the policies document.*

---

## 📊 Mode 3: The Evaluation Framework (The "Manager Impresser")
**Goal:** Prove that the architectural choices (HyDE + Fixed-Size Chunking) were scientifically tested and data-driven, rather than just guessed.

**Script:**
1. **Explain the Evaluation:** "To find the best retrieval architecture, I built an LLM-as-a-judge evaluation framework. It runs 90 tests: 3 RAG Strategies x 3 Chunking Techniques x 10 diverse queries."
2. **Run the evaluation script:**
   ```powershell
   python -m app.evaluation.eval_runner
   ```
   *(Note: This takes about 2 minutes to run live. You can let it run while you explain the next part, or just open the pre-generated report).*
3. **Open the Report:**
   ```powershell
   # Windows PowerShell
   notepad data\evaluation_results\comparison_report.txt
   # Or just open it in your VS Code/Editor
   ```
4. **Highlight the Key Findings to your Manager:**
   - **Fixed-Size Chunking Won (0.853 Avg):** Explain that Semantic Chunking actually performed worst (0.463) because it over-split the structured invoices into 970 tiny fragments, destroying the context.
   - **HyDE RAG Won:** Explain that generating a hypothetical answer first created embeddings that more closely matched the dense terminology found in actual telecom billing documents.
   - **Per-Query Breakdown:** Show how the basic customer queries scored near 1.00, while the AT&T Wikipedia queries were the main differentiators.
