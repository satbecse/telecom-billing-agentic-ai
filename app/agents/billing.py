"""
BillingAgent - The account specialist with RAG capabilities.

CONCEPT: RAG-Powered Agent
==========================
Unlike SalesAgent which answers from general knowledge,
BillingAgent uses RAG (Retrieval-Augmented Generation):

1. RETRIEVE: Search Pinecone for relevant document chunks
2. AUGMENT: Add these chunks to the LLM prompt
3. GENERATE: Create an answer grounded in real data

This ensures BillingAgent's answers are:
- Based on actual account data
- Verifiable with citations
- Not hallucinated
"""

import json
from typing import Dict, List, Tuple, Optional
from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.rag.retriever import TelecomRetriever
from app.utils.logging import get_logger, print_agent_action
from app.utils.guardrails import GuardrailsChecker

logger = get_logger("billing_agent")


# BillingAgent's system prompt - note how it emphasizes citations
BILLING_AGENT_SYSTEM_PROMPT = """You are a TelcoMax Wireless Billing Specialist with access to customer account data.

## Your Role
- Answer specific billing and account questions
- Provide accurate information based ONLY on retrieved documents
- Always cite your sources

## CRITICAL RULES
1. **ONLY use information from the provided documents**
   - Never make up or estimate amounts
   - If information isn't in the documents, say "not found"

2. **ALWAYS provide citations**
   - Reference the document ID and chunk for every fact
   - Include a short quote (≤20 words) as evidence

3. **For billing amounts, ALWAYS include:**
   - The exact dollar amount from the document
   - The billing period it applies to
   - Any relevant breakdowns if available

4. **If you cannot answer:**
   - Clearly state what information is missing
   - Suggest what information the customer could provide

## Response Format
You MUST respond in this exact JSON format:
{
    "answer": "Your detailed answer here with specific amounts and dates",
    "citations": [
        {
            "doc_id": "DOC_X_NAME",
            "chunk_id": "chunk number",
            "quote": "exact quote from document (max 20 words)"
        }
    ],
    "confidence_note": "Brief note on how confident you are in this answer"
}
"""


class BillingAgent:
    """
    BillingAgent handles account-specific billing queries using RAG.
    
    Workflow:
    1. Receive query from SalesAgent (via Router)
    2. Retrieve relevant document chunks from Pinecone
    3. Generate answer with citations
    4. Format response for ManagerAgent validation
    """
    
    def __init__(self):
        """Initialize BillingAgent with RAG components."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.retriever = TelecomRetriever()
        self.guardrails = GuardrailsChecker()
        self.system_prompt = BILLING_AGENT_SYSTEM_PROMPT
    
    def process_query(self, query: str, session_context: str = "") -> Dict:
        """
        Process a billing query using RAG.
        
        This is the main entry point for billing queries.
        
        Args:
            query: The customer's billing question
            session_context: Optional context from session memory (e.g., account ID)
        
        Returns:
            Dict with answer, citations, and top_score
        """
        print_agent_action("BillingAgent", "Processing billing query...")
        
        # Step 1: RETRIEVE relevant documents
        # If we have session context (like account ID), combine with query for better retrieval
        search_query = query
        if session_context:
            # Add context to improve retrieval (e.g., helps find correct account docs)
            search_query = f"{session_context} {query}"
            print_agent_action("BillingAgent", f"Enhanced query with session context")
        
        print_agent_action("BillingAgent", "Retrieving relevant documents from Pinecone...")
        chunks, top_score = self.retriever.retrieve(search_query)
        
        if not chunks:
            return self._create_not_found_response(query)
        
        # Step 2: AUGMENT - Format context for LLM
        context = self.retriever.format_context_for_llm(chunks)
        
        print_agent_action(
            "BillingAgent", 
            f"Found {len(chunks)} relevant chunks. Top score: {top_score:.3f}"
        )
        
        # Step 3: GENERATE response with citations
        # Pass session context to help LLM understand the conversation
        response = self._generate_response(query, context, chunks, session_context)
        
        # Add the top score for ManagerAgent validation
        response["top_score"] = top_score
        
        return response
    
    def _generate_response(
        self,
        query: str,
        context: str,
        chunks: List[Dict],
        session_context: str = ""
    ) -> Dict:
        """
        Generate a response using the LLM with retrieved context.
        
        Args:
            query: Customer's question
            context: Formatted context from retrieved documents
            chunks: Raw chunk data for reference
            session_context: Context from session memory (account, name, etc.)
        
        Returns:
            Dict with answer, citations, and metadata
        """
        print_agent_action("BillingAgent", "Generating response with citations...")
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "system", "content": f"## Retrieved Documents:\n{context}"},
        ]
        
        # NEW: Add session context if available
        if session_context:
            messages.append({
                "role": "system", 
                "content": f"## Customer Context:\n{session_context}\nUse this context to understand which customer/account this query is about."
            })
        
        messages.append({"role": "user", "content": query})
        
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.3,  # Lower temperature for factual accuracy
            max_tokens=1000
        )
        
        response_text = response.choices[0].message.content
        
        # Parse the JSON response
        try:
            # Try to extract JSON from the response
            response_data = self._parse_json_response(response_text)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            # Create a structured response from unstructured text
            response_data = self._create_fallback_response(response_text, chunks)
        
        # Validate response structure
        validation = self.guardrails.validate_billing_response_structure(response_data)
        if not validation.is_valid:
            logger.warning(f"Response structure invalid: {validation.reason}")
            response_data = self._create_fallback_response(response_text, chunks)
        
        return response_data
    
    def _parse_json_response(self, response_text: str) -> Dict:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Remove markdown code blocks if present
        text = response_text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        
        return json.loads(text.strip())
    
    def _create_fallback_response(
        self,
        response_text: str,
        chunks: List[Dict]
    ) -> Dict:
        """
        Create a structured response when JSON parsing fails.
        
        This ensures we always have a valid response structure
        even if the LLM doesn't follow the format perfectly.
        """
        # Create citations from chunks
        citations = self.retriever.create_citations_from_chunks(chunks, response_text)
        
        return {
            "answer": response_text,
            "citations": citations[:3],  # Limit to top 3 citations
            "confidence_note": "Response was reformatted for validation."
        }
    
    def _create_not_found_response(self, query: str) -> Dict:
        """Create a response when no relevant documents are found."""
        return {
            "answer": (
                "I couldn't find specific information about your account "
                "in our records. This might be because:\n"
                "- The account number or details weren't found\n"
                "- The billing period mentioned isn't available\n"
                "Please verify your account information."
            ),
            "citations": [],
            "top_score": 0.0,
            "confidence_note": "No relevant documents found.",
            "clarifying_questions": [
                "Can you confirm your account number?",
                "Which billing period are you asking about?"
            ]
        }
    
    def format_for_manager(self, response: Dict) -> Dict:
        """
        Format the response for ManagerAgent validation.
        
        This ensures all required fields are present and properly formatted.
        
        Args:
            response: The BillingAgent's response
        
        Returns:
            Formatted dict ready for ManagerAgent
        """
        return {
            "answer": response.get("answer", ""),
            "citations": response.get("citations", []),
            "top_score": response.get("top_score", 0.0),
            "confidence_note": response.get("confidence_note", ""),
            "raw_response": response
        }
