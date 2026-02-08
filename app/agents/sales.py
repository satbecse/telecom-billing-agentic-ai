"""
SalesAgent - The friendly front-desk of our telecom service.

CONCEPT: What is an "Agent"?
============================
An agent is an LLM (like GPT-4) given:
1. A specific ROLE (persona)
2. A set of RULES (what it can/cannot do)
3. TOOLS it can use (in our case, routing to other agents)

SalesAgent's Role:
- First point of contact for all customer queries
- Handles general questions about plans, pricing, policies
- Routes billing-specific questions to BillingAgent
- Never reveals specific account amounts (guardrail enforced)
"""

from typing import Dict, Optional, Tuple
from openai import OpenAI

from app.config import OPENAI_API_KEY, OPENAI_MODEL, QueryIntent
from app.utils.logging import get_logger, print_agent_action
from app.utils.guardrails import GuardrailsChecker

logger = get_logger("sales_agent")


# The SalesAgent's "personality" and rules - this is the SYSTEM PROMPT
SALES_AGENT_SYSTEM_PROMPT = """You are a friendly and professional TelcoMax Wireless customer service representative.

## Your Role
- Greet customers warmly and help with general inquiries
- Answer questions about plans, pricing, features, and policies
- Be helpful, empathetic, and professional

## CRITICAL RULES (You MUST follow these)
1. **NEVER provide specific dollar amounts for individual customer bills or balances**
   - You can discuss general pricing (e.g., "The Pro plan is $49.99/month")
   - You CANNOT say things like "Your bill is $137.14" or "You owe $50"
   
2. **Route billing-specific questions to the Billing Department**
   - If a customer asks about THEIR specific bill, charges, or account balance
   - Politely explain you're transferring them to billing support
   
3. **General information you CAN provide:**
   - Plan prices and features
   - Add-on service costs
   - General billing policies
   - Late fee policies (in general terms)
   - How to dispute charges

## Response Style
- Be concise but friendly
- Use simple language
- If you don't know something, say so
- Always offer to help further
"""


class SalesAgent:
    """
    The SalesAgent handles initial customer contact and general inquiries.
    
    Key behaviors:
    - Answers general questions about plans, pricing, policies
    - Blocks attempts to reveal account-specific billing amounts
    - Routes billing-specific queries to BillingAgent
    """
    
    def __init__(self):
        """Initialize the SalesAgent with OpenAI client."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set.")
        
        self.openai = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_MODEL
        self.guardrails = GuardrailsChecker()
        self.system_prompt = SALES_AGENT_SYSTEM_PROMPT
    
    def classify_query(self, query: str) -> str:
        """
        Classify the user's query intent.
        
        CONCEPT: Query Classification
        ==============================
        Before routing, we need to understand WHAT the user is asking.
        This helps us decide which agent should handle the request.
        
        Categories:
        - billing_account_specific: Questions about THEIR bill/account
        - billing_general: General questions about billing processes
        - sales_general: Questions about plans, pricing, features
        
        Args:
            query: The user's question
        
        Returns:
            Intent classification string
        """
        classification_prompt = f"""Classify this customer query into ONE of these categories:

1. "billing_account_specific" - Questions about the customer's OWN bill, charges, amounts, or account balance
   Examples: "What's my bill?", "How much do I owe?", "Why was I charged $X?"

2. "billing_general" - Questions about billing PROCESSES in general (not their specific account)
   Examples: "How does proration work?", "When are bills generated?", "What payment methods do you accept?"

3. "sales_general" - Questions about plans, pricing, features, or general policies
   Examples: "What plans do you offer?", "How much is the Pro plan?", "Do you have international calling?"

Customer Query: "{query}"

Respond with ONLY the category name, nothing else."""

        response = self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a query classifier. Respond with only the category name."},
                {"role": "user", "content": classification_prompt}
            ],
            temperature=0,  # Deterministic classification
            max_tokens=50
        )
        
        intent = response.choices[0].message.content.strip().lower()
        
        # Normalize to our defined intents
        if "billing_account_specific" in intent:
            return QueryIntent.BILLING_ACCOUNT_SPECIFIC
        elif "billing_general" in intent:
            return QueryIntent.BILLING_GENERAL
        else:
            return QueryIntent.SALES_GENERAL
    
    def generate_response(
        self,
        query: str,
        context: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Generate a response to the customer query.
        
        Args:
            query: The customer's question
            context: Optional additional context (e.g., from previous agents)
        
        Returns:
            Tuple of (response text, needs_billing_routing)
        """
        print_agent_action("SalesAgent", "Generating response...")
        
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        if context:
            messages.append({
                "role": "system",
                "content": f"Additional context from billing department:\n{context}"
            })
        
        messages.append({"role": "user", "content": query})
        
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content
        
        # Check if the response needs routing (contains billing-specific info)
        needs_routing = self._check_needs_billing_routing(query, response_text)
        
        return response_text, needs_routing
    
    def _check_needs_billing_routing(self, query: str, response: str) -> bool:
        """
        Check if this query should be routed to BillingAgent.
        
        Uses guardrails to detect if SalesAgent is trying to
        output billing-specific information.
        """
        intent = self.classify_query(query)
        
        if intent == QueryIntent.BILLING_ACCOUNT_SPECIFIC:
            # This should go to billing
            return True
        
        # Also check if response contains dollar amounts for account-specific queries
        validation = self.guardrails.check_sales_agent_response(response, intent)
        if not validation.is_valid:
            logger.warning(f"SalesAgent guardrail triggered: {validation.reason}")
            return True
        
        return False
    
    def create_handoff_message(self, query: str) -> str:
        """
        Create a friendly handoff message when routing to BillingAgent.
        
        Args:
            query: The original customer query
        
        Returns:
            Handoff message for the customer
        """
        return (
            "I'd be happy to help you with your billing question! "
            "Let me connect you with our billing support team who can "
            "access your account details. One moment please..."
        )
    
    def format_final_response(
        self,
        billing_response: Dict,
        approved: bool
    ) -> str:
        """
        Format the final response after billing agent processing.
        
        This is called when the workflow completes and we need to
        present the final answer to the customer.
        
        Args:
            billing_response: Response data from BillingAgent
            approved: Whether ManagerAgent approved the response
        
        Returns:
            Customer-friendly final response
        """
        if not approved:
            # Manager rejected - return clarifying questions
            return (
                "I apologize, but I couldn't find enough information to answer "
                "your question precisely. " + 
                billing_response.get("clarifying_message", 
                    "Could you please provide more details about your account?")
            )
        
        # Format approved response
        answer = billing_response.get("answer", "")
        citations = billing_response.get("citations", [])
        
        # Build response with citations
        response_parts = [answer]
        
        if citations:
            response_parts.append("\n\n📋 **Sources:**")
            for i, cite in enumerate(citations, 1):
                response_parts.append(
                    f"  [{i}] {cite.get('doc_id', 'Unknown')} - "
                    f"\"{cite.get('quote', '')[:50]}...\""
                )
        
        return "\n".join(response_parts)
