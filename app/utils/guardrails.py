"""
Guardrails module for the Telecom Billing Agentic AI system.

This module implements safety checks to ensure:
1. SalesAgent doesn't output billing-specific dollar amounts
2. BillingAgent citations are valid and present
3. ManagerAgent can validate answers against evidence

CONCEPT: Guardrails are "safety rails" that prevent the AI from:
- Hallucinating (making up) billing amounts
- Providing unverified information
- Bypassing the approval workflow
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

from app.config import CONFIDENCE_THRESHOLD, MAX_QUOTE_WORDS, STRICT_AMOUNT_VERIFICATION


@dataclass
class ValidationResult:
    """Result of a guardrail validation check."""
    is_valid: bool
    reason: str
    details: Dict = None


class GuardrailsChecker:
    """
    Implements guardrail checks for the multi-agent system.
    
    Guardrails are deterministic (rule-based) checks, not AI-based.
    This ensures consistent, predictable behavior.
    """
    
    # Regex pattern to find dollar amounts (e.g., $137.14, $5, $5.00)
    DOLLAR_PATTERN = re.compile(r'\$[\d,]+(?:\.\d{2})?')
    
    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        """
        Initialize the guardrails checker.
        
        Args:
            confidence_threshold: Minimum similarity score for approval (default: 0.75)
        """
        self.confidence_threshold = confidence_threshold
    
    # =========================================================================
    # SALES AGENT GUARDRAILS
    # =========================================================================
    
    def check_sales_agent_response(self, 
                                   response: str, 
                                   query_intent: str) -> ValidationResult:
        """
        Check if SalesAgent response is appropriate.
        
        RULE: SalesAgent must NOT output dollar amounts for billing-specific queries.
        
        Args:
            response: The SalesAgent's proposed response
            query_intent: The classified intent of the user's query
        
        Returns:
            ValidationResult indicating if the response is valid
        """
        # Find all dollar amounts in the response
        dollar_amounts = self.DOLLAR_PATTERN.findall(response)
        
        # If this is a billing-account-specific query, block dollar amounts
        if query_intent == "billing_account_specific" and dollar_amounts:
            return ValidationResult(
                is_valid=False,
                reason="SalesAgent cannot provide specific billing amounts. Routing to BillingAgent.",
                details={"blocked_amounts": dollar_amounts}
            )
        
        return ValidationResult(is_valid=True, reason="Response is appropriate for SalesAgent.")
    
    # =========================================================================
    # BILLING AGENT GUARDRAILS
    # =========================================================================
    
    def validate_billing_response_structure(self, 
                                            response_data: Dict) -> ValidationResult:
        """
        Validate that BillingAgent response has required structure.
        
        Required fields:
        - answer: The response text
        - citations: List of citation objects
        - top_score: Highest similarity score from retrieval
        
        Args:
            response_data: The structured response from BillingAgent
        
        Returns:
            ValidationResult indicating if structure is valid
        """
        required_fields = ["answer", "citations", "top_score"]
        
        missing = [f for f in required_fields if f not in response_data]
        if missing:
            return ValidationResult(
                is_valid=False,
                reason=f"Missing required fields: {', '.join(missing)}",
                details={"missing_fields": missing}
            )
        
        # Validate citations structure
        citations = response_data.get("citations", [])
        if not isinstance(citations, list):
            return ValidationResult(
                is_valid=False,
                reason="Citations must be a list",
                details={"citations_type": type(citations).__name__}
            )
        
        for i, citation in enumerate(citations):
            required_citation_fields = ["doc_id", "chunk_id", "quote"]
            missing_cite = [f for f in required_citation_fields if f not in citation]
            if missing_cite:
                return ValidationResult(
                    is_valid=False,
                    reason=f"Citation {i} missing fields: {', '.join(missing_cite)}",
                    details={"citation_index": i, "missing": missing_cite}
                )
        
        return ValidationResult(is_valid=True, reason="Response structure is valid.")
    
    # =========================================================================
    # MANAGER AGENT GUARDRAILS
    # =========================================================================
    
    def manager_validate_response(self, 
                                   answer: str,
                                   citations: List[Dict],
                                   top_score: float) -> ValidationResult:
        """
        ManagerAgent validation of BillingAgent response.
        
        Checks:
        1. Citations are not empty
        2. Confidence score meets threshold
        3. Dollar amounts in answer appear in citation quotes
        
        Args:
            answer: The proposed answer text
            citations: List of citation objects with doc_id, chunk_id, quote
            top_score: The highest similarity score from retrieval
        
        Returns:
            ValidationResult with approval status and reason
        """
        # CHECK 1: Citations must not be empty
        if not citations:
            return ValidationResult(
                is_valid=False,
                reason="No citations provided. Cannot verify answer without evidence.",
                details={"check": "citations_present", "needed": "At least one citation required"}
            )
        
        # CHECK 2: Confidence score must meet threshold
        if top_score < self.confidence_threshold:
            return ValidationResult(
                is_valid=False,
                reason=f"Confidence too low ({top_score:.2f} < {self.confidence_threshold}). "
                       f"Please provide more specific information.",
                details={
                    "check": "confidence_threshold",
                    "score": top_score,
                    "threshold": self.confidence_threshold,
                    "clarifying_questions": [
                        "Can you provide your account number?",
                        "Which billing period are you asking about?",
                        "Can you provide more details about your question?"
                    ]
                }
            )
        
        # CHECK 3: Dollar amounts in answer must appear in quotes
        # (Only enforce if STRICT_AMOUNT_VERIFICATION is enabled)
        if STRICT_AMOUNT_VERIFICATION:
            dollar_in_answer = set(self.DOLLAR_PATTERN.findall(answer))
            
            if dollar_in_answer:
                # Collect all dollar amounts from citation quotes
                dollar_in_quotes = set()
                for citation in citations:
                    quote = citation.get("quote", "")
                    dollar_in_quotes.update(self.DOLLAR_PATTERN.findall(quote))
                
                # Find amounts in answer but not in quotes (potential hallucinations)
                unverified = dollar_in_answer - dollar_in_quotes
                
                if unverified:
                    return ValidationResult(
                        is_valid=False,
                        reason=f"Dollar amounts {unverified} in answer not found in source documents.",
                        details={
                            "check": "amounts_verified",
                            "unverified_amounts": list(unverified),
                            "verified_amounts": list(dollar_in_quotes)
                        }
                    )
        
        # Get amounts for the response details
        dollar_in_answer = set(self.DOLLAR_PATTERN.findall(answer))
        
        # All checks passed
        return ValidationResult(
            is_valid=True,
            reason="Response approved. Citations present, confidence sufficient.",
            details={
                "citations_count": len(citations),
                "confidence_score": top_score,
                "amounts_in_answer": list(dollar_in_answer) if dollar_in_answer else []
            }
        )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def extract_dollar_amounts(self, text: str) -> List[str]:
        """Extract all dollar amounts from text."""
        return self.DOLLAR_PATTERN.findall(text)
    
    def truncate_quote(self, quote: str, max_words: int = MAX_QUOTE_WORDS) -> str:
        """Truncate a quote to maximum word count."""
        words = quote.split()
        if len(words) <= max_words:
            return quote
        return " ".join(words[:max_words]) + "..."
    
    def generate_clarifying_questions(self, missing_info: List[str]) -> List[str]:
        """Generate clarifying questions based on what information is missing."""
        question_templates = {
            "account": "Can you provide your account number or phone number?",
            "period": "Which billing period are you asking about (e.g., January 2026)?",
            "charge": "Can you specify which charge you have questions about?",
            "customer": "Can you confirm your name or account email?",
        }
        
        questions = []
        for info in missing_info:
            if info.lower() in question_templates:
                questions.append(question_templates[info.lower()])
            else:
                questions.append(f"Can you provide more information about {info}?")
        
        return questions if questions else ["Can you provide more details about your question?"]
