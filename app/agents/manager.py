"""
ManagerAgent - The quality control gatekeeper.

CONCEPT: Why Do We Need a Manager?
==================================
In production systems, we can't just trust LLM outputs blindly.
ManagerAgent acts as a "human-like" reviewer that:

1. VALIDATES: Are there citations? Is confidence high enough?
2. VERIFIES: Do the dollar amounts in the answer match the sources?
3. GATES: Either approves the response or requests clarification

This is a critical GUARDRAIL against hallucinations and errors.

The Manager uses DETERMINISTIC rules, not LLM judgment, for:
- Citation presence (must have at least 1)
- Confidence threshold (must be >= 0.75)
- Amount verification (dollars in answer must appear in quotes)
"""

from typing import Dict, List, Tuple, Optional

from app.config import CONFIDENCE_THRESHOLD
from app.utils.logging import get_logger, print_agent_action, print_decision
from app.utils.guardrails import GuardrailsChecker, ValidationResult

logger = get_logger("manager_agent")


class ManagerAgent:
    """
    ManagerAgent validates BillingAgent responses before approval.
    
    This is a DETERMINISTIC agent - it doesn't use an LLM.
    Instead, it applies strict rule-based validation.
    
    Validation Rules:
    1. Citations must not be empty
    2. Confidence score must meet threshold (default 0.75)
    3. Dollar amounts in answer must appear in citation quotes
    """
    
    def __init__(self, confidence_threshold: float = CONFIDENCE_THRESHOLD):
        """
        Initialize ManagerAgent with validation threshold.
        
        Args:
            confidence_threshold: Minimum required confidence score
        """
        self.confidence_threshold = confidence_threshold
        self.guardrails = GuardrailsChecker(confidence_threshold)
    
    def validate_response(self, billing_response: Dict) -> Tuple[bool, Dict]:
        """
        Validate a BillingAgent response.
        
        This is the main entry point for the manager workflow.
        
        Args:
            billing_response: Dict with answer, citations, top_score
        
        Returns:
            Tuple of (is_approved, result_dict)
            
            result_dict contains:
            - approved: bool
            - reason: str explaining the decision
            - response: the original or modified response
            - clarifying_questions: list if rejected
        """
        print_agent_action("ManagerAgent", "Validating BillingAgent response...")
        
        # Extract fields from billing response
        answer = billing_response.get("answer", "")
        citations = billing_response.get("citations", [])
        top_score = billing_response.get("top_score", 0.0)
        
        # Log what we're validating
        print_agent_action(
            "ManagerAgent",
            f"Checking: {len(citations)} citations, score={top_score:.3f}"
        )
        
        # Run guardrails validation
        validation = self.guardrails.manager_validate_response(
            answer=answer,
            citations=citations,
            top_score=top_score
        )
        
        if validation.is_valid:
            # APPROVED
            print_decision("Response APPROVED", approved=True)
            return True, self._create_approved_response(billing_response, validation)
        else:
            # REJECTED
            print_decision(f"Response REJECTED: {validation.reason}", approved=False)
            return False, self._create_rejected_response(billing_response, validation)
    
    def _create_approved_response(
        self,
        billing_response: Dict,
        validation: ValidationResult
    ) -> Dict:
        """Create the approved response payload."""
        return {
            "approved": True,
            "reason": validation.reason,
            "answer": billing_response.get("answer", ""),
            "citations": billing_response.get("citations", []),
            "confidence": billing_response.get("top_score", 0.0),
            "validation_details": validation.details
        }
    
    def _create_rejected_response(
        self,
        billing_response: Dict,
        validation: ValidationResult
    ) -> Dict:
        """Create the rejected response payload with clarifying questions."""
        # Determine clarifying questions based on rejection reason
        clarifying_questions = self._generate_clarifying_questions(validation)
        
        clarifying_message = (
            "I need a bit more information to answer your question accurately:\n" +
            "\n".join(f"• {q}" for q in clarifying_questions)
        )
        
        return {
            "approved": False,
            "reason": validation.reason,
            "clarifying_questions": clarifying_questions,
            "clarifying_message": clarifying_message,
            "validation_details": validation.details,
            # Include original response for debugging
            "original_answer": billing_response.get("answer", ""),
            "original_score": billing_response.get("top_score", 0.0)
        }
    
    def _generate_clarifying_questions(
        self,
        validation: ValidationResult
    ) -> List[str]:
        """Generate clarifying questions based on why validation failed."""
        questions = []
        
        if validation.details:
            check_type = validation.details.get("check", "")
            
            if check_type == "citations_present":
                questions = [
                    "Can you provide your account number or customer ID?",
                    "What specific billing period are you asking about?",
                    "Can you provide more details about your question?"
                ]
            
            elif check_type == "confidence_threshold":
                # Use pre-generated questions if available
                questions = validation.details.get("clarifying_questions", [
                    "Can you be more specific about what you're looking for?",
                    "Which billing period or invoice are you asking about?",
                    "Can you provide your account details?"
                ])
            
            elif check_type == "amounts_verified":
                unverified = validation.details.get("unverified_amounts", [])
                questions = [
                    f"I found amounts {unverified} in the answer but couldn't verify them.",
                    "Can you confirm which charges you're asking about?",
                    "Which specific invoice or bill are you referring to?"
                ]
        
        if not questions:
            questions = [
                "Can you provide more details about your question?",
                "What specific information are you looking for?"
            ]
        
        return questions
    
    def get_validation_summary(self, result: Dict) -> str:
        """
        Get a human-readable summary of the validation result.
        
        Useful for logging and debugging.
        
        Args:
            result: The validation result dict
        
        Returns:
            Formatted summary string
        """
        if result.get("approved"):
            return (
                f"✓ APPROVED\n"
                f"  Reason: {result.get('reason', 'N/A')}\n"
                f"  Confidence: {result.get('confidence', 0):.3f}\n"
                f"  Citations: {len(result.get('citations', []))}"
            )
        else:
            return (
                f"✗ REJECTED\n"
                f"  Reason: {result.get('reason', 'N/A')}\n"
                f"  Clarifying Questions: {len(result.get('clarifying_questions', []))}"
            )
