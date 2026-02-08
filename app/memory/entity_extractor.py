"""
Entity Extractor - Extracts structured information from user messages.

CONCEPT: What is Entity Extraction?
====================================
When a user says "My account is ACC-DEMO-001", we want to:
1. DETECT that they mentioned an account number
2. EXTRACT the actual value (ACC-DEMO-001)
3. STORE it in the session for later use

This is a simple regex-based extractor. In production, you might use:
- Named Entity Recognition (NER) models
- LLM-based extraction
- More sophisticated pattern matching

Entities we extract:
- account_id: Customer account numbers (ACC-XXXX-XXX)
- customer_name: Names mentioned in greetings
- billing_period: Months/dates mentioned
- dollar_amounts: Money amounts mentioned
"""

import re
from typing import Dict, Optional, List
from dataclasses import dataclass

from app.utils.logging import get_logger

logger = get_logger("entity_extractor")


@dataclass
class ExtractedEntities:
    """Container for all entities extracted from a message."""
    account_id: Optional[str] = None
    customer_name: Optional[str] = None
    billing_period: Optional[str] = None
    dollar_amounts: List[str] = None
    topic: Optional[str] = None
    
    def __post_init__(self):
        if self.dollar_amounts is None:
            self.dollar_amounts = []
    
    def has_entities(self) -> bool:
        """Check if any entities were extracted."""
        return any([
            self.account_id,
            self.customer_name,
            self.billing_period,
            self.dollar_amounts,
            self.topic
        ])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "account_id": self.account_id,
            "customer_name": self.customer_name,
            "billing_period": self.billing_period,
            "dollar_amounts": self.dollar_amounts,
            "topic": self.topic,
        }


class EntityExtractor:
    """
    Extracts structured entities from user messages.
    
    This uses simple regex patterns. It's fast and doesn't require
    API calls, but is less flexible than LLM-based extraction.
    
    Usage:
        extractor = EntityExtractor()
        entities = extractor.extract("My account is ACC-DEMO-001")
        print(entities.account_id)  # "ACC-DEMO-001"
    """
    
    # ==========================================================================
    # REGEX PATTERNS
    # ==========================================================================
    
    # Account ID pattern: ACC-XXXX-XXX or ACC-XXXXXXXXX
    ACCOUNT_PATTERNS = [
        r'\b(ACC-[A-Z0-9]+-[A-Z0-9]+)\b',  # ACC-DEMO-001
        r'\b(ACC-\d{9,12})\b',  # ACC-789456123 (our demo format)
        r'\baccount\s*(?:number|#|id)?:?\s*([A-Z0-9-]+)\b',  # account number: XYZ
        r'\baccount\s+is\s+([A-Z0-9-]+)\b',  # account is XYZ
    ]
    
    # Name patterns (simple - looks for "I'm X" or "My name is X")
    NAME_PATTERNS = [
        r"(?:I'm|I am|my name is|this is)\s+([A-Z][a-z]+)",
        r"^([A-Z][a-z]+)\s+here",  # "Dileep here"
    ]
    
    # Billing period patterns
    PERIOD_PATTERNS = [
        # Month Year: "January 2026", "Jan 2026"
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\b',
        # Month only in context of bill
        r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+bill\b',
        # This month / last month
        r'\b(this month|last month|current month|previous month)\b',
    ]
    
    # Dollar amount pattern
    DOLLAR_PATTERN = r'\$([0-9]+(?:\.[0-9]{2})?)'
    
    # Topic detection keywords
    TOPIC_KEYWORDS = {
        "billing": ["bill", "invoice", "charge", "payment", "amount", "due", "balance"],
        "plans": ["plan", "upgrade", "downgrade", "package", "subscription"],
        "dispute": ["dispute", "wrong", "incorrect", "error", "overcharge", "refund"],
        "late_fee": ["late", "fee", "penalty", "overdue"],
        "support": ["help", "support", "issue", "problem", "question"],
    }
    
    def __init__(self):
        # Pre-compile regex patterns for performance
        self._account_patterns = [re.compile(p, re.IGNORECASE) for p in self.ACCOUNT_PATTERNS]
        self._name_patterns = [re.compile(p, re.IGNORECASE) for p in self.NAME_PATTERNS]
        self._period_patterns = [re.compile(p, re.IGNORECASE) for p in self.PERIOD_PATTERNS]
        self._dollar_pattern = re.compile(self.DOLLAR_PATTERN)
    
    def extract(self, text: str) -> ExtractedEntities:
        """
        Extract all entities from a text message.
        
        Args:
            text: The user's message
        
        Returns:
            ExtractedEntities with any found values
        """
        entities = ExtractedEntities()
        
        # Extract each entity type
        entities.account_id = self._extract_account_id(text)
        entities.customer_name = self._extract_name(text)
        entities.billing_period = self._extract_billing_period(text)
        entities.dollar_amounts = self._extract_dollar_amounts(text)
        entities.topic = self._extract_topic(text)
        
        # Log what we found
        if entities.has_entities():
            logger.info(f"Extracted entities: {entities.to_dict()}")
        
        return entities
    
    def _extract_account_id(self, text: str) -> Optional[str]:
        """Extract account ID from text."""
        for pattern in self._account_patterns:
            match = pattern.search(text)
            if match:
                # Get the captured group
                account_id = match.group(1).upper()
                logger.debug(f"Found account ID: {account_id}")
                return account_id
        return None
    
    def _extract_name(self, text: str) -> Optional[str]:
        """Extract customer name from text."""
        for pattern in self._name_patterns:
            match = pattern.search(text)
            if match:
                name = match.group(1).title()
                logger.debug(f"Found name: {name}")
                return name
        return None
    
    def _extract_billing_period(self, text: str) -> Optional[str]:
        """Extract billing period from text."""
        for pattern in self._period_patterns:
            match = pattern.search(text)
            if match:
                # Handle different pattern groups
                groups = match.groups()
                if len(groups) >= 2 and groups[1]:
                    # Month + Year
                    period = f"{groups[0]} {groups[1]}"
                else:
                    # Just the match
                    period = match.group(0)
                
                logger.debug(f"Found billing period: {period}")
                return period
        return None
    
    def _extract_dollar_amounts(self, text: str) -> List[str]:
        """Extract dollar amounts from text."""
        matches = self._dollar_pattern.findall(text)
        if matches:
            amounts = [f"${m}" for m in matches]
            logger.debug(f"Found amounts: {amounts}")
            return amounts
        return []
    
    def _extract_topic(self, text: str) -> Optional[str]:
        """Detect the topic based on keywords."""
        text_lower = text.lower()
        
        # Count keyword matches for each topic
        topic_scores = {}
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                topic_scores[topic] = score
        
        # Return the topic with the highest score
        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            logger.debug(f"Detected topic: {best_topic}")
            return best_topic
        
        return None
    
    def extract_and_update_session(
        self, 
        text: str, 
        session_id: str,
        session_store
    ) -> ExtractedEntities:
        """
        Extract entities and automatically update the session.
        
        This is a convenience method that combines extraction and storage.
        
        Args:
            text: The user's message
            session_id: The session to update
            session_store: The SessionStore instance
        
        Returns:
            ExtractedEntities with the found values
        """
        entities = self.extract(text)
        
        # Build update dict with only non-None values
        updates = {}
        if entities.account_id:
            updates["account_id"] = entities.account_id
        if entities.customer_name:
            updates["customer_name"] = entities.customer_name
        if entities.billing_period:
            updates["billing_period"] = entities.billing_period
        if entities.topic:
            updates["current_topic"] = entities.topic
        
        # Update session if we have any entities
        if updates:
            session_store.update(session_id, **updates)
            logger.info(f"Updated session {session_id} with: {updates}")
        
        return entities


# =============================================================================
# GLOBAL EXTRACTOR INSTANCE
# =============================================================================

_global_extractor: Optional[EntityExtractor] = None


def get_entity_extractor() -> EntityExtractor:
    """Get the global entity extractor instance."""
    global _global_extractor
    if _global_extractor is None:
        _global_extractor = EntityExtractor()
    return _global_extractor
