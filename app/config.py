"""
Configuration module for the Telecom Billing Agentic AI system.

This module handles:
- Environment variable loading from .env file
- API key management (OpenAI, Pinecone)
- System-wide constants and thresholds
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directory containing documents
DATA_DIR = PROJECT_ROOT / "data" / "docs"

# =============================================================================
# API KEYS (loaded from environment)
# =============================================================================

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # Cost-effective for demo
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "telecom-billing-demo")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "telecom-docs")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# =============================================================================
# RAG CONFIGURATION
# =============================================================================

# Chunking parameters (for splitting documents)
CHUNK_SIZE = 400  # tokens per chunk (300-500 range as specified)
CHUNK_OVERLAP = 75  # overlap between chunks (50-100 range)

# Retrieval parameters
RETRIEVAL_TOP_K = 4  # Number of chunks to retrieve

# Embedding dimension for text-embedding-3-small
EMBEDDING_DIMENSION = 1536

# =============================================================================
# GUARDRAILS CONFIGURATION
# =============================================================================

# Minimum confidence score for ManagerAgent to approve an answer
# Lower for demo (limited docs); production would use 0.75+
CONFIDENCE_THRESHOLD = 0.40

# Maximum words in a citation quote
MAX_QUOTE_WORDS = 20

# Whether to require dollar amounts in answer to appear in citation quotes
# Set to False for demo (quotes are truncated, may not contain amounts)
# In production with full document access, set to True
STRICT_AMOUNT_VERIFICATION = False

# =============================================================================
# AGENT QUERY CLASSIFICATION
# =============================================================================

# Query intent categories
class QueryIntent:
    BILLING_ACCOUNT_SPECIFIC = "billing_account_specific"
    BILLING_GENERAL = "billing_general"
    SALES_GENERAL = "sales_general"


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate that required environment variables are set."""
    errors = []
    
    if not OPENAI_API_KEY:
        errors.append("OPENAI_API_KEY is not set")
    
    if not PINECONE_API_KEY:
        errors.append("PINECONE_API_KEY is not set")
    
    if errors:
        raise ValueError(
            "Missing required environment variables:\n" + 
            "\n".join(f"  - {e}" for e in errors) +
            "\n\nPlease set them in your .env file. See .env.example for reference."
        )


# Helper function to check if running in test mode
def is_test_mode():
    """Check if we're running in test/demo mode."""
    return os.getenv("TELECOM_TEST_MODE", "false").lower() == "true"
