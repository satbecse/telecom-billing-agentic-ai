"""
Logging utility for the Telecom Billing Agentic AI system.

Provides structured logging with:
- Color-coded console output
- Agent-specific log tags
- Trace information for debugging workflows
"""

import logging
import sys
from typing import Optional


# ANSI color codes for console output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Agent colors
    ROUTER = "\033[94m"      # Blue
    SALES = "\033[92m"       # Green
    BILLING = "\033[93m"     # Yellow
    MANAGER = "\033[95m"     # Magenta
    
    # Status colors
    SUCCESS = "\033[92m"     # Green
    WARNING = "\033[93m"     # Yellow
    ERROR = "\033[91m"       # Red
    INFO = "\033[96m"        # Cyan


class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors based on log level and agent."""
    
    LEVEL_COLORS = {
        logging.DEBUG: Colors.INFO,
        logging.INFO: Colors.RESET,
        logging.WARNING: Colors.WARNING,
        logging.ERROR: Colors.ERROR,
        logging.CRITICAL: Colors.ERROR + Colors.BOLD,
    }
    
    def format(self, record):
        # Get color based on level
        color = self.LEVEL_COLORS.get(record.levelno, Colors.RESET)
        
        # Format the message
        formatted = super().format(record)
        
        return f"{color}{formatted}{Colors.RESET}"


def setup_logging(level: int = logging.INFO) -> None:
    """
    Set up logging for the application.
    
    Args:
        level: Logging level (default: INFO)
    """
    # Create root logger
    root_logger = logging.getLogger("telecom")
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Create console handler with colored formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = ColoredFormatter(
        fmt="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name (e.g., "telecom.sales", "telecom.billing")
    
    Returns:
        Logger instance
    """
    return logging.getLogger(f"telecom.{name}")


# =============================================================================
# TRACE PRINTING UTILITIES
# =============================================================================

def print_trace_header(title: str) -> None:
    """Print a formatted header for trace output."""
    width = 60
    print("\n" + "=" * width)
    print(f"{Colors.BOLD}{title.center(width)}{Colors.RESET}")
    print("=" * width)


def print_agent_action(agent_name: str, action: str, details: Optional[str] = None) -> None:
    """
    Print an agent's action for trace output.
    
    Args:
        agent_name: Name of the agent (Router, Sales, Billing, Manager)
        action: What the agent is doing
        details: Optional additional details
    """
    # Get color for agent
    agent_colors = {
        "Router": Colors.ROUTER,
        "Sales": Colors.SALES,
        "SalesAgent": Colors.SALES,
        "Billing": Colors.BILLING,
        "BillingAgent": Colors.BILLING,
        "Manager": Colors.MANAGER,
        "ManagerAgent": Colors.MANAGER,
    }
    color = agent_colors.get(agent_name, Colors.RESET)
    
    print(f"\n{color}> [{agent_name}]{Colors.RESET} {action}")
    if details:
        # Indent details
        for line in details.split("\n"):
            print(f"  {line}")


def print_decision(decision: str, approved: bool = True) -> None:
    """Print a decision outcome."""
    if approved:
        icon = "[OK]"
        color = Colors.SUCCESS
    else:
        icon = "[X]"
        color = Colors.ERROR
    
    print(f"\n{color}{icon} {decision}{Colors.RESET}")


def print_final_response(response: str, citations: list = None) -> None:
    """Print the final response to the user."""
    print("\n" + "-" * 60)
    print(f"{Colors.BOLD}FINAL RESPONSE:{Colors.RESET}")
    print("-" * 60)
    print(response)
    
    if citations:
        print(f"\n{Colors.INFO}Sources:{Colors.RESET}")
        for i, citation in enumerate(citations, 1):
            doc_id = citation.get("doc_id", "Unknown")
            chunk_id = citation.get("chunk_id", "Unknown")
            quote = citation.get("quote", "")
            print(f"  [{i}] {doc_id} (chunk {chunk_id})")
            print(f"      \"{quote[:100]}...\"" if len(quote) > 100 else f"      \"{quote}\"")
    
    print("-" * 60 + "\n")
