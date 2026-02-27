"""
CLI Interface for Telecom Billing Agentic AI.

This is the main entry point for interacting with the system.

Usage:
    python -m app.cli "What is my bill for January 2026?"
    python -m app.cli --demo  # Run all 3 demo queries
    python -m app.cli --interactive  # Interactive mode

The CLI will:
1. Take your question
2. Route it through the LangGraph workflow
3. Show you which agents handled it
4. Return the final answer with citations
"""

import sys
import argparse
import uuid

from app.config import validate_config
from app.graph import run_query, set_rag_strategy
from app.utils.logging import setup_logging, Colors
from app.memory.session_store import get_session_store


# Demo queries that showcase the system's capabilities
DEMO_QUERIES = [
    "How much is my bill for January 2026?",
    "Why is my bill higher this month?",
    "What is the due date and what happens if I pay late?",
]


def print_banner():
    """Print the application banner."""
    banner = """
+===============================================================+
|                                                               |
|   TELCOMAX BILLING ASSISTANT (Agentic AI Demo)               |
|                                                               |
|   Powered by:                                                 |
|   * LangChain + LangGraph (Agent Orchestration)              |
|   * Pinecone (Vector RAG)                                     |
|   * OpenAI GPT (LLM + Embeddings)                            |
|                                                               |
+===============================================================+
"""
    print(f"{Colors.BOLD}{Colors.INFO}{banner}{Colors.RESET}")


def run_single_query(query: str, session_id: str = None):
    """Run a single query through the system."""
    print(f"\n{Colors.BOLD}Your Question:{Colors.RESET}")
    print(f"   \"{query}\"\n")
    
    result = run_query(query, session_id=session_id)
    
    return result


def run_demo_mode():
    """Run all demo queries to showcase the system."""
    print(f"\n{Colors.BOLD}DEMO MODE - Running 3 Demo Queries{Colors.RESET}")
    print("=" * 60)
    
    results = []
    
    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"\n{Colors.BOLD}{'─' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}Demo Query {i}/3:{Colors.RESET}")
        print(f"{'─' * 60}")
        
        result = run_single_query(query)
        results.append(result)
        
        # Pause between queries for readability
        if i < len(DEMO_QUERIES):
            print(f"\n{Colors.INFO}[Press Enter for next query...]{Colors.RESET}")
            input()
    
    # Summary
    print(f"\n{Colors.BOLD}{'=' * 60}")
    print("DEMO SUMMARY")
    print(f"{'=' * 60}{Colors.RESET}")
    
    for i, (query, result) in enumerate(zip(DEMO_QUERIES, results), 1):
        approved = result.get("approved")
        status = "[OK]" if approved else ("[?]" if approved is None else "[X]")
        citations = len(result.get("citations", []))
        print(f"\n{i}. \"{query[:40]}...\"")
        print(f"   Status: {status} | Citations: {citations}")
        print(f"   Trace: {' -> '.join(result.get('trace', [])[:3])}")


def run_interactive_mode():
    """
    Run in interactive mode where user can type multiple queries.
    
    NEW: Creates a session ID to remember context between queries.
    """
    # Generate a unique session ID for this interactive session
    session_id = f"interactive_{uuid.uuid4().hex[:8]}"
    
    print(f"\n{Colors.BOLD}INTERACTIVE MODE{Colors.RESET}")
    print(f"Session ID: {session_id}")
    print("\nThis session will remember context between queries!")
    print("Tip: Start with 'My account is ACC-789456123' or just ask about your bill.")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            query = input(f"{Colors.INFO}You: {Colors.RESET}").strip()
            
            if not query:
                continue
            
            if query.lower() in ("exit", "quit", "q"):
                # Show session summary before exiting
                store = get_session_store()
                session = store.get(session_id)
                if session and (session.account_id or session.customer_name):
                    print(f"\n{Colors.INFO}Session Summary:{Colors.RESET}")
                    if session.account_id:
                        print(f"  Account: {session.account_id}")
                    if session.customer_name:
                        print(f"  Name: {session.customer_name}")
                    print(f"  Queries: {len(session.conversation_history) // 2}")
                print(f"\n{Colors.SUCCESS}Goodbye!{Colors.RESET}\n")
                break
            
            # Special commands
            if query.lower() == "session":
                # Show current session info
                store = get_session_store()
                session = store.get(session_id)
                if session:
                    print(f"\n{Colors.INFO}Current Session:{Colors.RESET}")
                    print(f"  {session.get_context_summary() or 'No context yet'}")
                    print()
                continue
            
            run_single_query(query, session_id=session_id)
            print()
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.SUCCESS}Session ended. Goodbye!{Colors.RESET}\n")
            break
        except EOFError:
            break


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TelcoMax Billing Assistant - Agentic AI Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m app.cli "What is my bill for January 2026?"
    python -m app.cli --demo
    python -m app.cli --interactive
        """
    )
    
    parser.add_argument(
        "query",
        nargs="?",
        help="The question to ask the billing assistant"
    )
    
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run all 3 demo queries to showcase the system"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (type multiple queries)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--rag-strategy",
        choices=["naive", "hyde", "multi-query"],
        default="naive",
        help="RAG retrieval strategy: naive (default), hyde, or multi-query"
    )
    
    args = parser.parse_args()
    
    # Setup
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)
    
    print_banner()
    
    # Validate configuration
    try:
        validate_config()
    except ValueError as e:
        print(f"{Colors.ERROR}[ERROR] Configuration Error:{Colors.RESET}")
        print(f"   {e}")
        print(f"\n   Please set up your {Colors.BOLD}.env{Colors.RESET} file.")
        print(f"   See {Colors.BOLD}.env.example{Colors.RESET} for reference.")
        sys.exit(1)
    
    # Set RAG strategy
    set_rag_strategy(args.rag_strategy)
    print(f"  RAG Strategy: {Colors.INFO}{args.rag_strategy}{Colors.RESET}")
    
    # Determine mode
    if args.demo:
        run_demo_mode()
    elif args.interactive:
        run_interactive_mode()
    elif args.query:
        run_single_query(args.query)
    else:
        # No arguments - show help and suggest demo mode
        parser.print_help()
        print(f"\n{Colors.INFO}Tip: Try --demo to see the system in action!{Colors.RESET}\n")


if __name__ == "__main__":
    main()
