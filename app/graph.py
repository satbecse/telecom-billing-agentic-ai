"""
LangGraph Workflow for Telecom Billing Multi-Agent System.

CONCEPT: What is LangGraph?
===========================
LangGraph is a framework for building STATEFUL agent workflows.

Think of it like a flowchart:
- NODES are the "boxes" (things that happen)
- EDGES are the "arrows" (transitions between nodes)
- STATE is shared data that flows through the graph

Our Workflow:
                    ┌─────────────┐
                    │    START    │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   Router    │ ← Classifies the query
                    └──────┬──────┘
                           │
         ┌─────────────────┼─────────────────┐
         │                 │                 │
         ▼                 ▼                 ▼
    ┌─────────┐     ┌───────────┐     ┌──────────┐
    │  Sales  │     │  Billing  │     │  Sales   │
    │ General │     │  General  │     │ (needs   │
    │         │     │           │     │ routing) │
    └────┬────┘     └─────┬─────┘     └────┬─────┘
         │                │                │
         │                │         ┌──────▼──────┐
         │                │         │   Billing   │ ← RAG lookup
         │                │         │   Agent     │
         │                │         └──────┬──────┘
         │                │                │
         │                │         ┌──────▼──────┐
         │                │         │   Manager   │ ← Validates
         │                │         │   Agent     │
         │                │         └──────┬──────┘
         │                │                │
         └────────────────┴────────┬───────┘
                                   │
                            ┌──────▼──────┐
                            │  Format &   │
                            │  Return     │
                            └──────┬──────┘
                                   │
                            ┌──────▼──────┐
                            │    END      │
                            └─────────────┘
"""

from typing import Dict, Any, TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END

from app.config import QueryIntent
from app.agents.sales import SalesAgent
from app.agents.billing import BillingAgent
from app.agents.manager import ManagerAgent
from app.memory.session_store import get_session_store, SessionData
from app.memory.entity_extractor import get_entity_extractor
from app.utils.logging import (
    get_logger,
    print_trace_header,
    print_agent_action,
    print_decision,
    print_final_response
)

logger = get_logger("graph")


# =============================================================================
# STATE DEFINITION
# =============================================================================

class GraphState(TypedDict):
    """
    State that flows through the graph.
    
    CONCEPT: Shared State
    =====================
    Every node can read and modify this state.
    It's like a "clipboard" that passes between agents.
    
    Fields:
    - query: The original user question
    - session_id: Session identifier for memory
    - session_context: Context string from session memory
    - intent: Classified query type
    - messages: History of agent responses
    - billing_response: Response from BillingAgent
    - manager_result: Validation result from ManagerAgent
    - final_response: The approved final answer
    - trace: List of actions for debugging
    """
    query: str
    session_id: str  # NEW: Session ID for memory
    session_context: str  # NEW: Context from session
    intent: str
    messages: list[Dict[str, Any]]
    billing_response: Dict[str, Any]
    manager_result: Dict[str, Any]
    final_response: str
    trace: list[str]
    citations: list[Dict[str, Any]]


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

# Initialize agents (they're reused across calls)
sales_agent = SalesAgent()
billing_agent = BillingAgent()
manager_agent = ManagerAgent()


def router_node(state: GraphState) -> GraphState:
    """
    Router Node: Classifies the query and decides the path.
    
    This is the first node in the graph.
    It determines which path the query should take.
    
    NEW: Also extracts entities from the query and updates session.
    """
    print_trace_header("Router Node")
    
    query = state["query"]
    session_id = state.get("session_id", "")
    
    # NEW: Extract entities from the query and update session
    if session_id:
        store = get_session_store()
        extractor = get_entity_extractor()
        
        # Extract entities (account_id, name, etc.)
        entities = extractor.extract_and_update_session(query, session_id, store)
        
        if entities.has_entities():
            print_agent_action("Router", f"Extracted entities: {entities.to_dict()}")
        
        # Get updated session context
        session = store.get(session_id)
        if session:
            context = session.get_context_summary()
            if context:
                state["session_context"] = context
                print_agent_action("Router", f"Session context: {context}")
    
    print_agent_action("Router", f"Classifying query: '{query[:50]}...'")
    
    # Classify the query
    intent = sales_agent.classify_query(query)
    
    print_agent_action("Router", f"Intent classified as: {intent}")
    
    # Update state
    state["intent"] = intent
    state["trace"].append(f"Router: classified as {intent}")
    
    return state


def sales_node(state: GraphState) -> GraphState:
    """
    Sales Node: Handles general sales/policy questions.
    
    This node is used when the query is about:
    - General pricing
    - Plan features
    - Company policies
    """
    print_trace_header("Sales Agent")
    
    query = state["query"]
    intent = state["intent"]
    
    # Check if we have a manager result to format
    if state.get("manager_result"):
        # We're returning from billing flow
        manager_result = state["manager_result"]
        response = sales_agent.format_final_response(
            manager_result,
            manager_result.get("approved", False)
        )
    else:
        # Direct sales response
        response, needs_routing = sales_agent.generate_response(query)
        
        if needs_routing:
            state["trace"].append("SalesAgent: routing to BillingAgent")
            # Don't set final response - continue to billing
            return state
    
    state["final_response"] = response
    state["trace"].append("SalesAgent: provided response")
    
    return state


def billing_node(state: GraphState) -> GraphState:
    """
    Billing Node: Uses RAG to answer account-specific questions.
    
    This node:
    1. Retrieves relevant documents from Pinecone
    2. Generates an answer with citations
    3. Passes to ManagerAgent for validation
    
    NEW: Uses session context to enhance the query.
    """
    print_trace_header("Billing Agent")
    
    query = state["query"]
    session_context = state.get("session_context", "")
    
    # NEW: Enhance query with session context for better retrieval
    enhanced_query = query
    if session_context:
        # Add context to help BillingAgent know which account to look for
        enhanced_query = f"{session_context}\n\nUser question: {query}"
        print_agent_action("BillingAgent", f"Using session context for query")
    
    # Process with RAG (pass both original and enhanced query)
    billing_response = billing_agent.process_query(query, session_context=session_context)
    
    # Store response for manager validation
    state["billing_response"] = billing_response
    state["trace"].append(
        f"BillingAgent: retrieved docs, score={billing_response.get('top_score', 0):.3f}"
    )
    
    return state


def manager_node(state: GraphState) -> GraphState:
    """
    Manager Node: Validates BillingAgent responses.
    
    This node applies guardrails:
    - Checks citations are present
    - Validates confidence threshold
    - Verifies dollar amounts against sources
    """
    print_trace_header("Manager Agent")
    
    billing_response = state["billing_response"]
    
    # Validate the response
    approved, result = manager_agent.validate_response(billing_response)
    
    state["manager_result"] = result
    state["trace"].append(
        f"ManagerAgent: {'APPROVED' if approved else 'REJECTED'} - {result.get('reason', '')[:50]}"
    )
    
    # If approved, extract citations for final response
    if approved:
        state["citations"] = result.get("citations", [])
    
    return state


def format_response_node(state: GraphState) -> GraphState:
    """
    Format Response Node: Prepares the final user-facing response.
    
    This node:
    - Formats approved responses with citations
    - Formats rejection with clarifying questions
    """
    print_trace_header("Formatting Final Response")
    
    manager_result = state.get("manager_result", {})
    
    if manager_result.get("approved"):
        # Format approved response
        answer = manager_result.get("answer", "")
        citations = manager_result.get("citations", [])
        
        response_parts = [answer]
        
        if citations:
            response_parts.append("\n\nSources:")
            for i, cite in enumerate(citations, 1):
                doc_id = cite.get("doc_id", "Unknown")
                quote = cite.get("quote", "")[:60]
                response_parts.append(f"  [{i}] {doc_id}: \"{quote}...\"")
        
        state["final_response"] = "\n".join(response_parts)
        state["citations"] = citations
    else:
        # Format rejection with clarifying questions
        message = manager_result.get("clarifying_message", 
            "I need more information to answer your question.")
        state["final_response"] = message
    
    state["trace"].append("Formatted final response")
    
    return state


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def route_after_router(state: GraphState) -> str:
    """
    Determine the next node after the Router.
    
    This is a CONDITIONAL EDGE - it chooses the path based on state.
    """
    intent = state.get("intent", "")
    
    if intent == QueryIntent.BILLING_ACCOUNT_SPECIFIC:
        return "billing"
    elif intent == QueryIntent.BILLING_GENERAL:
        return "billing"  # Even general billing goes through RAG for accuracy
    else:
        return "sales"


def route_after_sales(state: GraphState) -> str:
    """
    Determine next node after Sales.
    
    If there's a final response, we're done.
    Otherwise, we need to continue the flow.
    """
    if state.get("final_response"):
        return "end"
    else:
        # Check if we need to go to billing (based on intent)
        if state.get("intent") in [QueryIntent.BILLING_ACCOUNT_SPECIFIC, 
                                     QueryIntent.BILLING_GENERAL]:
            return "billing"
        return "end"


def route_after_manager(state: GraphState) -> str:
    """Always go to format response after manager."""
    return "format_response"


# =============================================================================
# BUILD THE GRAPH
# =============================================================================

def create_workflow() -> StateGraph:
    """
    Create and compile the LangGraph workflow.
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("router", router_node)
    workflow.add_node("sales", sales_node)
    workflow.add_node("billing", billing_node)
    workflow.add_node("manager", manager_node)
    workflow.add_node("format_response", format_response_node)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges from router
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "sales": "sales",
            "billing": "billing"
        }
    )
    
    # Sales can either end or continue
    workflow.add_conditional_edges(
        "sales",
        route_after_sales,
        {
            "end": END,
            "billing": "billing"
        }
    )
    
    # Billing always goes to manager
    workflow.add_edge("billing", "manager")
    
    # Manager goes to format response
    workflow.add_edge("manager", "format_response")
    
    # Format response ends
    workflow.add_edge("format_response", END)
    
    # Compile the graph
    return workflow.compile()


# Create the compiled workflow
app_workflow = create_workflow()


def run_query(query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Run a query through the multi-agent workflow.
    
    This is the main entry point for processing queries.
    
    Args:
        query: The user's question
        session_id: Optional session ID for memory. If provided, the system
                   will remember context from previous queries in this session.
    
    Returns:
        Dict with final_response, citations, and trace
    """
    print_trace_header(f"Processing Query")
    print(f"Query: {query}")
    
    # NEW: Handle session
    session_context = ""
    if session_id:
        store = get_session_store()
        session = store.get_or_create(session_id)
        
        # Add user message to conversation history
        store.add_conversation_turn(session_id, "user", query)
        
        # Get existing context
        session_context = session.get_context_summary()
        if session_context:
            print(f"Session: {session_context}")
    
    print()  # Blank line for formatting
    
    # Initialize state
    initial_state: GraphState = {
        "query": query,
        "session_id": session_id or "",  # NEW
        "session_context": session_context,  # NEW
        "intent": "",
        "messages": [],
        "billing_response": {},
        "manager_result": {},
        "final_response": "",
        "trace": [],
        "citations": []
    }
    
    # Run the workflow
    final_state = app_workflow.invoke(initial_state)
    
    # NEW: Save the response to session
    final_response = final_state.get("final_response", "No response generated")
    if session_id:
        store = get_session_store()
        store.add_conversation_turn(session_id, "assistant", final_response)
        
        # Update last query/response for context
        store.update(session_id, 
            last_query=query,
            last_response=final_response[:500]  # Truncate for storage
        )
    
    # Print results
    print_final_response(
        final_response,
        final_state.get("citations", [])
    )
    
    # Print trace
    print("\n** Execution Trace: **")
    for i, step in enumerate(final_state.get("trace", []), 1):
        print(f"  {i}. {step}")
    
    return {
        "final_response": final_response,
        "citations": final_state.get("citations", []),
        "trace": final_state.get("trace", []),
        "approved": final_state.get("manager_result", {}).get("approved", None)
    }
