"""
LangGraph State Machine — RAG Orchestration Layer
Nodes: CLASSIFY → RETRIEVE → GENERATE → POST_PROCESS
With conditional routing for greetings, abusive content, and low-confidence fallback.
"""

import json
import logging
from typing import TypedDict, Optional, List
from datetime import datetime

from langgraph.graph import StateGraph, END

from app.rag.pipeline import (
    is_abusive, is_greeting, mask_pii,
    search_similar, generate_response,
    get_langfuse_handler
)
from app.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


# ========================
# STATE DEFINITION
# ========================

class RAGState(TypedDict):
    """State that flows through the LangGraph pipeline"""
    # Input
    query: str
    user_name: str
    user_email: str
    chat_history: list

    # Classification
    query_type: str  # "greeting" | "rag" | "abusive"

    # PII
    safe_query: str
    pii_found: bool
    pii_entities: list

    # Retrieval
    context: str
    sources: list
    confidence: float

    # Generation
    response: str
    latency: float

    # Error
    error: Optional[str]


# ========================
# NODES
# ========================

def classify_node(state: RAGState) -> dict:
    """Node 1: Classify the query type"""
    query = state["query"]

    if is_abusive(query):
        logger.info(f"Query classified: ABUSIVE")
        return {"query_type": "abusive"}
    elif is_greeting(query):
        logger.info(f"Query classified: GREETING")
        return {"query_type": "greeting"}
    else:
        logger.info(f"Query classified: RAG")
        return {"query_type": "rag"}


def reject_node(state: RAGState) -> dict:
    """Handle abusive queries"""
    return {
        "response": "I am a Legal AI Assistant. I can only respond to professional and respectful queries. Please rephrase your question.",
        "confidence": 0,
        "latency": 0,
        "sources": [],
        "error": "abusive_content"
    }


def greet_node(state: RAGState) -> dict:
    """Handle greetings — skip vector DB entirely"""
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    import time

    start = time.time()

    llm = ChatOpenAI(
        model="qwen/qwen3-235b-a22b-thinking-2507",
        openai_api_key=settings.OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1",
        temperature=0.7,
        max_tokens=150
    )

    greeting_prompt = """You are Indian Legal AI Expert & Advisor created by Ambuj Kumar Tripathi.
Respond to this greeting warmly and briefly (max 25 words).
Be friendly and professional. Mention you can help with Constitution, BNS, BNSS, Consumer Protection, IT Act & Motor Vehicles Act.
User: {query}"""

    chain = ChatPromptTemplate.from_template(greeting_prompt) | llm | StrOutputParser()
    response = chain.invoke({"query": state["query"]})

    return {
        "response": response,
        "confidence": 100,
        "latency": round(time.time() - start, 2),
        "sources": [],
        "safe_query": state["query"],
        "pii_found": False,
        "pii_entities": []
    }


async def retrieve_node(state: RAGState) -> dict:
    """Node 2: PII mask + Search Qdrant"""
    query = state["query"]

    # Simple PII Masking
    safe_query, pii_found, pii_entities = mask_pii(query)

    # Manual Keyword Expansion (to fix missing context like FIR => Section 173)
    KEYWORD_MAP = {
        "fir": "First Information Report Section 173 BNSS",
        "murder": "Section 103 BNS",
        "theft": "Section 303 BNS",
        "arrest": "Section 35 BNSS",
        "bail": "Section 479 BNSS"
    }
    
    search_query = safe_query
    for key, value in KEYWORD_MAP.items():
        if key in safe_query.lower():
            search_query += f" {value}"

    # Search Qdrant
    results = search_similar(search_query, top_k=15, user_email=state.get("user_email"))

    if not results:
        return {
            "safe_query": safe_query,
            "pii_found": pii_found,
            "pii_entities": pii_entities,
            "context": "",
            "sources": [],
            "confidence": 0
        }

    # Best confidence score (cosine similarity * 100)
    confidence = results[0]["score"] * 100

    # Build context from PARENT texts (larger, more contextual)
    context = "\n\n---\n\n".join([r["parent_text"] for r in results])

    # Format sources
    sources = [
        {
            "source_id": i + 1,
            "file": r["source_file"].replace(".pdf", ""),
            "page": r["page"] + 1,
            "preview": r["child_text"][:300],
            "score": round(r["score"], 3)
        }
        for i, r in enumerate(results)
    ]

    return {
        "safe_query": safe_query,
        "pii_found": pii_found,
        "pii_entities": pii_entities,
        "context": context,
        "sources": sources,
        "confidence": round(confidence, 1)
    }


def generate_node(state: RAGState) -> dict:
    """Node 3: Generate LLM response with context"""
    import time

    # Low confidence fallback
    if state.get("confidence", 0) < 40:
        return {
            "response": "I don't have specific information about this in my legal documents. Could you rephrase your question or ask about a specific law, act, or legal topic that I cover?\n\n**I can help with:** Constitution of India, BNS 2023, BNSS 2023, Consumer Protection Act, IT Act 2000, Motor Vehicles Act.\n\n> *\"⚠️ Disclaimer: I am an AI assistant. For critical legal matters, please consult a qualified professional.\"*",
            "latency": 0
        }

    # Format chat history
    history_text = "No previous history."
    if state.get("chat_history"):
        msgs = state["chat_history"][-6:]
        formatted = []
        for msg in msgs:
            prefix = "User: " if msg.get("role") == "user" else "Assistant: "
            formatted.append(prefix + str(msg.get("content", "")))
        history_text = "\n".join(formatted)

    try:
        response, latency = generate_response(
            question=state.get("safe_query", state["query"]),
            context=state.get("context", ""),
            history=history_text,
            user_name=state.get("user_name", "User")
        )
        return {"response": response, "latency": round(latency, 2)}
    except Exception as e:
        logger.error(f"Generate error: {e}")
        return {
            "response": "I'm temporarily unable to process your request. Please try again in a moment.",
            "latency": 0,
            "error": str(e)
        }


def post_process_node(state: RAGState) -> dict:
    """Node 4: Save to MongoDB + Log"""
    from app.db.database import save_message

    user_email = state.get("user_email", "anonymous")

    # Save user message
    save_message(
        user_email,
        "user",
        state.get("safe_query") or state["query"],
        pii_masked=state.get("pii_found", False),
        pii_entities=state.get("pii_entities", [])
    )

    # Save assistant response
    if state.get("response"):
        save_message(
            user_email,
            "assistant",
            state["response"],
            state.get("sources"),
            pii_masked=state.get("pii_found", False),
            pii_entities=state.get("pii_entities", [])
        )

    logger.info(json.dumps({
        "event": "rag_complete",
        "user": user_email,
        "query_type": state.get("query_type", "unknown"),
        "confidence": state.get("confidence", 0),
        "latency": state.get("latency", 0),
        "pii_found": state.get("pii_found", False),
        "timestamp": datetime.now().isoformat()
    }))

    return {}


# ========================
# ROUTING LOGIC
# ========================

def route_after_classify(state: RAGState) -> str:
    """Decide which node to go to after classification"""
    query_type = state.get("query_type", "rag")
    if query_type == "abusive":
        return "reject"
    elif query_type == "greeting":
        return "greet"
    return "retrieve"


# ========================
# BUILD GRAPH
# ========================

def build_rag_graph() -> StateGraph:
    """Build and compile the LangGraph RAG pipeline"""
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("classify", classify_node)
    graph.add_node("reject", reject_node)
    graph.add_node("greet", greet_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("post_process", post_process_node)

    # Set entry point
    graph.set_entry_point("classify")

    # Conditional routing after classify
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "reject": "reject",
            "greet": "greet",
            "retrieve": "retrieve"
        }
    )

    # Linear flow: retrieve → generate → post_process
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "post_process")

    # Terminal edges
    graph.add_edge("reject", END)
    graph.add_edge("greet", "post_process")
    graph.add_edge("post_process", END)

    return graph.compile()


# Compiled graph — ready to invoke
rag_graph = None


def get_rag_graph():
    """Get or create compiled RAG graph"""
    global rag_graph
    if rag_graph is None:
        rag_graph = build_rag_graph()
        logger.info("LangGraph RAG pipeline compiled ✅")
    return rag_graph
