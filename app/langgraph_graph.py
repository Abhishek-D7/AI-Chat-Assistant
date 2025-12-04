
from typing import TypedDict, Annotated, Literal, Optional, Dict, AsyncGenerator, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel
import operator
import os
import logging
from dotenv import load_dotenv
import asyncio
import json

load_dotenv()
from app.config import Config
from app.cache import SystemMessageCache
from app.state import AgentState
from app.agent_config import AGENTS_CONFIG, AGENT_NAMES
from app.agents import create_booking_agent, create_support_agent

logger = logging.getLogger(__name__)

# Initialize system message cache
system_message_cache = SystemMessageCache(max_size=Config.SYSTEM_MESSAGE_CACHE_SIZE)

agent_graph = None

# ================================================================================
# SUPERVISOR NODE
# ================================================================================

class RouteResponse(BaseModel):
    next: Literal["BookingAgent", "SupportAgent", "FINISH"]

def create_agent_graph():
    """Create the Hierarchical Agent Graph"""
    
    logger.info("🔨 Starting graph creation (Hierarchical)...")
    
    # Initialize LLM
    openrouter_api_key = Config.OPENROUTER_API_KEY
    
    logger.info("☁️ Using OpenRouter LLM")
    llm = ChatOpenAI(
        model='openai/gpt-oss-120b', # Or gpt-4o if preferred for routing
        temperature=0.1, # Low temp for routing
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        streaming=True,
        default_headers={
            "HTTP-Referer": "https://github.com/your-repo",
            "X-Title": Config.APP_TITLE
        }
    )
    
    # Create Sub-Agents
    booking_agent_graph = create_booking_agent(llm)
    support_agent_graph = create_support_agent(llm)
    
    # Supervisor Node
    def supervisor_node(state: AgentState):
        messages = state["messages"]
        user_name = state.get("user_name", "User")
        
        system_prompt = (
            "You are a supervisor tasked with managing a conversation between the"
            f" following workers: {AGENT_NAMES}. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with FINISH."
            "\n\n"
            "Worker Descriptions:\n"
        )
        
        for agent_id, config in AGENTS_CONFIG.items():
            system_prompt += f"- {config['name']}: {config['description']}\n"
            
        system_prompt += f"\nUser: {user_name}"
        
        # We use structured output for reliable routing
        try:
            # Using with_structured_output if available
            response = llm.with_structured_output(RouteResponse).invoke(
                [SystemMessage(content=system_prompt)] + messages
            )
            next_agent = response.next
        except Exception as e:
            logger.warning(f"⚠️ Structured output failed: {e}. Fallback to text analysis.")
            # Fallback: Ask for just the name
            fallback_prompt = system_prompt + "\n\nReturn ONLY the name of the next worker or FINISH."
            resp = llm.invoke([SystemMessage(content=fallback_prompt)] + messages)
            content = resp.content.strip()
            # Simple matching
            if "BookingAgent" in content:
                next_agent = "BookingAgent"
            elif "SupportAgent" in content:
                next_agent = "SupportAgent"
            else:
                next_agent = "FINISH"
        
        logger.info(f"🚦 Supervisor routed to: {next_agent}")
        return {"next": next_agent}

    # Build Graph
    workflow = StateGraph(AgentState)
    
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("BookingAgent", booking_agent_graph)
    workflow.add_node("SupportAgent", support_agent_graph)
    
    # Edges
    workflow.add_edge(START, "supervisor")
    
    # Conditional edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        lambda x: x["next"],
        {
            "BookingAgent": "BookingAgent",
            "SupportAgent": "SupportAgent",
            "FINISH": END
        }
    )
    
    # Edges from agents back to supervisor
    workflow.add_edge("BookingAgent", "supervisor")
    workflow.add_edge("SupportAgent", "supervisor")
    
    # Checkpointer
    checkpointer = MemorySaver()
    
    compiled_graph = workflow.compile(checkpointer=checkpointer)
    compiled_graph._llm = llm # Store for streaming access
    
    return compiled_graph

def get_agent_graph():
    """Get or create singleton graph"""
    global agent_graph
    if agent_graph is None:
        agent_graph = create_agent_graph()
    return agent_graph

# ================================================================================
# HELPER: DETECT INTENT
# ================================================================================

def detect_intent_from_messages(messages: list, tool_map: dict = None) -> str:
    """
    Detect intent based on the last active agent or tool calls.
    """
    # Simple heuristic: Check if any tool was called recently
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            return "tool_use"
            
    return "general"

# ================================================================================
# NON-STREAMING
# ================================================================================

def process_user_message_with_context(
    user_message: str,
    user_id: str,
    user_name: str,
    context_summary: str = "",
    thread_id: Optional[str] = None
) -> Dict[str, str]:
    
    try:
        graph = get_agent_graph()
        initial_state = AgentState(
            messages=[HumanMessage(content=user_message)],
            user_id=user_id,
            user_name=user_name,
            context_summary=context_summary,
            next=""
        )
        config = {"configurable": {"thread_id": thread_id}} if thread_id else None
        
        result = graph.invoke(initial_state, config=config)
        final_message = result["messages"][-1]
        
        response_text = final_message.content if hasattr(final_message, "content") else str(final_message)
        
        detected_intent = "general" 
        
        return {
            "bot_response": response_text,
            "intent": detected_intent
        }
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return {"bot_response": "Error processing message.", "intent": "error"}

# ================================================================================
# STREAMING HANDLER
# ================================================================================

async def process_user_message_with_context_streaming(
    user_message: str,
    user_id: str,
    user_name: str,
    context_summary: str = "",
    cancel_flag: Optional[asyncio.Event] = None,
    thread_id: Optional[str] = None
) -> AsyncGenerator[Dict[str, str], None]:

    graph = get_agent_graph()
    
    initial_state = AgentState(
        messages=[HumanMessage(content=user_message)],
        user_id=user_id,
        user_name=user_name,
        context_summary=context_summary,
        next=""
    )
    
    config = {"configurable": {"thread_id": thread_id}} if thread_id else None

    logger.info(f"🚀 Starting streaming graph execution for thread_id={thread_id}")

    try:
        # Use astream_events to get tokens and tool events
        async for event in graph.astream_events(initial_state, config=config, version="v1"):
            
            if cancel_flag and cancel_flag.is_set():
                yield {"type": "cancelled", "content": "Stream cancelled"}
                return

            event_type = event.get("event")
            
            # 1. Handle LLM Streaming Tokens
            if event_type == "on_chat_model_stream":
                data = event.get("data", {})
                chunk = data.get("chunk")
                
                # Only yield content tokens
                if hasattr(chunk, "content") and chunk.content:
                    yield {"type": "token", "content": chunk.content}

            # 2. Handle Tool Execution
            elif event_type == "on_tool_start":
                tool_name = event.get("name")
                if tool_name and tool_name not in ["_Exception", "LangGraph"]:
                    logger.info(f"🔧 Tool started: {tool_name}")
                    yield {"type": "intent", "content": tool_name}
            
    except Exception as e:
        logger.error(f"❌ Streaming error: {e}", exc_info=True)
        yield {"type": "error", "content": str(e)}

    yield {"type": "done", "content": ""}
