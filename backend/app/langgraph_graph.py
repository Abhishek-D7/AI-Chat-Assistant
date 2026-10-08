import re
import os
import logging
import asyncio
import json
from typing import TypedDict, Annotated, Literal, Optional, Dict, AsyncGenerator, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()
from app.config import Config
from app.cache import SystemMessageCache
from app.state import AgentState
from app.agent_config import AGENTS_CONFIG, AGENT_NAMES
from app.agents import create_booking_agent, create_support_agent

from app.ml.llm_client import get_resilient_llm, clean_llm_text, is_safety_or_invalid_output

logger = logging.getLogger(__name__)

# Initialize system message cache
system_message_cache = SystemMessageCache(max_size=Config.SYSTEM_MESSAGE_CACHE_SIZE)

agent_graph = None

# SUPERVISOR NODE (Deterministic Hybrid Routing for Free/Random Models)

def create_agent_graph():
    """Create the Hierarchical Agent Graph with resilient routing and fallbacks."""
    
    logger.info("🔨 Starting graph creation (Hierarchical)...")
    
    # Initialize resilient LLM with multi-model fallbacks
    logger.info("☁️ Initializing Resilient Free LLM client")
    llm = get_resilient_llm(streaming=True)
    
    # Create Sub-Agents
    booking_agent_graph = create_booking_agent(llm)
    support_agent_graph = create_support_agent(llm)
    
    def supervisor_node(state: AgentState):
        messages = state.get("messages", [])
        last_message = messages[-1] if messages else None
        
        # 1. If an agent has already produced an AIMessage with content, conversation turn is complete!
        if isinstance(last_message, AIMessage) and getattr(last_message, "content", ""):
            logger.info("🚦 Supervisor: Agent response completed -> FINISH")
            return {"next": "FINISH"}
            
        # 2. Extract latest user query text
        user_query = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                user_query = m.content
                break
            elif hasattr(m, "content") and getattr(m, "type", "") == "human":
                user_query = m.content
                break
            elif isinstance(m, dict) and m.get("role") == "user":
                user_query = m.get("content", "")
                break
        
        q_lower = (user_query or "").lower().strip()
        
        # 3. Deterministic intent routing (resilient against unpredictable free OpenRouter models)
        booking_keywords = [
            "book", "booking", "schedule", "appointment", "calendar", 
            "meeting", "consultation", "reschedule", "reserve", "slot"
        ]
        is_booking = any(kw in q_lower for kw in booking_keywords)
        
        if is_booking:
            next_agent = "BookingAgent"
        else:
            # Default for all questions, document queries, and general inquiries
            next_agent = "SupportAgent"
        
        logger.info(f"🚦 Supervisor routed user query to: {next_agent}")
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
    
    # Compile Graph (checkpointer is handled by LangGraph API/Studio or caller)
    compiled_graph = workflow.compile()
    compiled_graph._llm = llm
    
    return compiled_graph

def get_agent_graph():
    """Get or create singleton graph"""
    global agent_graph
    if agent_graph is None:
        agent_graph = create_agent_graph()
    return agent_graph

# Expose compiled graph for LangGraph Studio / CLI / LangSmith
graph = get_agent_graph()

# NON-STREAMING HANDLER

def process_user_message_with_context(
    user_message: str,
    user_id: str,
    user_name: str,
    context_summary: str = "",
    thread_id: Optional[str] = None,
    guardrail_settings: Optional[Dict[str, bool]] = None
) -> Dict[str, Any]:
    
    try:
        from app.guardrails import guardrail_manager
        
        # 1. Evaluate Input Guardrails (Prompt Injection, PII, Length)
        input_guard_res = guardrail_manager.check_input(user_message, guardrail_settings)
        if input_guard_res.blocked:
            logger.warning(f"🛡️ Message blocked by guardrail: {input_guard_res.guardrail}")
            return {
                "bot_response": input_guard_res.notification_message,
                "intent": "guardrail_blocked",
                "guardrail_triggered": True,
                "guardrail_info": input_guard_res.to_dict()
            }
        
        effective_message = input_guard_res.sanitized_input or user_message
        
        graph = get_agent_graph()
        
        msg_list = []
        if context_summary:
            msg_list.append(SystemMessage(content=f"Context:\n{context_summary}"))
        msg_list.append(HumanMessage(content=effective_message))
        
        initial_state = AgentState(
            messages=msg_list,
            user_id=user_id,
            user_name=user_name,
            context_summary=context_summary,
            next=""
        )
        active_thread_id = thread_id or f"thread_{user_id}"
        config = {"configurable": {"thread_id": active_thread_id}}
        
        result = graph.invoke(initial_state, config=config)

        # Find the last AIMessage with valid content
        response_text = ""
        intent = "general"
        
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and getattr(msg, "content", ""):
                candidate = clean_llm_text(str(msg.content))
                if not is_safety_or_invalid_output(candidate):
                    response_text = candidate
                    break
            elif hasattr(msg, "content") and getattr(msg, "type", "") == "ai":
                candidate = clean_llm_text(str(msg.content))
                if not is_safety_or_invalid_output(candidate):
                    response_text = candidate
                    break

        if not response_text:
            response_text = "I processed your request, but no response text was returned. Please try rephrasing your question."

        # 2. Output Guardrails (Secret & Credential Leak Filter)
        clean_response, output_guard_res = guardrail_manager.check_output(response_text, guardrail_settings)

        # 3. Detect if response is a tool guardrail notification
        is_tool_guardrail = (
            "🛡️ **Guardrail" in clean_response or 
            "outside our business hours" in clean_response.lower() or
            "falls outside our business hours" in clean_response.lower()
        )
        guardrail_triggered = is_tool_guardrail or input_guard_res.warning_only or bool(output_guard_res)
        
        guardrail_info = None
        if input_guard_res.warning_only:
            guardrail_info = input_guard_res.to_dict()
        elif output_guard_res:
            guardrail_info = output_guard_res.to_dict()
        elif is_tool_guardrail:
            guardrail_info = {
                "blocked": True,
                "guardrail": "booking_rules",
                "guardrail_name": "Working Hours & Booking Rules",
                "reason": "Appointment policy or working hours constraint violated",
                "suggestion": "Select a slot Monday-Friday between 9:00 AM and 6:00 PM."
            }

        # Detect active agent intent
        booking_keywords = [
            "book", "booking", "schedule", "appointment", "calendar", 
            "meeting", "consultation", "reschedule", "reserve", "slot"
        ]
        is_booking = any(kw in user_message.lower() for kw in booking_keywords)
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and any("booking" in getattr(tc, "name", tc.get("name", "")).lower() for tc in getattr(msg, "tool_calls", [])):
                is_booking = True
                break

        intent = "BookingAgent" if is_booking else "SupportAgent"

        return {
            "bot_response": clean_response,
            "intent": intent,
            "guardrail_triggered": guardrail_triggered,
            "guardrail_info": guardrail_info
        }
    except Exception as e:
        logger.error(f"❌ Error in process_user_message_with_context: {e}", exc_info=True)
        return {"bot_response": f"An error occurred while processing: {str(e)}", "intent": "error"}

# STREAMING HANDLER (With Instant Fallback for Non-Streaming Models)

async def process_user_message_with_context_streaming(
    user_message: str,
    user_id: str,
    user_name: str,
    context_summary: str = "",
    cancel_flag: Optional[asyncio.Event] = None,
    thread_id: Optional[str] = None,
    guardrail_settings: Optional[Dict[str, bool]] = None
) -> AsyncGenerator[Dict[str, Any], None]:

    from app.guardrails import guardrail_manager

    # 1. Evaluate Input Guardrails
    input_guard_res = guardrail_manager.check_input(user_message, guardrail_settings)
    if input_guard_res.blocked:
        logger.warning(f"🛡️ Streaming message blocked by guardrail: {input_guard_res.guardrail}")
        yield {"type": "intent", "content": "guardrail_blocked"}
        yield {"type": "guardrail", "data": input_guard_res.to_dict()}
        yield {"type": "content", "content": input_guard_res.notification_message}
        yield {"type": "done"}
        return

    effective_message = input_guard_res.sanitized_input or user_message

    if input_guard_res.warning_only:
        yield {"type": "guardrail", "data": input_guard_res.to_dict()}

    graph = get_agent_graph()
    
    # Emit intent early so UI labels the agent correctly
    booking_keywords = [
        "book", "booking", "schedule", "appointment", "calendar", 
        "meeting", "consultation", "reschedule", "reserve", "slot"
    ]
    initial_intent = "BookingAgent" if any(kw in effective_message.lower() for kw in booking_keywords) else "SupportAgent"
    yield {"type": "intent", "content": initial_intent}
    
    msg_list = []
    if context_summary:
        msg_list.append(SystemMessage(content=f"Context:\n{context_summary}"))
    msg_list.append(HumanMessage(content=effective_message))
    
    initial_state = AgentState(
        messages=msg_list,
        user_id=user_id,
        user_name=user_name,
        context_summary=context_summary,
        next=""
    )
    
    active_thread_id = thread_id or f"thread_{user_id}"
    config = {"configurable": {"thread_id": active_thread_id}}

    logger.info(f"🚀 Starting streaming graph execution for thread_id={active_thread_id}")

    yielded_tokens = 0
    in_think_tag = False

    try:
        # Use astream_events to stream tokens as they arrive
        async for event in graph.astream_events(initial_state, config=config, version="v2"):
            
            if cancel_flag and cancel_flag.is_set():
                yield {"type": "cancelled", "content": "Stream cancelled"}
                return

            event_type = event.get("event")
            
            # 1. Handle LLM Streaming Tokens
            if event_type == "on_chat_model_stream":
                data = event.get("data", {})
                chunk = data.get("chunk")
                
                if hasattr(chunk, "content") and chunk.content:
                    text_chunk = str(chunk.content)
                    
                    # Filter out <think> tags if reasoning model streams them
                    if "<think>" in text_chunk:
                        in_think_tag = True
                    if "</think>" in text_chunk:
                        in_think_tag = False
                        text_chunk = text_chunk.split("</think>")[-1]
                    
                    if not in_think_tag and text_chunk:
                        yielded_tokens += 1
                        yield {"type": "token", "content": text_chunk}

            # 2. Fallback if model didn't stream token-by-token but finished with output
            elif event_type == "on_chat_model_end":
                if yielded_tokens == 0:
                    data = event.get("data", {})
                    output = data.get("output")
                    if output and hasattr(output, "content") and output.content:
                        clean_text = clean_llm_text(str(output.content))
                        if clean_text:
                            yielded_tokens += 1
                            yield {"type": "token", "content": clean_text}

            # 3. Handle Tool Execution Notification
            elif event_type == "on_tool_start":
                tool_name = event.get("name")
                if tool_name and tool_name not in ["_Exception", "LangGraph"]:
                    logger.info(f"🔧 Tool started: {tool_name}")
                    yield {"type": "intent", "content": tool_name}
            
    except Exception as e:
        logger.error(f"❌ Streaming error: {e}", exc_info=True)
        # If streaming errored, attempt non-streaming fallback
        if yielded_tokens == 0:
            try:
                res = await asyncio.to_thread(
                    process_user_message_with_context,
                    user_message, user_id, user_name, context_summary, thread_id
                )
                yield {"type": "token", "content": res.get("bot_response", "")}
                yielded_tokens += 1
            except Exception:
                yield {"type": "error", "content": str(e)}

    # Fail-safe: If zero tokens were delivered during streaming, run non-streaming directly
    if yielded_tokens == 0:
        logger.warning("⚠️ No tokens were yielded during streaming. Invoking non-streaming fallback.")
        try:
            res = await asyncio.to_thread(
                process_user_message_with_context,
                user_message, user_id, user_name, context_summary, thread_id
            )
            bot_text = res.get("bot_response", "")
            if bot_text:
                yield {"type": "token", "content": bot_text}
        except Exception as fb_err:
            logger.error(f"❌ Fallback failed: {fb_err}")
            yield {"type": "token", "content": "Unable to generate a response at this moment. Please try again."}

    yield {"type": "done", "content": ""}
