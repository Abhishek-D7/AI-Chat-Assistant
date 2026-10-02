import re
import logging
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.state import AgentState
from app.tools.similarity_search_tool import similarity_search_tool
from app.tools.human_handoff_tool import human_handoff_tool
from app.config import Config
from app.agent_config import AGENTS_CONFIG

from app.ml.llm_client import invoke_resilient_chat, clean_llm_text

logger = logging.getLogger(__name__)

def create_support_agent(llm: ChatOpenAI = None):
    """
    Create the Support Agent sub-graph.
    Retrieves vector knowledge from Pinecone (top_k=4) via Hugging Face embeddings
    and generates grounded, structured responses resilient across all OpenRouter models.
    """
    
    def support_node(state: AgentState):
        messages = state.get("messages", [])

        # 1. Extract latest user question/query
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

        user_query_clean = (user_query or "").strip()
        q_lower = user_query_clean.lower()

        # 2. Check for explicit human handoff request (avoid matching 'agentic AI')
        handoff_triggers = [
            "talk to human", "real person", "human agent", "talk to someone", 
            "customer care representative", "speak to human", "escalate to human",
            "connect to human", "human support"
        ]
        if any(trigger in q_lower for trigger in handoff_triggers):
            logger.info("🙋 [support_agent] Escalating to human handoff")
            handoff_msg = human_handoff_tool.invoke({
                "issue_summary": user_query_clean,
                "severity": "Medium",
                "user_emotion": "Neutral"
            })
            return {"messages": [AIMessage(content=handoff_msg)]}

        # 3. Retrieve Top_K = 4 results from Pinecone vector database
        retrieved_context = ""
        if user_query_clean:
            try:
                from app.main import persistence
                matches = persistence.similarity_search(query=user_query_clean, top_k=4, min_score=0.15)
                if matches:
                    blocks = []
                    for i, match in enumerate(matches, 1):
                        source = match.get("source", "Uploaded Document")
                        text = match.get("text", "").strip()
                        score = match.get("score", 0.0)
                        blocks.append(f"[{i}] Source: {source} (Relevance: {score * 100:.1f}%)\n{text}")
                    retrieved_context = "\n\n".join(blocks)
                    logger.info(f"📚 [support_agent] Injected {len(matches)} vector chunks into prompt for query: '{user_query_clean[:35]}'")
            except Exception as search_err:
                logger.warning(f"⚠️ [support_agent] Vector search during node execution skipped: {search_err}")

        # 4. Construct grounded system prompt
        base_prompt = (
            "You are a professional, helpful, and articulate AI Support Assistant.\n"
            "Format your responses cleanly using Markdown (use bold headings, bullet points, and numbered lists where appropriate).\n"
            "Always be direct, informative, and courteous."
        )

        if retrieved_context:
            augmented_prompt = (
                f"{base_prompt}\n\n"
                f"=== RETRIEVED KNOWLEDGE BASE CONTEXT (TOP 4 MATCHES) ===\n"
                f"{retrieved_context}\n"
                f"========================================================\n\n"
                f"Instructions:\n"
                f"1. Synthesize a comprehensive answer grounded in the retrieved document context above.\n"
                f"2. Cite the source document name when referencing factual details.\n"
                f"3. If the context does not fully answer the question, supplement with helpful general knowledge while clearly distinguishing internal documents from general knowledge."
            )
        else:
            augmented_prompt = (
                f"{base_prompt}\n\n"
                f"Instructions:\n"
                f"Answer the user's question directly, clearly, and comprehensively using your knowledge.\n"
                f"Use structured formatting with sections or bullet points for readability."
            )

        # 5. Invoke LLM resiliently with automatic fallbacks and safety tag interception
        system_msg = SystemMessage(content=augmented_prompt)
        response = invoke_resilient_chat([system_msg] + messages)
        
        # 6. Clean reasoning / think tags if any
        if hasattr(response, "content") and response.content:
            response.content = clean_llm_text(str(response.content))

        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("support_agent", support_node)
    workflow.add_edge(START, "support_agent")
    workflow.add_edge("support_agent", END)
    
    return workflow.compile()
