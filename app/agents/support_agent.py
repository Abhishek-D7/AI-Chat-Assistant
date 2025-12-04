
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from app.state import AgentState
from app.tools.faq_tool import faq_agent_tool
from app.tools.human_handoff_tool import human_handoff_tool
from app.config import Config
from app.agent_config import AGENTS_CONFIG

def create_support_agent(llm: ChatOpenAI):
    """Create the Support Agent sub-graph"""
    
    tools = [faq_agent_tool, human_handoff_tool]
    tool_node = ToolNode(tools)
    
    llm_with_tools = llm.bind_tools(tools)
    
    def support_node(state: AgentState):
        messages = state["messages"]
        system_prompt = SystemMessage(content=AGENTS_CONFIG["support_agent"]["system_prompt"])
        response = llm_with_tools.invoke([system_prompt] + messages)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "__end__"

    workflow = StateGraph(AgentState)
    workflow.add_node("support_agent", support_node)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "support_agent")
    workflow.add_conditional_edges("support_agent", should_continue, {"tools": "tools", "__end__": END})
    workflow.add_edge("tools", "support_agent")
    
    return workflow.compile()
