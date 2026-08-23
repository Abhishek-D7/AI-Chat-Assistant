
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage
from app.state import AgentState
from app.tools.booking_tool import booking_agent_tool
from app.config import Config
from app.agent_config import AGENTS_CONFIG

def create_booking_agent(llm: ChatOpenAI): 
    """Create the Booking Agent sub-graph"""
    
    tools = [booking_agent_tool]
    tool_node = ToolNode(tools)
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)
    
    def booking_node(state: AgentState):
        messages = state["messages"]
        # Add system prompt if not present or just rely on the supervisor's context
        # But it's better to have a specific persona
        system_prompt = SystemMessage(content=AGENTS_CONFIG["booking_agent"]["system_prompt"])
        
        # Filter messages? Or just pass all? Passing all for context.
        response = llm_with_tools.invoke([system_prompt] + messages)
        return {"messages": [response]}

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls: 
            return "tools"
        return "__end__"

    workflow = StateGraph(AgentState)
    workflow.add_node("booking_agent", booking_node)
    workflow.add_node("tools", tool_node)
    
    workflow.add_edge(START, "booking_agent")
    workflow.add_conditional_edges("booking_agent", should_continue, {"tools": "tools", "__end__": END})
    workflow.add_edge("tools", "booking_agent")
    
    return workflow.compile()
