
from typing import TypedDict, Annotated, List, Optional
import operator
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    """State that flows through the graph"""
    messages: Annotated[list, operator.add]
    user_id: str
    user_name: str
    context_summary: str
    next: str # For supervisor routing
