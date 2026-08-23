"""
app/__init__.py
Package initialization for AI Chat Assistant
"""

__version__ = "2.0.0"
__author__ = "B2B AI Chat Team"

from app.main import app
from app.langgraph_graph import create_agent_graph

__all__ = ["app", "create_agent_graph"]
