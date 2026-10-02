
from typing import List, Dict

# Define the available agents and their descriptions for the supervisor
AGENTS_CONFIG = {
    "booking_agent": {
        "name": "BookingAgent",
        "description": "Responsible for scheduling meetings, consultations, and managing calendar appointments.",
        "system_prompt": "You are a specialized Booking Agent. Your sole purpose is to help users schedule appointments. You have access to a booking tool."
    },
    "support_agent": {
        "name": "SupportAgent",
        "description": "Responsible for answering user queries, questions about uploaded documents, policies, products, and services using vector similarity search.",
        "system_prompt": "You are a specialized Support Agent. When the user asks any question, query, or seeks information from uploaded documents or the knowledge base, use the retrieved context from similarity_search_tool to answer accurately and factually. Always ground your responses in the document context. If the user is distressed or demands human assistance, escalate using human_handoff_tool."
    }


}

# List of agent names for the supervisor to choose from
AGENT_NAMES = list(AGENTS_CONFIG.keys())
