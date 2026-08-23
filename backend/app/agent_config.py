
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
        "description": "Responsible for answering FAQs, technical support questions, and handling human handoffs.",
        "system_prompt": "You are a specialized Support Agent. You answer questions using the FAQ tool and can escalate to a human if needed."
    }
}

# List of agent names for the supervisor to choose from
AGENT_NAMES = list(AGENTS_CONFIG.keys())
