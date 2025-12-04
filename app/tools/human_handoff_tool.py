"""
app/tools/human_handoff_tool.py
Generic Human Handoff Tool for dynamic escalation based on severity and emotion.
"""

from langchain.tools import tool
import logging
from typing import Literal

logger = logging.getLogger(__name__)

@tool
def human_handoff_tool(
    issue_summary: str,
    severity: Literal["Low", "Medium", "High", "Critical"],
    user_emotion: str
) -> str:
    """
    Escalates the conversation to a human agent when the AI cannot handle the request 
    or when the user is distressed/demanding a human.

    Args:
        issue_summary: A brief summary of the user's issue or request.
        severity: The perceived severity of the issue.
                  - "Low": General questions, minor complaints.
                  - "Medium": Recurring issues, frustration.
                  - "High": Angry user, service outage, financial dispute.
                  - "Critical": Safety threats, legal threats, extreme distress.
        user_emotion: The detected emotion of the user (e.g., "Frustrated", "Angry", "Anxious", "Neutral", "Happy").

    Returns:
        A message confirming the handoff and next steps.
    """
    
    logger.info(f"🚨 Human Handoff Triggered")
    logger.info(f"   Summary: {issue_summary}")
    logger.info(f"   Severity: {severity}")
    logger.info(f"   Emotion: {user_emotion}")

    # Simulate backend ticketing/alerting logic
    ticket_id = f"TICKET-{abs(hash(issue_summary)) % 100000}"
    
    response = ""

    if severity == "Critical":
        response = f"""🚨 **IMMEDIATE ESCALATION TRIGGERED**

I have immediately alerted our Senior Support Team regarding this critical issue.

**Issue:** {issue_summary}
**Priority:** CRITICAL
**Ticket ID:** {ticket_id}

A human supervisor has been notified and will join this chat or contact you within **5 minutes**.

Please stay online."""

    elif severity == "High":
        response = f"""⚠️ **Escalating to Human Specialist**

I understand this is important and requires human attention. I've flagged this for immediate review.

**Issue:** {issue_summary}
**Status:** High Priority
**Ticket ID:** {ticket_id}

A specialist will review your case and respond within **1 hour**. 

Is there anything else you'd like to add to the ticket before they review it?"""

    elif severity == "Medium":
        response = f"""👤 **Connecting you with Support**

I see you're feeling {user_emotion.lower()}. I'm passing this conversation to a support agent who can help you better.

**Ticket Created:** {ticket_id}
**Topic:** {issue_summary}

An agent will be with you shortly (estimated wait: 2-4 hours)."""

    else: # Low
        response = f"""📋 **Support Ticket Created**

I've created a support ticket for your request: "{issue_summary}".

**Ticket ID:** {ticket_id}

Our team reviews these requests daily. You can expect a response via email within 24 hours. 

Can I help you with anything else in the meantime?"""

    return response
