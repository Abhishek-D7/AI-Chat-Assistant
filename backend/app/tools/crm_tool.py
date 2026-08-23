"""
app/tools/crm_tool.py
Onboarding Agent Tool + CRM Management
"""

from langchain.tools import tool
import uuid
import random
import string
from datetime import datetime
from typing import Dict


class CRMTool:
    """Simple CRM for user management"""
    
    def __init__(self):
        self.users = {}
    
    def create_user(self, user_id: str, details: Dict) -> Dict:
        """Create new user in CRM"""
        user = {
            "id": f"CRM_{uuid.uuid4().hex[:8]}",
            "user_id": user_id,
            "details": details,
            "created_at": datetime.now().isoformat(),
            "status": "active"
        }
        self.users[user_id] = user
        return user
    
    def generate_otp(self, phone: str) -> str:
        """Generate demo OTP"""
        return ''.join(random.choices(string.digits, k=6))


# Global CRM instance
crm = CRMTool()


@tool
def onboarding_agent_tool(user_message: str) -> str:
    """
    Onboarding Agent - Handles new user registration and setup.
    
    Use this tool when the user:
    - Is new and wants to sign up or register
    - Needs account creation or onboarding help
    - Says "I'm new", "sign up", "register", "create account"
    
    Args:
        user_message: The user's onboarding request
        
    Returns:
        Onboarding guidance and next steps
    """
    
    # Generate demo OTP
    otp = crm.generate_otp("+1234567890")
    
    response = f"""👋 **Welcome to B2B Business Solutions!**

I'm excited to help you get started! Let's set up your account.

**📋 Quick Onboarding (3 Steps):**

**Step 1/3:** Please provide:
• Your full name
• Email address
• Phone number
• Company name

You can share all at once or one by one!

---

**For Demo:** Here's your verification code: **{otp}**
(In production, this would be sent via SMS)

---

Once verified, I'll help you:
✅ Set up your account
✅ Choose the right service package
✅ Schedule an onboarding call with our team

Ready to begin? Share your details! 🚀"""
    
    return response


@tool
def crm_agent_tool(user_message: str) -> str:
    """
    Performs operations related to the Customer Relationship Management (CRM) system, 
    such as creating new leads, updating customer information, or querying CRM data 
    based on the user's message.
    """
    if "add" in user_message and "@" in user_message:
        return "CRM entry added for {}! Do you wish to attach notes?".format(user_message)
    return "Let me know the contact details you'd like to add to CRM."

@tool
def general_agent_tool(user_message: str) -> str:
    """
    General Agent - Handles general conversation and ambiguous queries.
    
    Use this tool for:
    - General conversation, greetings, small talk
    - Unclear or ambiguous queries
    - Follow-up questions
    - Anything not matching specific categories
    
    Args:
        user_message: The user's general message
        
    Returns:
        Conversational response
    """
    
    response = """Hello! 👋

I'm your B2B AI assistant. I can help you with:

📅 **Schedule meetings** - Book time with our team
❓ **Answer questions** - Learn about our services
🚀 **Get started** - New user onboarding
🆘 **Get support** - Crisis assistance if needed

What would you like to do today?"""
    
    return response
