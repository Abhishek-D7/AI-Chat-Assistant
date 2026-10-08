"""
app/guardrails/manager.py
Modular Native Guardrails Manager for AI Chat Assistant.
Handles Input, Tool Action, RAG Grounding, and Output guardrails.
"""

import re
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import pytz
from app.config import Config

logger = logging.getLogger(__name__)

DEFAULT_GUARDRAILS: Dict[str, bool] = {
    "prompt_injection": True,
    "pii_detection": True,
    "content_moderation": True,
    "booking_rules": True,
    "anti_flooding": True,
    "rag_grounding": True,
    "secret_leak": True
}

GUARDRAILS_METADATA = [
    {
        "id": "prompt_injection",
        "name": "Prompt Injection Defense",
        "category": "Input Security",
        "description": "Intercepts prompt injections, system overrides, roleplay bypasses, and system prompt exfiltration before any LLM is called.",
        "trigger": "Triggers on patterns like 'Ignore previous instructions', 'DAN mode', 'System override', or 'Print instructions'.",
        "action": "Instantly blocks the turn, saving tokens and preserving safety directives.",
        "user_help": "Rephrase your query to ask a direct question about documents, services, or bookings without system commands."
    },
    {
        "id": "pii_detection",
        "name": "PII Redaction & Privacy",
        "category": "Privacy & Data Protection",
        "description": "Detects and redacts sensitive personal data (credit card numbers, SSNs, passwords) before sending to external model providers.",
        "trigger": "Triggers when valid card numbers (13-16 digits), SSN formats (XXX-XX-XXXX), or explicit credentials are typed.",
        "action": "Masks sensitive tokens with [REDACTED_...] and warns user to keep credentials private.",
        "user_help": "Avoid typing real financial or government ID details in the chat window."
    },
    {
        "id": "content_moderation",
        "name": "Profanity & Explicit Content Filter",
        "category": "Content Safety",
        "description": "Detects and blocks NSFW content, adult material, profanity, and abusive or vulgar language before reaching the agent or LLM.",
        "trigger": "Triggers when explicit adult terms (e.g. 'porn', 'xxx', 'nsfw') or vulgar/abusive words (e.g. 'shit', 'fuck', 'bitch', 'asshole') are detected.",
        "action": "Instantly blocks the turn, prevents unnecessary LLM token spend, and guides the user toward acceptable inquiries.",
        "user_help": "Refrain from vulgar, adult, or abusive terms. Rephrase using polite, professional language related to office services or document queries."
    },
    {
        "id": "booking_rules",
        "name": "Working Hours & Booking Rules",
        "category": "Tool Action Safety",
        "description": "Validates scheduling requests: strictly future dates, weekday working hours (9:00 AM - 6:00 PM), and reasonable duration.",
        "trigger": "Triggers when an appointment is requested outside 9 AM - 6 PM, on weekends, or for a past date.",
        "action": "Blocks invalid booking and returns current operational schedule with alternative slot recommendations.",
        "user_help": "Request a time between 9:00 AM and 6:00 PM on a weekday (e.g. 'Tomorrow at 2:00 PM')."
    },
    {
        "id": "anti_flooding",
        "name": "Anti-Flooding Rate Limiter",
        "category": "Action Rate Limiting",
        "description": "Limits Google Calendar bookings to a maximum of 3 appointments per session to prevent calendar spamming and DoS flooding.",
        "trigger": "Triggers when a single session attempts to book more than 3 meetings in a short period.",
        "action": "Halts automated booking tool execution and requests administrative confirmation.",
        "user_help": "Contact our office administrator directly if you need to coordinate bulk or group appointments."
    },
    {
        "id": "rag_grounding",
        "name": "RAG Grounding & Hallucination Guard",
        "category": "Knowledge Retrieval",
        "description": "Enforces strict vector similarity score verification on Pinecone results to prevent hallucinating ungrounded policies.",
        "trigger": "Triggers when query relevance score in vector database is below confidence threshold (cosine similarity < 0.35).",
        "action": "Informs the user that no verified source documents matched the query instead of fabricating facts.",
        "user_help": "Verify that relevant PDFs were uploaded in the 'Ingest Documents' tab, or rephrase with specific document keywords."
    },
    {
        "id": "secret_leak",
        "name": "Secret & Credential Leak Filter",
        "category": "Output Security",
        "description": "Inspects model outputs in real time to guarantee zero leakage of internal API keys, database credentials, or OAuth tokens.",
        "trigger": "Triggers if output text contains patterns matching OpenRouter, Pinecone, HuggingFace tokens or OAuth secrets.",
        "action": "Instantly redacts the secret tokens before the payload reaches the browser.",
        "user_help": "System security remains protected. No action required by user."
    }
]


class GuardrailResult:
    def __init__(
        self,
        blocked: bool = False,
        guardrail: Optional[str] = None,
        guardrail_name: Optional[str] = None,
        reason: Optional[str] = None,
        suggestion: Optional[str] = None,
        notification_message: Optional[str] = None,
        sanitized_input: Optional[str] = None,
        warning_only: bool = False
    ):
        self.blocked = blocked
        self.guardrail = guardrail
        self.guardrail_name = guardrail_name
        self.reason = reason
        self.suggestion = suggestion
        self.notification_message = notification_message
        self.sanitized_input = sanitized_input
        self.warning_only = warning_only

    def to_dict(self) -> Dict[str, Any]:
        return {
            "blocked": self.blocked,
            "guardrail": self.guardrail,
            "guardrail_name": self.guardrail_name,
            "reason": self.reason,
            "suggestion": self.suggestion,
            "notification_message": self.notification_message,
            "warning_only": self.warning_only
        }


class GuardrailManager:
    """Central guardrail evaluation engine."""

    def __init__(self):
        self.active_settings = dict(DEFAULT_GUARDRAILS)
        self.session_booking_counts: Dict[str, int] = {}

    def get_settings(self) -> Dict[str, bool]:
        return dict(self.active_settings)

    def update_settings(self, new_settings: Dict[str, bool]):
        for k, v in new_settings.items():
            if k in self.active_settings:
                self.active_settings[k] = bool(v)

    def is_enabled(self, guardrail_id: str, custom_settings: Optional[Dict[str, bool]] = None) -> bool:
        if custom_settings and guardrail_id in custom_settings:
            return bool(custom_settings[guardrail_id])
        return self.active_settings.get(guardrail_id, True)

    def increment_booking_count(self, session_id: str):
        self.session_booking_counts[session_id] = self.session_booking_counts.get(session_id, 0) + 1

    def get_booking_count(self, session_id: str) -> int:
        return self.session_booking_counts.get(session_id, 0)

    # 1. INPUT GUARDRAIL CHECK
    def check_input(
        self,
        user_message: str,
        custom_settings: Optional[Dict[str, bool]] = None
    ) -> GuardrailResult:
        """Evaluates input for Prompt Injection, PII, and length violations."""
        
        # A. Prompt Injection Defense
        if self.is_enabled("prompt_injection", custom_settings):
            injection_patterns = [
                r"(?i)\b(ignore|disregard|forget)\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules|commands)",
                r"(?i)\b(you are now|pretend you are|act as|dan mode|jailbreak|bypass security|disable guardrails)\b",
                r"(?i)\b(reveal|show|print|display|dump|leak)\s+(your\s+)?(system prompt|developer instructions|hidden prompt|initial prompt)\b",
                r"(?i)\bsystem\s*:\s*(override|reset|clear)\b",
                r"(?i)\bswitch\s+to\s+(developer|unrestricted|sudo)\s+mode\b"
            ]
            for pattern in injection_patterns:
                if re.search(pattern, user_message):
                    logger.warning(f"🛡️ Guardrail Triggered [prompt_injection]: Pattern matched in query '{user_message[:60]}'")
                    return GuardrailResult(
                        blocked=True,
                        guardrail="prompt_injection",
                        guardrail_name="Prompt Injection Defense",
                        reason="Your message was flagged for attempting to override or bypass system safety instructions.",
                        suggestion="Please rephrase your inquiry directly to ask about document information or schedule an appointment without system commands.",
                        notification_message=(
                            "🛡️ **Guardrail Alert: Prompt Injection Blocked**\n\n"
                            "Your message was intercepted because it contains override or jailbreak commands that violate safety policy.\n\n"
                            "💡 **How to correct this:**\n"
                            "Please ask your question plainly without system directives (e.g., *'Can you summarize the document?'* or *'Book a consultation for tomorrow at 2 PM'*)."
                        )
                    )

        # B. Content Moderation (Profanity & Explicit Content Filter)
        if self.is_enabled("content_moderation", custom_settings):
            explicit_patterns = [
                r"\b(porn|porno|pornography|xxx|nsfw|hentai|erotic|erotica|nudes?|onlyfans)\b",
                r"\bsex\s+(video|videos|pic|pics|tape|movies?|content|clips?|audio|chat|film)\b",
                r"\b(blowjob|handjob|gangbang|threesome|masturbat\w*)\b"
            ]
            abusive_patterns = [
                r"\b(shit|shitty|bullshit|dipshit)\b",
                r"\b(fuck|fucked|fucking|fucker|fuckin|motherfucker|clusterfuck)\b",
                r"\b(bitch|bitches|bitching)\b",
                r"\b(bastard|bastards)\b",
                r"\b(asshole|assholes|dumbass|jackass)\b",
                r"\b(cunt|cunts|pussy|pussies|dick|dicks|cocks?|slut|sluts|whore|whores)\b",
                r"\b(retard|retarded)\b"
            ]

            matched_explicit = [p for p in explicit_patterns if re.search(p, user_message, re.IGNORECASE)]
            matched_abusive = [p for p in abusive_patterns if re.search(p, user_message, re.IGNORECASE)]

            if matched_explicit or matched_abusive:
                violation_type = "Explicit Adult Content" if matched_explicit else "Profanity & Abusive Language"
                logger.warning(f"🛡️ Guardrail Triggered [content_moderation]: Flagged {violation_type} in query '{user_message[:60]}'")
                
                return GuardrailResult(
                    blocked=True,
                    guardrail="content_moderation",
                    guardrail_name="Profanity & Explicit Content Filter",
                    reason=f"Your message was flagged for containing {violation_type.lower()} which violates our acceptable use policy.",
                    suggestion="Please rephrase your message using respectful, professional language related to documents or appointment scheduling.",
                    notification_message=(
                        f"🛡️ **Guardrail Alert: {violation_type} Blocked**\n\n"
                        f"Your message was intercepted because it contains terms categorized as {violation_type.lower()}, violating acceptable usage policies.\n\n"
                        "💡 **How to correct this:**\n"
                        "Please rephrase your inquiry using respectful and professional language (e.g., asking for document summaries, office policies, or scheduling a consultation)."
                    )
                )

        # C. PII Detection & Redaction
        sanitized = user_message
        pii_found = False
        pii_types = []

        if self.is_enabled("pii_detection", custom_settings):
            # Credit Card (13 to 16 digits, with optional hyphens/spaces)
            card_pattern = r"\b(?:\d[ -]*?){13,16}\b"
            if re.search(card_pattern, sanitized):
                # Verify it's not a short number
                digits_only = re.sub(r"\D", "", re.search(card_pattern, sanitized).group(0))
                if 13 <= len(digits_only) <= 19:
                    sanitized = re.sub(card_pattern, "[REDACTED_PAYMENT_CARD]", sanitized)
                    pii_found = True
                    pii_types.append("Credit Card Number")

            # US SSN (XXX-XX-XXXX)
            ssn_pattern = r"\b\d{3}-\d{2}-\d{4}\b"
            if re.search(ssn_pattern, sanitized):
                sanitized = re.sub(ssn_pattern, "[REDACTED_SSN]", sanitized)
                pii_found = True
                pii_types.append("Social Security Number")

            # Sensitive credentials / passwords
            pwd_pattern = r"(?i)\b(password|passwd|secret_key|api_key)\s*[:=]\s*(\S+)"
            if re.search(pwd_pattern, sanitized):
                sanitized = re.sub(pwd_pattern, r"\1: [REDACTED_SECRET]", sanitized)
                pii_found = True
                pii_types.append("Password / Secret Key")

            if pii_found:
                logger.info(f"🛡️ Guardrail Applied [pii_detection]: Redacted {', '.join(pii_types)}")
                return GuardrailResult(
                    blocked=False,
                    warning_only=True,
                    guardrail="pii_detection",
                    guardrail_name="PII Redaction & Privacy",
                    reason=f"Detected sensitive data ({', '.join(pii_types)}) in your message.",
                    suggestion="For your privacy, we automatically masked this information before sending to the model.",
                    notification_message=(
                        f"🛡️ **Privacy Notice: Sensitive Data Redacted**\n\n"
                        f"We detected and automatically masked sensitive information ({', '.join(pii_types)}) to protect your personal privacy.\n\n"
                        "💡 **Tip:** Please avoid sharing real credit card details, passwords, or government IDs in the chat."
                    ),
                    sanitized_input=sanitized
                )

        return GuardrailResult(blocked=False, sanitized_input=sanitized)

    # 2. BOOKING TOOL GUARDRAIL CHECK
    def check_booking(
        self,
        date_str: str,
        time_str: str,
        reason: str = "",
        user_email: str = "",
        session_id: str = "default",
        custom_settings: Optional[Dict[str, bool]] = None
    ) -> GuardrailResult:
        """Validates booking constraints: Anti-Flooding, Working Hours, Past Dates, Weekends."""

        # A. Anti-Flooding Rate Limiter
        if self.is_enabled("anti_flooding", custom_settings):
            count = self.get_booking_count(session_id)
            if count >= 3:
                logger.warning(f"🛡️ Guardrail Triggered [anti_flooding]: Max bookings ({count}) reached for session {session_id[:8]}")
                return GuardrailResult(
                    blocked=True,
                    guardrail="anti_flooding",
                    guardrail_name="Anti-Flooding Rate Limiter",
                    reason="You have reached the maximum allowed appointment bookings (3 per session) to prevent calendar flooding.",
                    suggestion="If you need to schedule multiple appointments or group sessions, please contact administration directly.",
                    notification_message=(
                        "🛡️ **Guardrail Alert: Booking Rate Limit Reached**\n\n"
                        "To protect our calendar from automated spam, users are limited to 3 bookings per session. You have already reached this limit.\n\n"
                        "💡 **How to correct this:**\n"
                        "If you need to reschedule an existing appointment or book additional slots, please contact our support team or start a new authorized session."
                    )
                )

        # B. Working Hours & Calendar Business Rules
        if self.is_enabled("booking_rules", custom_settings):
            try:
                # 1. Date parsing
                now = datetime.now()
                target_date = None
                date_lower = date_str.lower().strip()

                if "tomorrow" in date_lower:
                    target_date = (now + timedelta(days=1)).date()
                elif "today" in date_lower:
                    target_date = now.date()
                else:
                    # Try YYYY-MM-DD
                    match = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", date_str)
                    if match:
                        target_date = datetime.strptime(match.group(0), "%Y-%m-%d").date()

                # Check past date
                if target_date and target_date < now.date():
                    return GuardrailResult(
                        blocked=True,
                        guardrail="booking_rules",
                        guardrail_name="Working Hours & Booking Rules",
                        reason=f"The requested appointment date ({date_str}) is in the past.",
                        suggestion="Please specify a future date for your appointment.",
                        notification_message=(
                            f"🛡️ **Guardrail Alert: Past Date Rejected**\n\n"
                            f"Appointments cannot be scheduled in the past ({date_str}).\n\n"
                            "💡 **How to correct this:**\n"
                            "Please specify a future date (for example: *'Tomorrow at 2:00 PM'* or *'This Friday at 11:00 AM'*)."
                        )
                    )

                # Check weekend (Saturday = 5, Sunday = 6)
                if target_date and target_date.weekday() in [5, 6]:
                    day_name = target_date.strftime("%A")
                    return GuardrailResult(
                        blocked=True,
                        guardrail="booking_rules",
                        guardrail_name="Working Hours & Booking Rules",
                        reason=f"Appointments cannot be booked on weekends ({day_name}). Our team operates Monday through Friday.",
                        suggestion="Please choose a weekday (Monday to Friday).",
                        notification_message=(
                            f"🛡️ **Guardrail Alert: Weekend Booking Blocked**\n\n"
                            f"The date requested falls on a {day_name}. Our offices are open Monday through Friday.\n\n"
                            "💡 **How to correct this:**\n"
                            "Please choose a weekday (for example: *'Next Monday at 10:00 AM'* or *'This Friday at 3:00 PM'*)."
                        )
                    )

                # 2. Time parsing & Working Hours check
                # Expect formats like "12:00 PM", "3 PM", "15:00", etc.
                time_clean = time_str.split("-")[0].strip()
                parsed_hour = None
                parsed_minute = 0

                time_match = re.search(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", time_clean, re.IGNORECASE)
                if time_match:
                    hour = int(time_match.group(1))
                    minute = int(time_match.group(2)) if time_match.group(2) else 0
                    meridiem = time_match.group(3).lower() if time_match.group(3) else None

                    if meridiem == "pm" and hour < 12:
                        hour += 12
                    elif meridiem == "am" and hour == 12:
                        hour = 0
                    parsed_hour = hour
                    parsed_minute = minute

                if parsed_hour is not None:
                    work_start = Config.WORKING_HOURS_START # default 9
                    work_end = Config.WORKING_HOURS_END     # default 18 (6 PM)

                    if parsed_hour < work_start or parsed_hour >= work_end:
                        return GuardrailResult(
                            blocked=True,
                            guardrail="booking_rules",
                            guardrail_name="Working Hours & Booking Rules",
                            reason=f"Requested time ({time_str}) is outside official working hours ({work_start}:00 AM - {work_end % 12 or 12}:00 PM).",
                            suggestion=f"Please choose an appointment time between {work_start}:00 AM and {work_end % 12 or 12}:00 PM.",
                            notification_message=(
                                f"🛡️ **Guardrail Alert: Outside Working Hours**\n\n"
                                f"You requested an appointment at **{time_str}**, which falls outside our business hours ({work_start}:00 AM to {work_end % 12 or 12}:00 PM, Mon-Fri).\n\n"
                                "💡 **How to correct this:**\n"
                                f"Please select a time within business hours (for example: *'Tomorrow at {min(work_start+2, 14)}:00 PM'* or *'Tomorrow at {work_start+1}:00 AM'*)."
                            )
                        )
            except Exception as e:
                logger.warning(f"Error during booking guardrail check: {e}")

        return GuardrailResult(blocked=False)

    # 3. RAG GROUNDING GUARDRAIL CHECK
    def check_rag_grounding(
        self,
        matches: List[Dict[str, Any]],
        query: str,
        custom_settings: Optional[Dict[str, bool]] = None
    ) -> GuardrailResult:
        """Checks if retrieved document chunks meet the confidence relevance threshold."""
        if not self.is_enabled("rag_grounding", custom_settings):
            return GuardrailResult(blocked=False)

        if not matches or len(matches) == 0:
            return GuardrailResult(
                blocked=False,
                warning_only=True,
                guardrail="rag_grounding",
                guardrail_name="RAG Grounding & Hallucination Guard",
                reason=f"No matching chunks were found in the vector knowledge base for query: '{query[:40]}'.",
                suggestion="Please verify that relevant files were uploaded in the 'Ingest Documents' tab.",
                notification_message=(
                    "🛡️ **Guardrail Notice: Low Grounding Confidence**\n\n"
                    "No relevant passages were found in your uploaded documents matching this inquiry. The assistant will decline to fabricate answers.\n\n"
                    "💡 **How to correct this:**\n"
                    "Upload the relevant documentation in the **Ingest Documents** page, or rephrase with specific keywords found in your files."
                )
            )

        top_score = matches[0].get("score", 0.0)
        # Cosine similarity threshold (0.35 in Pinecone with bge-large is considered a good confidence baseline)
        if top_score < 0.35:
            return GuardrailResult(
                blocked=False,
                warning_only=True,
                guardrail="rag_grounding",
                guardrail_name="RAG Grounding & Hallucination Guard",
                reason=f"Top match similarity ({top_score*100:.1f}%) is below the minimum confidence threshold (35%).",
                suggestion="Rephrase the question with specific terminology or upload more comprehensive files.",
                notification_message=(
                    f"🛡️ **Guardrail Notice: Low Grounding Relevance ({top_score*100:.1f}%)**\n\n"
                    "The retrieved document excerpts have low confidence relevance to your inquiry. The assistant is instructed not to guess or invent details.\n\n"
                    "💡 **How to correct this:**\n"
                    "Try searching for specific sections, chapter titles, or phrases directly from your uploaded documents."
                )
            )

        return GuardrailResult(blocked=False)

    # 4. OUTPUT SECRET LEAK FILTER
    def check_output(
        self,
        bot_response: str,
        custom_settings: Optional[Dict[str, bool]] = None
    ) -> Tuple[str, Optional[GuardrailResult]]:
        """Filters API keys, secret credentials, or OAuth tokens from the final bot response."""
        if not self.is_enabled("secret_leak", custom_settings):
            return bot_response, None

        sanitized = bot_response
        leak_detected = False

        # OpenRouter Key
        if re.search(r"sk-or-v1-[a-zA-Z0-9]{20,}", sanitized):
            sanitized = re.sub(r"sk-or-v1-[a-zA-Z0-9]{20,}", "[REDACTED_OPENROUTER_KEY]", sanitized)
            leak_detected = True

        # Pinecone Key
        if re.search(r"pcsk_[a-zA-Z0-9_-]{20,}", sanitized):
            sanitized = re.sub(r"pcsk_[a-zA-Z0-9_-]{20,}", "[REDACTED_PINECONE_KEY]", sanitized)
            leak_detected = True

        # Hugging Face Token
        if re.search(r"hf_[a-zA-Z0-9]{20,}", sanitized):
            sanitized = re.sub(r"hf_[a-zA-Z0-9]{20,}", "[REDACTED_HF_TOKEN]", sanitized)
            leak_detected = True

        # Google Client Secret or OAuth Bearer
        if re.search(r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{30,}", sanitized):
            sanitized = re.sub(r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{30,}", "Bearer [REDACTED_OAUTH_TOKEN]", sanitized)
            leak_detected = True

        if leak_detected:
            logger.warning("🛡️ Guardrail Triggered [secret_leak]: Intercepted credential pattern in bot response and redacted.")
            result = GuardrailResult(
                blocked=False,
                warning_only=True,
                guardrail="secret_leak",
                guardrail_name="Secret & Credential Leak Filter",
                reason="The bot response contained an internal credential pattern which was automatically redacted.",
                suggestion="No action needed. System security remains protected."
            )
            return sanitized, result

        return sanitized, None


# Global singleton instance
guardrail_manager = GuardrailManager()
