from langchain_core.tools import tool, StructuredTool
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import os
import asyncio
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from googleapiclient.discovery import build
import uuid
import logging
import re
import pytz
from app.config import Config

logger = logging.getLogger(__name__)

class GoogleCalendarManager:
    """Google Calendar API Manager - Async Wrapper"""
    
    def __init__(self):
        """Initialize Google Calendar manager"""
        self.credentials_file = Config.GOOGLE_CREDENTIALS
        self.token_file = Config.GOOGLE_TOKEN
        self.scopes = ['https://www.googleapis.com/auth/calendar']
        self.service = None
        
        # Calendar configuration from Config
        self.calendar_id = "primary"
        self.meeting_duration = Config.MEETING_DURATION_MINUTES
        self.buffer_time = Config.BUFFER_TIME_MINUTES
        self.working_hours = Config.get_working_hours()
        self.timezone = pytz.timezone(Config.DEFAULT_TIMEZONE)
        
        # Authenticate synchronously on init (or could be lazy loaded)
        self._authenticate()
    
    def _authenticate(self):
        """Authenticate with Google Calendar with Robust Error Handling"""
        creds = None
        
        try:
            # 1. Try to load existing token
            if os.path.exists(self.token_file):
                logger.info("📝 Loading Google Calendar token...")
                try:
                    creds = Credentials.from_authorized_user_file(self.token_file, self.scopes)
                except Exception as e:
                    logger.warning(f"⚠️ Corrupt token file: {e}")
                    creds = None
            
            # 2. Check validity / Refresh
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        logger.info("🔄 Refreshing token...")
                        creds.refresh(Request())
                    except (RefreshError, Exception) as e:
                        logger.warning(f"⚠️ Token refresh failed: {e}")
                        logger.info("🗑️ Deleting invalid token.json to force re-login...")
                        if os.path.exists(self.token_file):
                            os.remove(self.token_file)
                        creds = None
                
                # 3. New Login (if no creds or refresh failed)
                if not creds:
                    if not os.path.exists(self.credentials_file):
                        logger.warning(f"Google credentials file ({self.credentials_file}) not found. Calendar booking unavailable.")
                        return
                    # Avoid blocking backend startup with interactive prompt if token.json is missing
                    logger.warning(f"Google Calendar token ({self.token_file}) not found. Please authenticate Google Calendar separately.")
                    return
            
            self.service = build('calendar', 'v3', credentials=creds)
            logger.info("Google Calendar authenticated successfully")

        except Exception as e:
            logger.error(f"❌ Calendar auth failed completely: {e}")
            self.service = None

    def _ensure_authenticated(self) -> bool:
        """Dynamically re-checks token.json if not yet authenticated."""
        if not self.service:
            self._authenticate()
        return bool(self.service)

    async def is_slot_available(self, date_str: str, time_slot: str) -> bool:
        """Check if a specific time slot is available (Non-blocking)"""
        if not self._ensure_authenticated(): return False
        
        return await asyncio.to_thread(self._is_slot_available_sync, date_str, time_slot)

    def _is_slot_available_sync(self, date_str: str, time_slot: str) -> bool:
        """Synchronous implementation of availability check"""
        try:
            # Parse datetime
            if date_str.lower() == "tomorrow":
                target_date = datetime.now() + timedelta(days=1)
                date_str = target_date.strftime("%Y-%m-%d")
            
            # Parse time
            time_obj = datetime.strptime(time_slot.split(" - ")[0].strip(), "%I:%M %p").time()
            meeting_start = datetime.combine(
                datetime.strptime(date_str, "%Y-%m-%d").date(),
                time_obj
            )
            meeting_end = meeting_start + timedelta(minutes=self.meeting_duration)
            
            # Set timezone
            meeting_start = self.timezone.localize(meeting_start)
            meeting_end = self.timezone.localize(meeting_end)
            
            logger.info(f"🔍 Checking availability: {meeting_start} to {meeting_end}")
            
            # Query calendar
            events = self.service.events().list(
                calendarId=self.calendar_id,
                timeMin=meeting_start.isoformat(),
                timeMax=meeting_end.isoformat(),
                singleEvents=True
            ).execute().get('items', [])
            
            is_available = len(events) == 0
            logger.info(f"{'✅' if is_available else '❌'} Slot is {'available' if is_available else 'booked'}")
            
            return is_available
        
        except Exception as e:
            logger.error(f"❌ Error checking availability: {e}")
            return False
    
    async def get_available_slots(self, date_str: str = None, num_slots: int = 5) -> List[str]:
        """Get available slots from Google Calendar (Non-blocking)"""
        if not self.service: return []
        
        return await asyncio.to_thread(self._get_available_slots_sync, date_str, num_slots)

    def _get_available_slots_sync(self, date_str: str = None, num_slots: int = 5) -> List[str]:
        """Synchronous implementation of getting slots"""
        try:
            if not date_str or date_str.lower() == "tomorrow":
                date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            
            logger.info(f"📅 Fetching slots for {date_str}")
            
            target_date = datetime.strptime(date_str, "%Y-%m-%d")
            start_time = target_date.replace(hour=self.working_hours["start"], minute=0, second=0)
            end_time = target_date.replace(hour=self.working_hours["end"], minute=0, second=0)
            
            start_time = self.timezone.localize(start_time)
            end_time = self.timezone.localize(end_time)
            
            # Get events
            events = self.service.events().list(
                calendarId=self.calendar_id,
                timeMin=start_time.isoformat(),
                timeMax=end_time.isoformat(),
                singleEvents=True,
                orderBy='startTime'
            ).execute().get('items', [])
            
            booked_times = []
            for event in events:
                if 'start' in event and 'end' in event:
                    start = datetime.fromisoformat(event['start']['dateTime'].replace('Z', '+00:00'))
                    end = datetime.fromisoformat(event['end']['dateTime'].replace('Z', '+00:00'))
                    booked_times.append((start, end))
            
            available_slots = []
            current_time = start_time
            
            while len(available_slots) < num_slots and current_time < end_time:
                slot_end = current_time + timedelta(minutes=self.meeting_duration)
                
                is_available = True
                for booked_start, booked_end in booked_times:
                    if current_time < booked_end and slot_end > booked_start:
                        is_available = False
                        current_time = booked_end + timedelta(minutes=self.buffer_time)
                        break
                
                if is_available and slot_end <= end_time:
                    slot_str = current_time.strftime("%I:%M %p") + " - " + slot_end.strftime("%I:%M %p")
                    available_slots.append(slot_str)
                
                current_time += timedelta(minutes=self.meeting_duration + self.buffer_time)
            
            return available_slots if available_slots else ["9:00 AM - 10:00 AM", "10:00 AM - 11:00 AM"]
        
        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return []
    
    async def book_meeting(self, user_email: str, slot: str, meeting_title: str = "B2B Consultation", date_str: str = None) -> Dict:
        """Book a meeting (Non-blocking)"""
        if not self.service: raise Exception("Calendar service not authenticated")

        return await asyncio.to_thread(self._book_meeting_sync, user_email, slot, meeting_title, date_str)

    def _book_meeting_sync(self, user_email: str, slot: str, meeting_title: str, date_str: str) -> Dict:
        """Synchronous implementation of booking"""
        try:
            if not date_str or date_str.lower() == "tomorrow":
                date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            
            logger.info(f"📅 Booking: {meeting_title} at {slot} on {date_str}")
            
            # Parse time
            time_parts = slot.split(" - ")
            start_time_str = time_parts[0].strip()
            
            meeting_start = datetime.strptime(f"{date_str} {start_time_str}", "%Y-%m-%d %I:%M %p")
            meeting_end = meeting_start + timedelta(minutes=self.meeting_duration)
            
            meeting_start = self.timezone.localize(meeting_start)
            meeting_end = self.timezone.localize(meeting_end)
            
            # Create event
            event = {
                'summary': meeting_title,
                'start': {'dateTime': meeting_start.isoformat(), 'timeZone': 'UTC'},
                'end': {'dateTime': meeting_end.isoformat(), 'timeZone': 'UTC'},
                'conferenceData': {
                    'createRequest': {
                        'requestId': str(uuid.uuid4())
                    }
                },
                'attendees': [{'email': user_email}],
            }
            
            event = self.service.events().insert(
                calendarId=self.calendar_id,
                body=event,
                conferenceDataVersion=1,
                sendUpdates='all'
            ).execute()
            
            meet_link = "https://meet.google.com"
            if 'conferenceData' in event and 'entryPoints' in event['conferenceData']:
                for entry in event['conferenceData']['entryPoints']:
                    if entry['entryPointType'] == 'video':
                        meet_link = entry['uri']
            
            logger.info(f"✅ Meeting booked: {event['id']}")
            
            return {
                "booking_id": event['id'],
                "user_email": user_email,
                "title": meeting_title,
                "slot": slot,
                "date": date_str,
                "meet_link": meet_link,
                "status": "confirmed"
            }
        
        except Exception as e:
            logger.error(f"❌ Booking failed: {e}")
            raise


    async def cancel_meeting(self, user_email: str, reason: str) -> Optional[str]:
        """Cancel a meeting based on user email and reason (Non-blocking)"""
        if not self.service: return None
        
        return await asyncio.to_thread(self._cancel_meeting_sync, user_email, reason)

    def _cancel_meeting_sync(self, user_email: str, reason: str) -> Optional[str]:
        """Synchronous implementation of meeting cancellation"""
        try:
            logger.info(f"🗑️ Attempting to cancel meeting for {user_email} with reason: {reason}")
            
            # Search for future events
            now = datetime.utcnow().isoformat() + 'Z'  # 'Z' indicates UTC time
            
            events_result = self.service.events().list(
                calendarId=self.calendar_id,
                timeMin=now,
                maxResults=20,
                singleEvents=True,
                orderBy='startTime'
            ).execute()
            events = events_result.get('items', [])

            for event in events:
                # Check attendees
                attendees = event.get('attendees', [])
                attendee_emails = [a.get('email') for a in attendees]
                
                # Check if user is an attendee and reason matches summary
                # We use a loose match for reason in summary
                if user_email in attendee_emails and reason.lower() in event.get('summary', '').lower():
                    logger.info(f"✅ Found meeting to cancel: {event.get('summary')} at {event.get('start')}")
                    
                    self.service.events().delete(
                        calendarId=self.calendar_id,
                        eventId=event['id'],
                        sendUpdates='all'
                    ).execute()
                    
                    return f"{event.get('summary')} on {event.get('start').get('dateTime') or event.get('start').get('date')}"

            logger.info("⚠️ No matching meeting found to cancel.")
            return None

        except Exception as e:
            logger.error(f"❌ Error cancelling meeting: {e}")
            return None


# Global instance
try:
    calendar_manager = GoogleCalendarManager()
except Exception as e:
    logger.error(f"⚠️ Failed to initialize Calendar Manager: {e}")
    calendar_manager = None


async def _booking_agent_tool_async(date: str, time: str, reason: str = "General Consultation", user_email: str = "abhi.dhaka16@gmail.com", reschedule: bool = False) -> str:
    """Async implementation of booking agent tool."""
    logger.info(f"📥 Booking Request: date={date}, time={time}, reason={reason}, reschedule={reschedule}")

    # Normalize date if needed
    date_str = date
    if date_str and isinstance(date_str, str):
        try:
            # Convert "29 November 2025" → "2025-11-29"
            parsed = datetime.strptime(date_str, "%d %B %Y")
            date_str = parsed.strftime("%Y-%m-%d")
        except:
            pass
    
    time_str = time

    logger.info(f"📅 Parsed date={date_str}, time={time_str}, reason={reason}")

    if not date_str or not time_str:
        return "I need both date and time to book your appointment."
    
    from app.guardrails import guardrail_manager
    
    # 0. Evaluate Booking Guardrails (Working Hours, Past Dates, Weekend, Anti-Flooding)
    guard_res = guardrail_manager.check_booking(date_str, time_str, reason, user_email)
    if guard_res.blocked:
        logger.warning(f"🛡️ Booking blocked by guardrail: {guard_res.guardrail}")
        return guard_res.notification_message

    if not calendar_manager or not calendar_manager._ensure_authenticated():
        return (
            f"📅 Booking Request received for **{reason}** on **{date_str} at {time_str}**.\n\n"
            f"Google Calendar credentials are not connected (offline demo mode). "
            f"Please click **'Open Calendar'** below to view available slots and manage your appointment!"
        )

    try:
        # Handle Rescheduling (Cancel old meeting first)
        cancel_msg = ""
        if reschedule:
            cancelled_meeting = await calendar_manager.cancel_meeting(user_email, reason)
            if cancelled_meeting:
                cancel_msg = f"🗑️ **Previous meeting cancelled:** {cancelled_meeting}\n\n"
            else:
                cancel_msg = "⚠️ Could not find a previous meeting to cancel, but proceeding with new booking.\n\n"

        # Check availability (ASYNC)
        is_available = await calendar_manager.is_slot_available(date_str, time_str)

        if not is_available:
            # Get slots (ASYNC)
            available_slots = await calendar_manager.get_available_slots(date_str)
            return (
                f"{cancel_msg}"
                f"⛔ The slot **{time_str} on {date_str}** is not available.\n"
                f"Available slots:\n" +
                "\n".join(f"- {slot}" for slot in available_slots)
            )

        # Book appointment (ASYNC)
        slot_end = (
            datetime.strptime(time_str, "%I:%M %p") + timedelta(minutes=Config.MEETING_DURATION_MINUTES)
        ).strftime("%I:%M %p")

        booking = await calendar_manager.book_meeting(
            user_email=user_email,
            date_str=date_str,
            slot=f"{time_str} - {slot_end}",
            meeting_title=f"Meeting: {reason}"
        )

        guardrail_manager.increment_booking_count("default")

        return (
            f"{cancel_msg}"
            f"✅ Appointment booked!\n\n"
            f"📌 **Topic:** {reason}\n"
            f"📅 **Date:** {booking['date']}\n"
            f"⏰ **Time:** {booking['slot']}\n"
            f"🔗 **Meet Link:** {booking['meet_link']}"
        )

    except Exception as e:
        logger.error(f"❌ Error in booking_agent_tool: {e}", exc_info=True)
        return "Something went wrong while booking. Try again."

def _booking_agent_tool_sync(date: str, time: str, reason: str = "General Consultation", user_email: str = "abhi.dhaka16@gmail.com", reschedule: bool = False) -> str:
    """Sync implementation of booking agent tool supporting both worker threads and event loops."""
    import concurrent.futures
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, _booking_agent_tool_async(date, time, reason, user_email, reschedule))
                return future.result()
        else:
            return loop.run_until_complete(_booking_agent_tool_async(date, time, reason, user_email, reschedule))
    except RuntimeError:
        return asyncio.run(_booking_agent_tool_async(date, time, reason, user_email, reschedule))

booking_agent_tool = StructuredTool.from_function(
    func=_booking_agent_tool_sync,
    coroutine=_booking_agent_tool_async,
    name="booking_agent_tool",
    description=(
        "Booking agent tool to book, reschedule, and check calendar appointment slots. "
        "Provide date (e.g. 'tomorrow' or '2025-11-27'), time (e.g. '12:00 PM'), and topic/reason."
    )
)
