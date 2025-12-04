import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from app.tools.booking_tool import booking_agent_tool, GoogleCalendarManager

async def test_booking_agent_tool_async():
    # Mock the calendar manager to avoid real API calls
    with patch('app.tools.booking_tool.calendar_manager', new_callable=AsyncMock) as mock_manager:
        # Setup mock return values
        mock_manager.is_slot_available.return_value = True
        mock_manager.book_meeting.return_value = {
            "booking_id": "test_id",
            "user_email": "test@example.com",
            "title": "Meeting: Test",
            "slot": "10:00 AM - 11:00 AM",
            "date": "2025-11-27",
            "meet_link": "https://meet.google.com/test",
            "status": "confirmed"
        }

        # Run the tool
        result = await booking_agent_tool.ainvoke({
            "date": "2025-11-27",
            "time": "10:00 AM",
            "reason": "Test",
            "user_email": "test@example.com"
        })

        # Verify it was awaited and returned expected string
        print(f"DEBUG: Result: {result}")
        assert "Appointment booked" in result
        assert "2025-11-27" in result
        
        # Verify async calls were made
        mock_manager.is_slot_available.assert_awaited_once()
        mock_manager.book_meeting.assert_awaited_once()

async def test_booking_agent_tool_unavailable():
    with patch('app.tools.booking_tool.calendar_manager', new_callable=AsyncMock) as mock_manager:
        mock_manager.is_slot_available.return_value = False
        mock_manager.get_available_slots.return_value = ["11:00 AM - 12:00 PM"]

        result = await booking_agent_tool.ainvoke({
            "date": "2025-11-27",
            "time": "10:00 AM"
        })

        assert "not available" in result
        assert "11:00 AM - 12:00 PM" in result
        mock_manager.is_slot_available.assert_awaited_once()
        mock_manager.get_available_slots.assert_awaited_once()

if __name__ == "__main__":
    async def run_tests():
        print("Running test_booking_agent_tool_async...")
        await test_booking_agent_tool_async()
        print("✅ test_booking_agent_tool_async passed")
        
        print("Running test_booking_agent_tool_unavailable...")
        await test_booking_agent_tool_unavailable()
        print("✅ test_booking_agent_tool_unavailable passed")
        
    asyncio.run(run_tests())
