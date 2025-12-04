import streamlit as st
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json
import sseclient
import uuid
import time

# ================================================================================
# CONFIG
# ================================================================================

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="B2B AI Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    * { color: white; }
    body { background-color: #000000; }
    .main { background-color: #000000; }

    .chat-user {
        background-color: #222222;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        text-align: right;
        border-left: 3px solid #666666;
    }
    .chat-bot {
        background-color: #111111;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-right: 3px solid #00AAFF;
    }
    .streaming-dot {
        height: 10px;
        width: 10px;
        background-color: #f0f0f0;
        border-radius: 50%;
        display: inline-block;
        margin-left: 5px;
        animation: blink 1s infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
</style>
""", unsafe_allow_html=True)


# ================================================================================
# SESSION STATE INITIALIZATION
# ================================================================================

if 'show_login' not in st.session_state:
    st.session_state.show_login = True
if 'user_name' not in st.session_state:
    st.session_state.user_name = None
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'enable_streaming' not in st.session_state:
    st.session_state.enable_streaming = True
if 'user_stats' not in st.session_state:
    st.session_state.user_stats = {}
if 'show_calendar_picker' not in st.session_state:
    st.session_state.show_calendar_picker = False
if 'calendar_message_idx' not in st.session_state:
    st.session_state.calendar_message_idx = -1
if 'current_session_id' not in st.session_state:
    st.session_state.current_session_id = None
if 'chat_history' not in st.session_state or not isinstance(st.session_state.chat_history, list):
    st.session_state.chat_history = []
if 'external_history' not in st.session_state:
    st.session_state.external_history = []
if 'context_summary' not in st.session_state:
    st.session_state.context_summary = "The user has just started a new chat session."

# --- NEW: Thread ID for LangGraph Memory ---
if 'thread_id' not in st.session_state:
    # Initialize a thread ID. This persists as long as the page is open/session is active.
    # If the user logs out, we clear this.
    st.session_state.thread_id = str(uuid.uuid4())

# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def display_message(role: str, content: str, agent: str = ""):
    if role == "user":
        st.markdown(
            f"<div class='chat-user'>👤 <strong>You:</strong> {content}</div>",
            unsafe_allow_html=True
        )
    else:
        agent_label = f"[{agent.upper()}] " if agent else ""
        st.markdown(
            f"<div class='chat-bot'>🤖 <strong>Assistant ({agent_label}):</strong> {content}</div>",
            unsafe_allow_html=True
        )

def cancel_stream():
    session_id = st.session_state.current_session_id
    if session_id:
        try:
            requests.post(f"{API_URL}/chat/cancel", json={"session_id": session_id}, timeout=2)
            st.warning("Stream cancellation signal sent.")
        except Exception:
            st.warning("Could not reach API to cancel stream.")
    st.session_state.current_session_id = None
    st.rerun()

# ================================================================================
# API CALLS
# ================================================================================

def login_user(user_name: str) -> Optional[Dict]:
    try:
        response = requests.post(f"{API_URL}/user/login", json={"user_name": user_name})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Login failed: {e}")
        return None

def get_user_stats(user_name: str) -> Dict:
    try:
        response = requests.get(f"{API_URL}/user/{user_name}/stats")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return {}

def get_user_history(user_name: str, limit: int = 20) -> List[Dict]:
    try:
        response = requests.get(f"{API_URL}/user/{user_name}/history", params={"limit": limit})
        response.raise_for_status()
        return response.json().get('history', [])
    except requests.exceptions.RequestException:
        return []

def send_message(user_input: str, user_name: str) -> Optional[Dict]:
    try:
        payload = {
            "user_message": user_input,
            "user_name": user_name,
            "user_id": st.session_state.user_id,
            "stream_enabled": False,
            "context_summary": st.session_state.context_summary,
            "thread_id": st.session_state.thread_id # Pass thread_id
        }
        response = requests.post(f"{API_URL}/chat", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error sending message: {e}")
        return None

def send_message_streaming(user_input: str, user_name: str, placeholder) -> Optional[Dict]:
    full_bot_response = ""
    final_intent = "general"
    
    # current_session_id is for this specific turn's streaming cancellation
    current_session_id = str(uuid.uuid4())
    st.session_state.current_session_id = current_session_id
    
    try:
        payload = {
            "user_message": user_input,
            "user_name": user_name,
            "user_id": st.session_state.user_id,
            "stream_enabled": True,
            "context_summary": st.session_state.context_summary,
            "thread_id": st.session_state.thread_id # Pass persistent thread_id
        }
        
        with requests.Session() as s:
            response = s.post(f"{API_URL}/chat", json=payload, stream=True, timeout=120)
            response.raise_for_status()
            
            client = sseclient.SSEClient(response)
            
            for event in client.events():
                if event.data == "[DONE]":
                    break
                
                try:
                    data = json.loads(event.data)
                    
                    if data.get("type") == "token":
                        full_bot_response += data.get("content", "")
                        placeholder.markdown(
                            f"<div class='chat-bot'>🤖 <strong>Assistant:</strong> {full_bot_response} <span class='streaming-dot'></span></div>", 
                            unsafe_allow_html=True
                        )
                    elif data.get("type") == "intent":
                        final_intent = data.get("content", final_intent)
                    elif data.get("type") in ["done", "cancelled", "error"]:
                        if data.get("type") == "error":
                            st.error(f"Backend Stream Error: {data.get('content')}")
                        elif data.get("type") == "cancelled":
                            full_bot_response += "\n\n(Stream cancelled by user.)"
                        break
                        
                except json.JSONDecodeError:
                    if event.data:
                        full_bot_response += event.data
                        placeholder.markdown(
                            f"<div class='chat-bot'>🤖 <strong>Assistant:</strong> {full_bot_response} <span class='streaming-dot'></span></div>", 
                            unsafe_allow_html=True
                        )
                    continue

    except requests.exceptions.RequestException as e:
        st.error(f"Error sending streaming message: {e}")
        return None
    
    placeholder.markdown(
        f"<div class='chat-bot'>🤖 <strong>Assistant ({final_intent.upper()}):</strong> {full_bot_response}</div>", 
        unsafe_allow_html=True
    )
    
    st.session_state.current_session_id = None
    return {"bot_response": full_bot_response, "intent": final_intent}


# ================================================================================
# UI SCREENS
# ================================================================================

def show_login_screen():
    st.title("Login to B2B AI Chat")
    
    with st.form("login_form"):
        user_name_input = st.text_input("Enter your Name", key="login_name")
        submitted = st.form_submit_button("Start Chatting")

        if submitted and user_name_input:
            response = login_user(user_name_input)
            if response and response.get("status") == "success":
                st.session_state.user_name = response["user_name"]
                st.session_state.user_id = response["user_id"]
                st.session_state.show_login = False
                st.session_state.chat_history = [] 
                st.session_state.external_history = []
                st.session_state.user_stats = {}
                # Create a new thread_id for this session
                st.session_state.thread_id = str(uuid.uuid4())
                st.rerun()

def show_chat_screen():
    # --- Sidebar ---
    if st.session_state.user_name:
        if not st.session_state.user_stats:
            st.session_state.user_stats = get_user_stats(st.session_state.user_name)

    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.title(f"💬 Chat - {st.session_state.user_name}")
    with col2:
        try:
            r = requests.get(f"{API_URL}/health", timeout=2)
            if r.status_code == 200: st.success("🟢 Online")
            else: st.error("🟡 Unresponsive")
        except: st.error("🔴 Offline")
    with col3:
        st.session_state.enable_streaming = st.checkbox(
            "🌊 Streaming", value=st.session_state.enable_streaming, key="stream_toggle"
        )
    with col4:
        if st.button("🚪 Logout"):
            if st.session_state.current_session_id: cancel_stream()
            st.session_state.user_name = None
            st.session_state.user_id = None
            st.session_state.chat_history = []
            st.session_state.show_login = True
            st.session_state.thread_id = None # Clear thread
            st.rerun()
    
    st.divider()
    
    # Sidebar
    with st.sidebar:
        st.subheader(f"👤 {st.session_state.user_name}")
        st.caption(f"Thread: {st.session_state.thread_id[:8]}...")
        if st.button("🆕 New Conversation"):
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.success("Started new memory thread!")
            time.sleep(1)
            st.rerun()
            
        st.divider()
        st.markdown("### 📊 Stats")
        # (Stats display logic same as before...)

    # Chat Area
    st.subheader("💬 Messages")
    for idx, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            display_message("user", msg["content"])
        else:
            display_message("bot", msg["content"], msg.get("agent", ""))
            if msg.get("is_booking", False):
                if st.button("📅 Open Calendar", key=f"cal_{idx}"):
                    st.session_state.show_calendar_picker = True
                    st.session_state.calendar_message_idx = idx
                    st.rerun()
    
    # Calendar Picker (same as before)
    if st.session_state.show_calendar_picker:
        st.divider()
        st.subheader("📅 Select Meeting Time")
        col1, col2 = st.columns(2)
        with col1:
            selected_date = st.date_input("Date", value=datetime.now().date())
        with col2:
            st.write("Time Selection")
            if st.button("✅ Confirm"):
                st.success(f"✅ Meeting booked for {selected_date}!")
                st.session_state.show_calendar_picker = False
                st.rerun()
        if st.button("❌ Cancel"):
            st.session_state.show_calendar_picker = False
            st.rerun()
    
    st.divider()
    
    # Input
    st.subheader("✍️ Your Message")
    col_input, col_send, col_cancel = st.columns([5, 1, 1])
    with col_input:
        user_input = st.text_input("Message", label_visibility="collapsed", placeholder="Type your message...", key="user_input")
    with col_send:
        send_btn = st.button("📤 Send", use_container_width=True)
    with col_cancel:
        if st.session_state.current_session_id:
            if st.button("🛑 Stop", use_container_width=True):
                cancel_stream()
    
    if send_btn and user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input, "agent": ""})
        placeholder = st.empty()
        
        if st.session_state.enable_streaming:
            placeholder.markdown(f"<div class='chat-bot'>🤖 <strong>Assistant:</strong><br><span class='streaming-dot'></span></div>", unsafe_allow_html=True)
            response = send_message_streaming(user_input, st.session_state.user_name, placeholder)
        else:
            with st.spinner("Processing..."):
                response = send_message(user_input, st.session_state.user_name)
            placeholder.empty()
        
        if response:
            bot_msg = response.get("bot_response", "")
            intent = response.get("intent", "general")
            is_booking = intent == "booking" or "BOOKING_REQUEST" in bot_msg
            st.session_state.chat_history.append({
                "role": "bot", "content": bot_msg, "agent": intent, "is_booking": is_booking
            })
            st.session_state.external_history = get_user_history(st.session_state.user_name, limit=50)
            st.session_state.user_stats = get_user_stats(st.session_state.user_name)
            st.rerun()


if st.session_state.show_login:
    show_login_screen()
else:
    show_chat_screen()