"""
Streamlit Frontend for E-commerce Company Chatbot
"""
import streamlit as st
import requests
import uuid
from datetime import datetime

# Configuration
API_URL = "http://127.0.0.1:8000"

# Page config
st.set_page_config(
    page_title = "OGE E-commerce QC Info Assistant",
    page_icon = "💬",
    layout = "wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .sidebar-info {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

def send_message(message: str):
    """Send message to the backend API"""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json = {
                "session_id": st.session_state.session_id,
                "message": message
            },
            timeout=30
        )

        if response.status_code == 200:
            data = response.json()
            return data
        else:
            st.error(f"Error {response.status_code} - {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error(" Cannot connect to backend. Make sure the API server is running on port 8000.")
        st.info("Run: `python chatbot_api.py`")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None
    
def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        data = response.json()
        return data.get("status") == "healthy"
    except Exception as e:
        st.error(f"Error connecting to backend: {e}")
        return False
    

# Header
st.markdown('<div class="main-header">💬 OGE E-commerce QC Solutions</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Your AI Assistant for Company Information</div>', unsafe_allow_html=True)

# Check API health
api_healthy = check_api_health()

if not api_healthy:
    st.error(" Backend API is not running. Please start the API server first.")
    st.code("python chatbot_api.py", language="bash")
    st.stop()
else:
    st.success(" Connected to chatbot backend")

# Sidebar
with st.sidebar:
    st.header(" About")
    
    st.markdown("""
    <div class="sidebar-info">
    <h4>What can I help you with?</h4>
    <p>Ask me anything about:</p>
    <ul>
        <li>Our products and services</li>
        <li>Pricing and plans</li>
        <li>Technical capabilities</li>
        <li>Customer success stories</li>
        <li>Contact information</li>
        <li>Integration options</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Session info
    st.subheader(" Session Info")
    st.text(f"Session: {st.session_state.session_id[:8]}...")
    st.metric("Messages", len(st.session_state.messages))
    
    if st.button(" New Conversation", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    st.divider()

    # Quick questions
    st.subheader(" Quick Questions")
    
    quick_questions = [
        "What services do you offer?",
        "What are your pricing plans?",
        "How accurate are your models?",
        "How can I contact you?",
        "Tell me about your technology",
        "Do you have API access?"
    ]
    
    for question in quick_questions:
        if st.button(question, key=f"quick_{quick_questions.index(question)}", use_container_width=True):
            st.session_state.user_input = question
            st.rerun()
    
    st.divider()

# Company quick facts
    st.markdown("""
    <div class="sidebar-info">
    <h4>Quick Facts</h4>
    <ul>
        <li> 100+ Customers</li>
        <li> 1000K+ Products Analyzed</li>
        <li> 80% Retention Rate</li>
        <li> 3 Offices</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Main chat area
st.subheader("💬 Chat")

# Display chat messages
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        
        with st.chat_message(role):
            st.write(content)


# Chat input
user_input = None
if "user_input" in st.session_state:
    user_input = st.session_state.user_input
    del st.session_state.user_input

if prompt := (user_input or st.chat_input("Ask me anything about our company...")):
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
    
    # Add to messages
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    })
    
    # Get response from backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_data = send_message(prompt)
            
            if response_data:
                response_text = response_data.get("response", "")
                st.write(response_text)
                
                # Add to messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "timestamp": datetime.now().isoformat()
                })
                
                st.rerun()

    # Footer
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**OGE E-commerce QC Solutions**")
    
with col2:
    st.markdown("info@ecommerce-ogeqc.com")
    
with col3:
    st.markdown("+2347084684214")

st.caption("Powered by LangChain & OpenAI | © 2024 OGE E-commerce QC Solutions")
