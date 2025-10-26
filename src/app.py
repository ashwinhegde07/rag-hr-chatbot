import streamlit as st
import requests
from datetime import datetime
import json
import os

# ---------- CONFIG ----------
st.set_page_config(page_title="RAG HR Chatbot", page_icon="🤖", layout="wide")

# ---------- STYLE ----------
st.markdown("""
    <style>
    .stChatMessage {border-radius: 10px; padding: 10px; margin: 5px 0;}
    .stChatInputContainer textarea {border-radius: 10px;}
    </style>
""", unsafe_allow_html=True)

# ---------- FILE PERSISTENCE ----------
HISTORY_FILE = "data/chat_history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(history):
    os.makedirs("data", exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

# ---------- SESSION STATE ----------
if "history" not in st.session_state:
    st.session_state.history = load_history()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("💬 Chat History")

    user_msgs = [m for m in st.session_state.history if m["role"] == "user"]

    if user_msgs:
        if st.button("🗑️ Clear chat history"):
            st.session_state.history.clear()
            save_history(st.session_state.history)
            st.experimental_rerun()

        st.markdown("---")

        for i, msg in enumerate(reversed(user_msgs)):
            preview = msg["content"][:60] + ("..." if len(msg["content"]) > 60 else "")
            st.write(f"**{len(user_msgs)-i}.** {preview}")
            st.caption(msg["time"])
    else:
        st.info("No previous chats yet.")

# ---------- MAIN ----------
st.title("🤖 HR Policy Chatbot")
st.caption("Ask about HR policies, benefits, or leave rules.")

# Display chat history
for chat in st.session_state.history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Auto-scroll to newest
st.markdown("""
<script>
var chatContainer = window.parent.document.querySelector('.stChatMessageContainer');
if (chatContainer) chatContainer.scrollTop = chatContainer.scrollHeight;
</script>
""", unsafe_allow_html=True)

# ---------- CHAT INPUT ----------
if prompt := st.chat_input("Ask your HR question here..."):
    # Show user message immediately
    st.session_state.history.append({
        "role": "user",
        "content": prompt,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_history(st.session_state.history)
    with st.chat_message("user"):
        st.markdown(prompt)

    # Show spinner and get response
    with st.chat_message("assistant"):
        with st.spinner("Fetching answer..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:5000/query",
                    json={"question": prompt},
                    timeout=30
                )
                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer found.")
                else:
                    answer = f"Error: backend returned {response.status_code}"
            except Exception as e:
                answer = f"Request failed: {e}"

            st.markdown(answer)

    # Append assistant response
    st.session_state.history.append({
        "role": "assistant",
        "content": answer,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_history(st.session_state.history)
