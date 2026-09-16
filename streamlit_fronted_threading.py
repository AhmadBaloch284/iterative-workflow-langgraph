import streamlit as st
from langgraph_tool_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid
from streamlit.components.v1 import html


def lucide_icon(icon_name):
    icons = {
        "search": """
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"
        viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"/>
        <path d="m21 21-4.3-4.3"/>
        </svg>
        """,

        "calculator": """
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"
        viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect width="16" height="20" x="4" y="2" rx="2"/>
        <line x1="8" x2="16" y1="6" y2="6"/>
        <line x1="16" x2="16" y1="14" y2="18"/>
        <line x1="8" x2="8" y1="14" y2="14"/>
        <line x1="8" x2="8" y1="18" y2="18"/>
        <line x1="12" x2="12" y1="14" y2="14"/>
        <line x1="12" x2="12" y1="18" y2="18"/>
        </svg>
        """,

        "bot": """
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"
        viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect width="18" height="10" x="3" y="11" rx="2"/>
        <circle cx="12" cy="5" r="2"/>
        <path d="M12 7v4"/>
        <line x1="8" x2="8" y1="15" y2="17"/>
        <line x1="16" x2="16" y1="15" y2="17"/>
        </svg>
        """,

        "check": """
        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18"
        viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M20 6 9 17l-5-5"/>
        </svg>
        """
    }

    return icons.get(icon_name, "")





# **************************** Utility Functions ****************************

def generate_thread_id():
    return str(uuid.uuid4())


def reset_chat():
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id

    add_thread(thread_id)

    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


def get_message_text(content):
    """
    Convert Gemini message content into clean text.
    """

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        text_parts = []

        for block in content:

            if isinstance(block, dict):

                if "text" in block:
                    text_parts.append(block["text"])

            elif isinstance(block, str):
                text_parts.append(block)

        return "".join(text_parts)

    return str(content)


# **************************** Session Setup ****************************

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []


add_thread(st.session_state["thread_id"])


# **************************** Sidebar UI ****************************

st.sidebar.title("LangGraph Chatbot")


if st.sidebar.button("New Chat"):
    reset_chat()


st.sidebar.header("My Conversations")


for thread_id in st.session_state["chat_threads"][::-1]:

    if st.sidebar.button(str(thread_id)):

        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:

            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"

            temp_messages.append({
                "role": role,
                "content": get_message_text(msg.content)
            })

        st.session_state["message_history"] = temp_messages


# **************************** Main UI ****************************

# Load conversation history

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# **************************** Chat Input ****************************

user_input = st.chat_input("Type here")


if user_input:

    # Add user message to history
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.write(user_input)


    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }


    # Assistant response
    # Assistant response
with st.chat_message("assistant"):

    with st.status("🤖 Thinking...", expanded=True) as status:

        ai_message = st.write_stream(
            get_message_text(message_chunk.content)
            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages"
            )
            if metadata.get("langgraph_node") == "chat_node"
            and get_message_text(message_chunk.content)
        )

        status.update(
            label="✅ Completed",
            state="complete",
            expanded=False
        )


    # Save assistant response
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })