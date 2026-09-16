import streamlit as st
import uuid

from langgraph_database_backend import (
    chatbot,
    retrieve_all_threads,
    retrieve_conversation
)

from langchain_core.messages import HumanMessage

# Generate a unique thread ID
def generate_thread_id():
    return str(uuid.uuid4())

# Initialize thread ID
if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

# Initialize message history
if "message_history" not in st.session_state:
    st.session_state.message_history = []

# Retrieve existing threads
if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads()

# Configure sidebar
st.sidebar.title("LangGraph ChatBot")

# Create new chat
if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True
):
    st.session_state.thread_id = generate_thread_id()
    st.session_state.message_history = []
    st.rerun()

# Refresh conversation threads
st.session_state.chat_threads = retrieve_all_threads()

# Display conversations
st.sidebar.markdown("## My Conversations")

for thread_id in st.session_state.chat_threads:

    conversation = retrieve_conversation(thread_id)

    if not conversation:
        continue

    first_message = None

    for message in conversation:
        if message.type == "human":
            first_message = message.content
            break

    if not first_message:
        continue

    title = first_message[:30]

    if st.sidebar.button(
        title,
        key=thread_id,
        use_container_width=True
    ):
        st.session_state.thread_id = thread_id

        st.session_state.message_history = [
            {
                "role": "user" if message.type == "human" else "assistant",
                "content": message.content
            }
            for message in conversation
            if message.type in ["human", "ai"]
        ]

        st.rerun()

# Display previous messages
for message in st.session_state.message_history:

    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_input = st.chat_input("Type here")

if user_input:

    # Display user message
    with st.chat_message("user"):
        st.write(user_input)

    # Add user message to session state
    st.session_state.message_history.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Configure thread
    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }

    # Generate streaming response
    def response_generator():

        for message_chunk, metadata in chatbot.stream(
            {
                "messages": [
                    HumanMessage(content=user_input)
                ]
            },
            config=config,
            stream_mode="messages"
        ):

            if (
                hasattr(message_chunk, "content")
                and message_chunk.content
            ):
                yield message_chunk.content

    # Display assistant response
    with st.chat_message("assistant"):
        ai_message = st.write_stream(response_generator())

    # Add assistant message to session state
    st.session_state.message_history.append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )

    # Refresh conversations
    st.session_state.chat_threads = retrieve_all_threads()

    st.rerun()