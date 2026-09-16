import os
import sqlite3
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Create LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Define graph state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Define chat node
def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {
        "messages": [response]
    }

# Create SQLite connection
conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False
)

# Create SQLite checkpointer
checkpointer = SqliteSaver(conn)

# Create graph
graph = StateGraph(ChatState)

# Add node
graph.add_node("chat_node", chat_node)

# Add edges
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Compile graph with SQLite persistence
chatbot = graph.compile(
    checkpointer=checkpointer
)

# Retrieve all conversation thread IDs
def retrieve_all_threads():
    all_threads = set()

    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        all_threads.add(thread_id)

    return list(all_threads)

# Retrieve messages from a specific conversation
def retrieve_conversation(thread_id):
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    state = chatbot.get_state(config)

    if not state.values:
        return []

    return state.values.get("messages", [])