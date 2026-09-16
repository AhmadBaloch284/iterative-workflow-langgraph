import os
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

# Load Environment Variables
load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# Define Graph State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Chat Node
def chat_node(state: ChatState):

    response = llm.invoke(state["messages"])

    return {
        "messages": [response]
    }

# Memory Checkpointer
checkpointer = InMemorySaver()

# Build Graph
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

# Compile Graph
chatbot = graph.compile(checkpointer=checkpointer)