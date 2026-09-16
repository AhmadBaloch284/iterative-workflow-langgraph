from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from dotenv import load_dotenv

import sqlite3
import requests
import os

# ------------------------
# 0. Load environment
# ------------------------
load_dotenv()

# ------------------------
# 1. LLM - Gemini
# ------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0
)

# ------------------------
# 2. Tools
# ------------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str
) -> dict:
    """
    Perform a basic arithmetic operation.

    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num

        elif operation == "sub":
            result = first_num - second_num

        elif operation == "mul":
            result = first_num * second_num

        elif operation == "div":
            if second_num == 0:
                return {
                    "error": "Division by zero is not allowed"
                }

            result = first_num / second_num

        else:
            return {
                "error": f"unsupported operation '{operation}'"
            }

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result
        }

    except Exception as e:
        return {
            "error": str(e)
        }


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a given symbol
    such as AAPL or TSLA using Alpha Vantage.
    """

    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

    if not api_key:
        return {
            "error": "ALPHA_VANTAGE_API_KEY is missing from .env"
        }

    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE"
        f"&symbol={symbol}"
        f"&apikey={api_key}"
    )

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()

        return response.json()

    except requests.RequestException as e:
        return {
            "error": str(e)
        }


tools = [
    search_tool,
    get_stock_price,
    calculator
]

# Make Gemini tool-aware
llm_with_tools = llm.bind_tools(tools)

# ------------------------
# 3. State
# ------------------------
class ChatState(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]

# ------------------------
# 4. Nodes
# ------------------------
def chat_node(state: ChatState):
    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response]
    }


tool_node = ToolNode(tools)

# ------------------------
# 5. Checkpointer
# ------------------------
conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn=conn)

# ------------------------
# 6. Graph
# ------------------------
graph = StateGraph(ChatState)

graph.add_node(
    "chat_node",
    chat_node
)

graph.add_node(
    "tools",
    tool_node
)

graph.add_edge(
    START,
    "chat_node"
)

graph.add_conditional_edges(
    "chat_node",
    tools_condition
)

graph.add_edge(
    "tools",
    "chat_node"
)

# Compile graph
chatbot = graph.compile(
    checkpointer=checkpointer
)

# ------------------------
# 7. Helper
# ------------------------
def retrieve_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(None):

        thread_id = checkpoint.config[
            "configurable"
        ]["thread_id"]

        all_threads.add(thread_id)

    return list(all_threads)

# ------------------------
# 8. Test
# ------------------------
config = {
    "configurable": {
        "thread_id": "test-thread"
    }
}

out = chatbot.invoke(
    {
        "messages": [
            HumanMessage(
                content=(
                    "What is the stock price of Apple? "
                    "How much would it cost to purchase 50 shares?"
                )
            )
        ]
    },
    config=config
)

# ------------------------
# 9. Clean output
# ------------------------
content = out["messages"][-1].content

if isinstance(content, list):

    for block in content:

        if isinstance(block, dict):
            if "text" in block:
                print(block["text"])

else:
    print(content)