from langgraph.graph import StateGraph, START
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv

import asyncio
import aiosqlite


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
# 2. MCP client
# ------------------------
client = MultiServerMCPClient(
    {
        "arith": {
            "transport": "stdio",
            "command": "python",
            "args": [
                "C:/Users/hp/OneDrive - Higher Education Commission/Desktop/Iterative_workflow Langgraph/arith_server.py"
            ],
        }
    }
)


# ------------------------
# 3. Search tool
# ------------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


# ------------------------
# 4. State
# ------------------------
class ChatState(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages
    ]


# ------------------------
# 5. Build chatbot
# ------------------------
async def build_graph():

    print("Loading MCP tools...")

    try:
        tools = await client.get_tools()
        print(f"MCP tools loaded: {len(tools)}")

    except Exception as e:
        print("MCP server connection failed:")
        print(e)

        print("Continuing with local tools...")

        tools = []

    # Add search tool
    tools.append(search_tool)

    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState):

        response = await llm_with_tools.ainvoke(
            state["messages"]
        )

        return {
            "messages": [response]
        }

    tool_node = ToolNode(tools)

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

    return graph
# ------------------------
# 6. Create checkpointer
# ------------------------
async def create_chatbot():

    graph = await build_graph()

    conn = await aiosqlite.connect(
        "chatbot.db"
    )

    checkpointer = AsyncSqliteSaver(conn)

    chatbot = graph.compile(
        checkpointer=checkpointer
    )

    return chatbot


# ------------------------
# 7. Global chatbot
# ------------------------
chatbot = asyncio.run(
    create_chatbot()
)


# ------------------------
# 8. Test chatbot
# ------------------------
async def main():

    result = await chatbot.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content="Add an expense of 500 rupees for a Udemy course on 10 nov"
                )
            ]
        },
        config={
            "configurable": {
                "thread_id": "test-thread"
            }
        }
    )

    print(
        result["messages"][-1].content
    )


# ------------------------
# 9. Run directly
# ------------------------
if __name__ == "__main__":
    asyncio.run(main())