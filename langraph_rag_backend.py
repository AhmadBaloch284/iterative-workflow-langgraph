from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

import requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY was not found in your .env file. "
        "Make sure the .env file is in the project folder and contains "
        "GEMINI_API_KEY=your_key_here"
    )


LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001",
)

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
    max_retries=2,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=GEMINI_API_KEY,
)


_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None


def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    if not file_bytes:
        raise ValueError("No PDF bytes received for ingestion.")

    thread_id = str(thread_id)
    temp_path = None

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf",
    ) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        if not docs:
            raise ValueError("The PDF contains no readable pages.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No readable text was found in the PDF.")

        vector_store = FAISS.from_documents(
            chunks,
            embeddings,
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        _THREAD_RETRIEVERS[thread_id] = retriever
        _THREAD_METADATA[thread_id] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str,
) -> dict:
    """Perform basic arithmetic: add, sub, mul, or div."""

    operation = operation.strip().lower()

    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed."}
            result = first_num / second_num
        else:
            return {
                "error": (
                    f"Unsupported operation '{operation}'. "
                    "Use add, sub, mul, or div."
                )
            }

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }

    except Exception as exc:
        return {"error": str(exc)}


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock quote using Alpha Vantage."""

    alpha_vantage_key = os.getenv("ALPHAVANTAGE_API_KEY")

    if not alpha_vantage_key:
        return {
            "error": (
                "ALPHAVANTAGE_API_KEY is not configured in .env. "
                "Add an Alpha Vantage API key to use the stock tool."
            )
        }

    symbol = symbol.strip().upper()

    url = "https://www.alphavantage.co/query"

    response = requests.get(
        url,
        params={
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": alpha_vantage_key,
        },
        timeout=15,
    )

    response.raise_for_status()
    data = response.json()

    if "Note" in data:
        return {"error": data["Note"]}

    if "Information" in data:
        return {"error": data["Information"]}

    return data


@tool
def rag_tool(
    query: str,
    thread_id: Optional[str] = None,
) -> dict:
    """
    Retrieve relevant information from the PDF uploaded to this chat.
    The thread_id must be included when calling this tool.
    """

    if not thread_id:
        return {
            "error": "No thread_id was supplied to rag_tool.",
            "query": query,
        }

    retriever = _get_retriever(thread_id)

    if retriever is None:
        return {
            "error": (
                "No document is indexed for this chat. "
                "Upload a PDF first."
            ),
            "query": query,
        }

    result = retriever.invoke(query)

    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(
            str(thread_id),
            {},
        ).get("filename"),
    }


tools = [
    search_tool,
    get_stock_price,
    calculator,
    rag_tool,
]

llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(
    state: ChatState,
    config=None,
):
    """Call Gemini and allow it to request available tools."""

    thread_id = None

    if config and isinstance(config, dict):
        thread_id = (
            config.get("configurable", {})
            .get("thread_id")
        )

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant powered by Google Gemini. "
            "Answer clearly and accurately. "
            "For questions about the uploaded PDF, use the rag_tool "
            "and always pass the current thread_id "
            f"'{thread_id}'. "
            "Do not invent information from a PDF. "
            "If the PDF does not contain the requested information, "
            "say that clearly. "
            "Use web search for current or general web information "
            "when appropriate. "
            "Use the calculator for arithmetic when appropriate. "
            "Use the stock tool for stock-price questions. "
            "If a user asks about a PDF but no PDF is indexed, "
            "ask them to upload one."
        )
    )

    messages = [
        system_message,
        *state["messages"],
    ]

    response = llm_with_tools.invoke(
        messages,
        config=config,
    )

    return {"messages": [response]}


tool_node = ToolNode(tools)


database_path = os.getenv(
    "CHATBOT_DATABASE",
    "chatbot.db",
)

conn = sqlite3.connect(
    database=database_path,
    check_same_thread=False,
)

checkpointer = SqliteSaver(conn=conn)


graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges(
    "chat_node",
    tools_condition,
)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(
    checkpointer=checkpointer,
)


def retrieve_all_threads():
    all_threads = set()

    for checkpoint in checkpointer.list(None):
        config = checkpoint.config or {}
        configurable = config.get("configurable", {})
        thread_id = configurable.get("thread_id")

        if thread_id:
            all_threads.add(str(thread_id))

    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})
