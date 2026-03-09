import os
import json
import asyncio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse 
from typing import List, Dict
import socketio
import uvicorn
import time
import functools
import inspect
import logging
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from script import format_documents, original_chain, condense_chain

from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# --- Global Task Tracker ---
active_tasks: Dict[str, asyncio.Task] = {}

def get_today_log_file():
    today = datetime.now().strftime("%d%m%Y")
    return os.path.join(LOG_DIR, f"log{today}.txt")

def timed(func):
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000
        log_timing(f"[TOTAL] {func.__name__} took {duration:.2f} ms")
        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        duration = (time.perf_counter() - start) * 1000
        log_timing(f"[TOTAL] {func.__name__} took {duration:.2f} ms")
        return result

    if inspect.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


def log_timing(message: str):
    log_file = get_today_log_file()
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

logger = logging.getLogger("timing_logger")
# --- Your Existing RAG and LangChain Imports ---
from vector import retriever
# New, correct line
from script import format_documents, original_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
from config import settings

# --- CONFIGURATION ---
HISTORY_DIR = settings.HISTORY_DIR
os.makedirs(HISTORY_DIR, exist_ok=True)

# --- Custom Chat History Class for File Persistence ---
# This class is already perfect and needs no changes.
class FileChatMessageHistory(BaseChatMessageHistory):
    session_id: str
    file_path: str

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.file_path = os.path.join(HISTORY_DIR, f"{self.session_id}.json")

    @property
    def messages(self) -> List[BaseMessage]:
        if not os.path.exists(self.file_path):
            return []
        with open(self.file_path, "r", encoding="utf-8") as f:
            try:
                dicts = json.load(f)
                return messages_from_dict(dicts)
            except json.JSONDecodeError:
                return []

    def add_messages(self, messages: List[BaseMessage]) -> None:
        current_messages = self.messages
        current_messages.extend(messages)
        with open(self.file_path, "w", encoding="utf-8") as f:
            dicts = messages_to_dict(current_messages)
            json.dump(dicts, f, indent=2)

    def clear(self) -> None:
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

# --- FastAPI & Socket.IO App Initialization ---
app = FastAPI(
    title="RAG API Server with Per-User History",
    description="An API for a multi-user document assistant that persists conversation history.",
    version="2.2.0"
)

# Mount the 'static' directory to serve your HTML, CSS, and JS files.
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the Socket.IO server with CORS enabled for all origins for easier development.
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
# Combine the FastAPI and Socket.IO apps into one.
socket_app = socketio.ASGIApp(sio, app)

# In-memory cache for history objects to improve performance.
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = FileChatMessageHistory(session_id=session_id)
    return store[session_id]

# Wrap the LangChain chain with the history management logic.
chain_with_history = RunnableWithMessageHistory(
    original_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key=HISTORY_DIR,
)

# --- Socket.IO Event Handlers ---
@sio.event
async def connect(sid, environ):
    print(f"New client connected: {sid}")

@sio.event
async def disconnect(sid):
    print(f"Client disconnected: {sid}")
    # Clean up from the in-memory cache on disconnect.
    if sid in store:
        del store[sid]

@sio.on("chat_request")
@timed
async def handle_chat_request(sid, data: dict):
    log_timing("\n================ NEW REQUEST ================")
    log_timing(f"Session: {sid}")
    
    if not data.get("message"):
        await sio.emit("error", {"message": "No message content found."}, to=sid)
        return

    # Create the task and store it
    task = asyncio.create_task(run_llm_logic(sid, data))
    active_tasks[sid] = task
    
    try:
        await task
    except asyncio.CancelledError:
        log_timing(f"Task for {sid} was STOPPED by user.")
        # When cancelled, the logic in run_llm_logic stops immediately
    finally:
        # Always clean up the task from memory
        active_tasks.pop(sid, None)
        
async def run_llm_logic(sid, data: dict):
    """Actual RAG and LLM logic extracted from the main handler."""
    user_query = data.get("message")
    
    # 0. Get history and condense the question
    history_obj = get_session_history(sid)
    chat_history = history_obj.messages

    if chat_history:
        standalone_question = await condense_chain.ainvoke({
            "question": user_query,
            settings.HISTORY_DIR: chat_history,
        })
    else:
        standalone_question = user_query

    # 1. Retrieval
    retrieved_docs = await retriever.ainvoke(standalone_question)
    context = format_documents(retrieved_docs)
    if not context:
        context = "No relevant documents found in the knowledge base."

    # 2. Streaming Response
    full_response = ""
    try:
        async for chunk in chain_with_history.astream(
            {"context": context, "question": user_query},
            config={"configurable": {"session_id": sid}}
        ):
            chunk_text = chunk if isinstance(chunk, str) else chunk.content
            full_response += chunk_text
            
            if chunk_text:
                await sio.emit("chat_stream_chunk", {"chunk": chunk_text}, to=sid)

        # Signal completion
        await sio.emit("chat_stream_end", {"message": "Stream complete"}, to=sid)
        
    except Exception as e:
        print(f"Error during streaming: {e}")
        await sio.emit("error", {"message": f"Stream error: {str(e)}"}, to=sid)

@sio.on("stop_generation")
async def handle_stop_generation(sid):
    if sid in active_tasks:
        active_tasks[sid].cancel()
        # Add this line to tell the frontend we stopped manually
        await sio.emit("chat_stopped_manually", to=sid)
        print(f"Manual stop triggered for session: {sid}")

# --- Standard FastAPI Endpoints ---
@app.get("/")
async def read_root():
    """Serves the main index.html file for the chat interface."""
    return FileResponse("templates/index.html")

@app.get("/v1/history/{session_id}")
def get_history(session_id: str):
    """A utility endpoint to inspect the history file of a given session."""
    history = FileChatMessageHistory(session_id=session_id)
    return history.messages

if __name__ == "__main__":
    uvicorn.run(
        "server:socket_app",
        host="0.0.0.0",
        port=8099,
        reload=False,  # Set to True if you want auto-reload during development
    )