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

from datetime import datetime

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

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
    history_messages_key="chat_history",
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
    user_query = data.get("message")
    total_start = time.perf_counter()
    log_timing("\n================ NEW REQUEST ================")
    log_timing(f"Session: {sid}")
    log_timing(f"Question: {user_query}")
    if not user_query:
        await sio.emit("error", {"message": "No message content found."}, to=sid)
        return

    print(f"Received query from {sid}: {user_query}")
    
    # 1. Use ainvoke to prevent blocking the async event loop
    start = time.perf_counter()
    retrieved_docs = await retriever.ainvoke(user_query)
    log_timing(f"Retriever took {(time.perf_counter() - start) * 1000:.2f} ms")

    start = time.perf_counter()
    context = format_documents(retrieved_docs)
    log_timing(f"Context formatting took {(time.perf_counter() - start) * 1000:.2f} ms")
    
    # Ensure context is never None/empty - use a default if needed
    if not context:
        context = "No relevant documents found in the knowledge base."

    print(f"Invoking RAG chain for session {sid} with streaming...")
    
    # 2. Get the full response using ainvoke (compatible with RunnableWithMessageHistory)
    full_response = ""
    try:
        start = time.perf_counter()
        history_obj = get_session_history(sid)

        log_timing("----- SESSION HISTORY -----")
        for msg in history_obj.messages:
            log_timing(f"{msg.type.upper()}: {msg.content}")
        log_timing("----- END SESSION HISTORY -----")
        response = await chain_with_history.ainvoke(
            {"context": context, "question": user_query},
            config={"configurable": {"session_id": sid}}
        )
        log_timing(f"LLM inference took {(time.perf_counter() - start) * 1000:.2f} ms")
        full_response = response
        log_timing("Answer:")
        log_timing(full_response)
        
        # Stream the response in chunks to the client
        chunk_size = 20
        start = time.perf_counter()
        for i in range(0, len(full_response), chunk_size):
            chunk = full_response[i:i + chunk_size]
            if chunk:
                await sio.emit(
                    "chat_stream_chunk", 
                    {"chunk": chunk}, 
                    to=sid
                )
                # Small delay to create streaming effect
                await asyncio.sleep(0.05)
        log_timing(f"Streaming took {(time.perf_counter() - start) * 1000:.2f} ms")
        
        # Emit end-of-stream signal
        await sio.emit(
            "chat_stream_end", 
            {"message": "Stream complete"},
            to=sid
        )
        
    except Exception as e:
        print(f"Error during streaming: {e}")
        await sio.emit(
            "error", 
            {"message": f"Stream error: {str(e)}"}, 
            to=sid
        )

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
        port=8000,
        reload=False,  # Set to True if you want auto-reload during development
    )