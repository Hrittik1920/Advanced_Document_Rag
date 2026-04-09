#sever.py
import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, StreamingResponse
from typing import List, Dict
import socketio
import uvicorn
import re
import time
import functools
import inspect
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import warnings
import sys
import io
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
def log_debug(message: str):
    log_timing(f"[DEBUG] {message}")

logger = logging.getLogger("timing_logger")
# --- Your Existing RAG and LangChain Imports ---
from retriever import retriever
import fitz

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
                all_messages = messages_from_dict(dicts)
                all_messages = all_messages[-6:]
                for msg in all_messages:
                    if hasattr(msg, "content") and isinstance(msg.content, str):
                        msg.content = msg.content.strip()[:300]
                return all_messages
            except json.JSONDecodeError:
                return []

    def add_messages(self, messages: List[BaseMessage]) -> None:
        # current_messages = self.messages
        if not os.path.exists(self.file_path):
            current_messages = []
        else:
            with open(self.file_path, "r", encoding="utf-8") as f:
                try:
                    dicts = json.load(f)
                    current_messages = messages_from_dict(dicts)
                except json.JSONDecodeError:
                    current_messages = []
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
    log_debug(f"Received query: {user_query}")
    
    # 0. Get history and condense the question
    history_obj = get_session_history(sid)
    chat_history = history_obj.messages

    if chat_history:
        condensed_result = await condense_chain.ainvoke({
            "question": user_query,     
            settings.HISTORY_DIR: chat_history,
        })
        if isinstance(condensed_result, dict):
            standalone_question = condensed_result.get("retrieval_queries", [user_query])   # safety net
            generation_question = condensed_result.get("generation_question", user_query)
        elif isinstance(condensed_result, list):
            standalone_question = condensed_result
            generation_question = " and " .join(condensed_result)
        else:
            standalone_questions = [user_query]
            generation_question = user_query
    else:
        standalone_question = [user_query] 
        generation_question = user_query
    log_debug(f"Standalone question: {standalone_question}")
    log_debug(f"Generation question: {generation_question}")
    # 1. Retrieval
    retrieval_tasks = [retriever.ainvoke(q) for q in standalone_question]
    results = await asyncio.gather(*retrieval_tasks)
    unique_docs_map = {}
    for doc_list in results:
        for doc in doc_list:
            # Use page_content as the unique key (or doc.metadata['chunk_id'] if you have it)
            if doc.page_content not in unique_docs_map:
                unique_docs_map[doc.page_content] = doc
                
    retrieved_docs = list(unique_docs_map.values())
    log_debug(f"Total Unique Retrieved Docs: {len(retrieved_docs)}")
    context, citation = format_documents(retrieved_docs)
    if not context:
        context = "No relevant documents found in the knowledge base."
        citation=[]
    preview_context = context[:500] + "..." if len(context) > 500 else context
    with open("chunk.txt", "w", encoding="utf-8") as f:
        f.write(context)
    log_debug(f"Context Preview: {preview_context}")
    log_debug(f"Context Length: {len(context)} chars")
    # 2. Streaming Response
    full_response = ""
    final_prompt_preview = f"""
        --- FINAL PROMPT ---
        Context:
        {context[:800]}

        Question:
        {generation_question}
        ---------------------
        """
    log_debug(final_prompt_preview)
    
    # Initialize final_citations to empty list BEFORE try block
    final_citations = []
    
    try:
        async for chunk in chain_with_history.astream(
            {"context": context, "question": generation_question},
            config={"configurable": {"session_id": sid}}
        ):
            chunk_text = chunk if isinstance(chunk, str) else chunk.content
            full_response += chunk_text
            
            if chunk_text:
                await sio.emit("chat_stream_chunk", {"chunk": chunk_text}, to=sid)

        # Signal completion - extract citations from response
        try:
            used_ids_matches = re.findall(r'[\[【](\d+)[\]】]', full_response)
            used_ids = set()
            for idx_str in used_ids_matches:
                try:
                    used_ids.add(int(idx_str))
                except (ValueError, TypeError):
                    log_debug(f"⚠️ Could not convert citation index '{idx_str}' to int")
                    continue
            
            # Filter citations that were actually used in response
            if used_ids:
                final_citations = [c for c in citation if c["id"] in used_ids]
                
        except Exception as cite_error:
            log_debug(f"⚠️ Error extracting citations: {cite_error}")
            final_citations = citation  # fallback to all citations
        
        await sio.emit("chat_stream_end", {"message": "Stream complete", "citation": final_citations}, to=sid)
        log_debug(f"Final Response Preview: {full_response[:500]}")
        log_debug(f"Response Length: {len(full_response)} chars")
        log_debug(f"Citations sent: {len(final_citations)}")
        
    except Exception as e:
        log_debug(f"Error during streaming: {e}")
        log_debug(f"Error type: {type(e).__name__}")
        import traceback
        log_debug(f"Traceback: {traceback.format_exc()}")
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

@app.get("/v1/document-page")
async def get_document_page(source: str, page: int = 0):
    """
    Renders a single page of a document as a PNG image.
    - source: full or relative path to the file (from doc metadata)
    - page:   0-indexed page number
    """
    # Resolve path safely — only allow files inside DOCUMENTS_DIR
    if not os.path.isabs(source) and not source.startswith(settings.DOCUMENTS_DIR):
        safe_filename = os.path.basename(source)
        target_path = os.path.join(settings.DOCUMENTS_DIR, safe_filename)
    else:
        target_path = source
    abs_source = os.path.realpath(target_path)
    abs_allowed = os.path.realpath(settings.DOCUMENTS_DIR)
    if not abs_source.startswith(abs_allowed):
        print(f"DEBUG: Blocked access. {abs_source} is not inside {abs_allowed}")
        raise HTTPException(status_code=403, detail="Access denied.")
    if not os.path.exists(abs_source):
        print(f"DEBUG: File not found at {abs_source}")
        raise HTTPException(status_code=404, detail="Document not found.")

    ext = os.path.splitext(abs_source)[1].lower()

    # ── PDF ──────────────────────────────────────
    if ext == ".pdf":
        doc = fitz.open(abs_source)
        if page < 0 or page >= len(doc):
            raise HTTPException(status_code=400, detail=f"Page {page} out of range (0–{len(doc)-1}).")
        pdf_page = doc[page]
        mat = fitz.Matrix(2.0, 2.0)   # 2× zoom for readability
        pix = pdf_page.get_pixmap(matrix=mat, alpha=False)
        img_bytes = pix.tobytes("png")

    # ── Word / DOCX — convert the whole doc to PDF first ─
    elif ext in (".docx", ".doc"):
        try:
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp_pdf = tmp.name
            subprocess.run(
                ["libreoffice", "--headless", "--convert-to", "pdf",
                 "--outdir", os.path.dirname(tmp_pdf), abs_source],
                check=True, capture_output=True
            )
            converted = os.path.splitext(abs_source)[0] + ".pdf"
            doc = fitz.open(converted)
            pdf_page = doc[min(page, len(doc) - 1)]
            mat = fitz.Matrix(2.0, 2.0)
            pix = pdf_page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("png")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")

    # ── Standalone image file ─────────────────────
    elif ext in (".png", ".jpg", ".jpeg", ".webp"):
        with open(abs_source, "rb") as f:
            img_bytes = f.read()

    else:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")

    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")


if __name__ == "__main__":
    port=8010
    reload_flag=False
    if "dev" in sys.argv:
        port=8100
        reload_flag=True
    uvicorn.run(
        "server:socket_app",
        host="0.0.0.0",
        port=port,
        reload=reload_flag,  # Set to True if you want auto-reload during development
    )