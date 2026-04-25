#server.py
import os
import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, StreamingResponse
from typing import Any, List, Dict
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
import tempfile
from extraction import MultiFormatDocumentLoader, SuryaLayoutExtractor
warnings.filterwarnings("ignore", category=FutureWarning)
from script import format_documents, original_chain, condense_chain, router_chain
import os 
from datetime import datetime
from main import app as langgraph_app
from logger_utils import log_timing, log_debug, timed
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# --- Global Task Tracker ---
active_tasks: Dict[str, asyncio.Task] = {}
# --- Your Existing RAG and LangChain Imports ---
from retriever import retriever
import fitz

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, messages_from_dict, messages_to_dict
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

# Initialize the Socket.IO server with CORS stricted to allowed origins
# allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8010").split(",")
sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
# Combine the FastAPI and Socket.IO apps into one.
socket_app = socketio.ASGIApp(sio, app)

# In-memory cache for history objects to improve performance.
store: Dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = FileChatMessageHistory(session_id=session_id)
    return store[session_id]

def _extract_stream_text(event: Dict[str, Any]) -> str:
    """Extract token text from either chat-model or llm stream events."""
    data = event.get("data") or {}
    chunk = data.get("chunk")
    if chunk is None:
        return ""

    if isinstance(chunk, str):
        return chunk

    text = getattr(chunk, "text", None)
    if isinstance(text, str):
        return text

    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        return "".join(parts)

    return ""

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
    """Actual RAG and LLM logic orchestrated by LangGraph."""
    user_query = data.get("message", "")
    file_info = data.get("file")
    ui_target_files= data.get("target_files",[])
    log_debug(f"Received query from {sid}: {user_query}")
    
    uploaded_doc_text = ""
    tmp_path = None
    
    # ------------------ FILE UPLOAD HANDLING ------------------
            
    if file_info:
        file_name = file_info.get("name", "uploaded_document")
        file_bytes = file_info.get("data")
        suffix = os.path.splitext(file_name)[1]
        log_debug(f"Processing uploaded file: {file_name}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            tmp_path = tmp.name 

        try:
            def process_uploaded_file():
                thread_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(thread_loop)
                try:
                    
                    with SuryaLayoutExtractor() as loader:
                        return loader.extract(tmp_path)
                finally:
                    thread_loop.close()
                
            docs = await asyncio.to_thread(process_uploaded_file)
            if isinstance(docs, str):
                uploaded_doc_text=docs
            elif isinstance(docs, list):
                uploaded_doc_text = "\n\n".join([d.page_content for d in docs])
            else:
                uploaded_doc_text=str(docs)
            log_debug(f"Extracted {len(uploaded_doc_text)} chars from uploaded document.")
            
        except Exception as e:
            log_debug(f"File extraction error: {e}")
            await sio.emit("chat_stream_chunk", {"chunk": "⚠️ *Warning: Could not read the uploaded file correctly. Proceeding with database context only.* \n\n"}, to=sid)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as cleanup_error:
                    log_debug(f"Could not remove temp file: {cleanup_error}")

    # ------------------ LANGGRAPH ORCHESTRATION ------------------
    # 0. Get history
    files_response = await get_available_files()
    available_files = files_response.get("files", [])
    history_obj = get_session_history(sid)

    # 1. Initialize Graph State
    initial_state = {
        "question": user_query,
        "chat_histories": history_obj.messages,
        
        # Note: If your LangGraph needs uploaded_doc_text, you can pass it here
        "uploaded_doc_text": uploaded_doc_text,
        "available_files": available_files,
        "target_files": ui_target_files
    }

    final_response_nodes = {"text_path", "synthesize_response"}
    final_citations = []
    final_response_text = ""
    fallback_final_response = ""

    start = time.perf_counter()

    try:
        # Use astream_events to catch both node updates AND LLM tokens
        async for event in langgraph_app.astream_events(initial_state, version="v2"):
            kind = event["event"]
            
            # Catch Node Completions (For UI status updates and Citations)
            if kind == "on_chain_end":
                node_name = event["name"]
                state_update = event["data"].get("output", {})
                
                if isinstance(state_update, dict):
                    if "citations" in state_update and not final_citations:
                        final_citations = state_update["citations"]
                    if node_name in final_response_nodes:
                        maybe_final = state_update.get("final_response", "")
                        if isinstance(maybe_final, str) and maybe_final.strip():
                            fallback_final_response = maybe_final

                    if node_name == "classify_and_retrieve":
                        if state_update.get("math_intent"):
                            await sio.emit("chat_stream_chunk", {"chunk": "\n🔢 **Math intent detected. Activating Math Agent...**\n"}, to=sid)
                        else:
                            await sio.emit("chat_stream_chunk", {"chunk": "🔍 *Searching knowledge base...*\n"}, to=sid)
                    
                    # elif node_name == "math_extract":
                    #     await sio.emit("chat_stream_chunk", {"chunk": "📊 *Parsing numerical data...*\n"}, to=sid)
                    elif node_name == "math_generate":
                        await sio.emit("chat_stream_chunk", {"chunk": "⚙️ *Generating calculation logic...*\n"}, to=sid)
                    elif node_name == "math_execute":
                        await sio.emit("chat_stream_chunk", {"chunk": "⏳ *Computing results in sandbox...*\n\n"}, to=sid)

            # Catch live LLM token streaming
            elif kind in ("on_chat_model_stream", "on_llm_stream"):
                event_node = (event.get("metadata") or {}).get("langgraph_node")
                if event_node and event_node not in final_response_nodes:
                    continue
                chunk_text = _extract_stream_text(event)
                if chunk_text:
                    final_response_text += chunk_text
                    await sio.emit("chat_stream_chunk", {"chunk": chunk_text}, to=sid)

        if not final_response_text and fallback_final_response:
            final_response_text = fallback_final_response
            await sio.emit("chat_stream_chunk", {"chunk": final_response_text}, to=sid)

        if not final_response_text:
            final_response_text = "I could not generate a response this time. Please try again."
            await sio.emit("chat_stream_chunk", {"chunk": final_response_text}, to=sid)

        total_time = (time.perf_counter() - start) * 1000
        log_timing(f"[SUMMARY] Total LangGraph Execution: {total_time:.0f}ms")

        # 3. Save to history and close stream
        history_obj.add_messages([
            HumanMessage(content=user_query),
            AIMessage(content=final_response_text)
        ])
        
        await sio.emit("chat_stream_end", {"message": "Stream complete", "citation": final_citations}, to=sid)

    except asyncio.CancelledError:
        log_timing(f"Generation cancelled for {sid}")
        raise # Allow the outer task wrapper to handle the cancellation

    except Exception as e:
        log_debug(f"Error during Graph Execution: {e}")
        import traceback
        log_debug(traceback.format_exc())
        await sio.emit("error", {"message": str(e)}, to=sid)

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
    """A utility endpoint to inspect the history file of a given session.
    
    Note: This endpoint should be protected with authentication in production.
    Currently gated behind DEBUG_HISTORY_ENDPOINT env flag.
    """
    if not os.getenv("DEBUG_HISTORY_ENDPOINT", "false").lower() == "true":
        raise HTTPException(status_code=403, detail="History endpoint disabled.")
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
    
    # Use os.path.commonpath to ensure strict containment
    try:
        common = os.path.commonpath([abs_source, abs_allowed])
        if common != abs_allowed:
            raise HTTPException(status_code=403, detail="Access denied.")
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied.")
    
    if not os.path.exists(abs_source):
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
            import subprocess
            with tempfile.TemporaryDirectory() as tmp_dir:
                src_stem = os.path.splitext(os.path.basename(abs_source))[0]
                subprocess.run(
                    ["libreoffice", "--headless", "--convert-to", "pdf",
                     "--outdir", tmp_dir, abs_source],
                    check=True, capture_output=True, timeout=30
                )
                converted = os.path.join(tmp_dir, f"{src_stem}.pdf")
                if not os.path.exists(converted):
                    raise HTTPException(status_code=500, detail="PDF conversion did not produce output.")
                doc = fitz.open(converted)
                pdf_page = doc[min(page, len(doc) - 1)]
                mat = fitz.Matrix(2.0, 2.0)
                pix = pdf_page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes("png")
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=500, detail="Conversion timed out.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Conversion failed: {e}")

    # ── Standalone image file ─────────────────────
    elif ext in (".png", ".jpg", ".jpeg", ".webp"):
        with open(abs_source, "rb") as f:
            img_bytes = f.read()

    else:
        raise HTTPException(status_code=415, detail=f"Unsupported file type: {ext}")

    return StreamingResponse(io.BytesIO(img_bytes), media_type="image/png")

@app.get("/v1/available-files")
async def get_available_files():
    """Returns a list of all available documents for the frontend mention feature."""
    files = []
    if os.path.exists(settings.DOCUMENTS_DIR):
        for f in os.listdir(settings.DOCUMENTS_DIR):
            if os.path.isfile(os.path.join(settings.DOCUMENTS_DIR, f)) and not f.startswith("."):
                files.append(f)
    return {"files": files}


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
