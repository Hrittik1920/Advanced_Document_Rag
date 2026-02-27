import os
import json
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse 
from typing import List, Dict
import socketio
import uvicorn

# --- Your Existing RAG and LangChain Imports ---
from vector import retriever
# New, correct line
from script import format_documents, original_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

# --- CONFIGURATION ---
HISTORY_DIR = "chat_histories"
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
async def handle_chat_request(sid, data: dict):
    user_query = data.get("message")
    if not user_query:
        await sio.emit("error", {"message": "No message content found."}, to=sid)
        return

    print(f"Received query from {sid}: {user_query}")
    retrieved_docs = retriever.invoke(user_query)
    context = format_documents(retrieved_docs)

    if not context.strip():
        final_answer = "I could not find relevant information in the uploaded documents to answer your question."
    else:
        print(f"Invoking RAG chain for session {sid}...")
        response = await chain_with_history.ainvoke(
            {"context": context, "question": user_query},
            config={"configurable": {"session_id": sid}}
        )
        final_answer = response
    
    await sio.emit(
        "chat_response", 
        {"role": "assistant", "content": final_answer}, 
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
        reload=True
    )

# <--- REMOVED: The __main__ block is removed to prevent incorrect usage --->
# Running the server should be done via the command line.

# import os
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Dict
# import uvicorn

# # --- Import Your Existing RAG Logic ---
# # We will reuse the core components from your scripts.
# # Assuming 'chain' is a standard LangChain runnable (LCEL).
# from vector import retriever
# from script import format_documents, chain as original_chain

# # --- Import LangChain History Components ---
# from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.messages import BaseMessage

# # --- FastAPI App Initialization ---
# app = FastAPI(
#     title="RAG API Server with History",
#     description="An API for the multi-format document assistant that remembers conversation history.",
#     version="1.2.2" # Version updated
# )

# # --- CORS (Cross-Origin Resource Sharing) Middleware ---
# origins = [
#     "http://127.0.0.1:5500",
#     "http://localhost:5500",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # --- CONFIGURATION ---
# # Define the window size for the chat history.
# CHAT_HISTORY_WINDOW_SIZE = 10

# # --- Custom Chat History Class with a Fixed-Size Window ---
# class WindowedInMemoryChatMessageHistory(InMemoryChatMessageHistory):
#     """
#     An in-memory chat message history that only keeps the last 'k' messages.
#     """
#     # FIX: Declare 'k' with a default value and remove the custom __init__.
#     # Pydantic will now automatically handle initialization correctly.
#     k: int = 10

#     def add_messages(self, messages: List[BaseMessage]) -> None:
#         """Add messages to the store and enforce the window size."""
#         super().add_messages(messages)
#         # self.k is now correctly set by Pydantic's default initializer.
#         if len(self.messages) > self.k:
#             self.messages = self.messages[-self.k:]

# # --- In-Memory Store for Chat Histories ---
# store: Dict[str, BaseChatMessageHistory] = {}

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     """
#     Retrieves a chat history for a given session ID.
#     If one doesn't exist, a new windowed history is created.
#     """
#     if session_id not in store:
#         # This call now works perfectly with the new class definition.
#         store[session_id] = WindowedInMemoryChatMessageHistory(k=CHAT_HISTORY_WINDOW_SIZE)
#     return store[session_id]

# # --- Wrap the original chain with history management ---
# chain_with_history = RunnableWithMessageHistory(
#     original_chain,
#     get_session_history,
#     input_messages_key="question",
#     history_messages_key="chat_history",
# )

# # --- Define Request/Response Models ---
# class ChatRequestMessage(BaseModel):
#     role: str
#     content: str

# class ChatRequest(BaseModel):
#     model: str
#     messages: List[ChatRequestMessage]

# class ChatResponseMessage(BaseModel):
#     role: str
#     content: str

# class Choice(BaseModel):
#     index: int = 0
#     message: ChatResponseMessage

# class ChatCompletionResponse(BaseModel):
#     id: str = "chatcmpl-local-rag-with-history"
#     object: str = "chat.completion"
#     choices: List[Choice]

# # --- API Endpoint ---
# @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
# async def chat_completions(request: ChatRequest):
#     """
#     This endpoint receives a question, retrieves context, and returns an answer,
#     while maintaining a fixed-size conversation history.
#     """
#     user_query = request.messages[-1].content
#     session_id = f"session_{request.model}"

#     print(f"Session ID: {session_id}, Received query: {user_query}")

#     retrieved_docs = retriever.invoke(user_query)
#     context = format_documents(retrieved_docs)

#     if not context.strip():
#         final_answer = "I could not find relevant information in the uploaded documents to answer your question."
#     else:
#         print("Invoking RAG chain with retrieved context and history...")
#         response = chain_with_history.invoke(
#             {"context": context, "question": user_query},
#             config={"configurable": {"session_id": session_id}}
#         )
#         final_answer = response

#     return ChatCompletionResponse(
#         choices=[
#             Choice(
#                 message=ChatResponseMessage(role="assistant", content=final_answer)
#             )
#         ]
#     )

# @app.get("/")
# def read_root():
#     return {"status": "RAG API with History is running"}

# @app.get("/v1/history/{session_id}")
# def get_history(session_id: str):
#     """A simple endpoint to inspect the history of a session."""
#     full_session_id = f"session_{session_id}"
#     if full_session_id in store:
#         return {"session_id": full_session_id, "history": store[full_session_id].messages}
#     return {"error": "Session not found"}

# # @app.get("/v1/models")
# # def list_models():
# #     return {
# #         "object": "list",
# #         "data": [
# #             {
# #                 "id": "chatcmpl-local-rag-with-history",
# #                 "object": "model",
# #                 "created": 0,
# #                 "owned_by": "local-user",
# #             }
# #         ],
# #     }

# if __name__ == "__main__":
#     uvicorn.run(
#         "server:app",
#         host="0.0.0.0",
#         port=8000,
#         reload=True
# )

# import os
# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Dict

# # --- Import Your Existing RAG Logic ---
# # We will reuse the core components from your scripts.
# # Assuming 'chain' is a standard LangChain runnable (LCEL).
# from vector import retriever
# from script import format_documents, chain as original_chain

# # --- Import LangChain History Components ---
# from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory

# # --- FastAPI App Initialization ---
# app = FastAPI(
#     title="RAG API Server with History",
#     description="An API for the multi-format document assistant that remembers conversation history.",
#     version="1.1.0"
# )

# # --- CORS (Cross-Origin Resource Sharing) Middleware ---
# # This is the key change to fix the browser security error.
# origins = [
#     "http://127.0.0.1:5500",  # The origin of your frontend (e.g., from VS Code Live Server)
#     "http://localhost:5500",
#     # You can add other origins here if needed
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,  # Allows specific origins to make requests
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
#     allow_headers=["*"],  # Allows all headers
# )


# # --- In-Memory Store for Chat Histories ---
# # In a production environment, you would replace this with a persistent store like Redis or a database.
# store: Dict[str, BaseChatMessageHistory] = {}

# def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     """
#     Retrieves a chat history for a given session ID.
#     If one doesn't exist, a new one is created.
#     """
#     if session_id not in store:
#         store[session_id] = InMemoryChatMessageHistory()
#     return store[session_id]

# # --- Wrap the original chain with history management ---
# # This creates a new runnable that automatically handles the loading and saving of messages.
# chain_with_history = RunnableWithMessageHistory(
#     original_chain,
#     get_session_history,
#     input_messages_key="question",  # The key for the user's question in the input dictionary.
#     history_messages_key="chat_history", # The key for the chat history in the input dictionary.
# )

# # --- Define Request/Response Models ---
# # These models ensure that Open WebUI can communicate with your server.

# class ChatRequestMessage(BaseModel):
#     role: str
#     content: str

# class ChatRequest(BaseModel):
#     model: str
#     messages: List[ChatRequestMessage]

# class ChatResponseMessage(BaseModel):
#     role: str
#     content: str

# class Choice(BaseModel):
#     index: int = 0
#     message: ChatResponseMessage

# class ChatCompletionResponse(BaseModel):
#     id: str = "chatcmpl-local-rag-with-history"
#     object: str = "chat.completion"
#     choices: List[Choice]
    
# # --- API Endpoint ---
# @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
# async def chat_completions(request: ChatRequest):
#     """
#     This endpoint receives a question, retrieves context, and returns an answer,
#     while maintaining conversation history.
#     """
#     # 1. Extract the user's question from the request payload.
#     # The last message is typically the user's most recent question.
#     user_query = request.messages[-1].content
    
#     # 2. Create a unique session ID from the request model name.
#     # This allows for separate conversation histories for different models/sessions.
#     session_id = f"session_{request.model}"
    
#     print(f"Session ID: {session_id}, Received query: {user_query}")

#     # 3. Retrieve relevant documents using your existing retriever.
#     retrieved_docs = retriever.invoke(user_query)
#     context = format_documents(retrieved_docs)

#     if not context.strip():
#         # If no context is found, return a helpful message without invoking the chain.
#         final_answer = "I could not find relevant information in the uploaded documents to answer your question."
#     else:
#         # 4. Invoke the RAG chain with history management.
#         # The 'RunnableWithMessageHistory' wrapper handles the 'chat_history' automatically
#         # based on the 'session_id' provided in the config.
#         print("Invoking RAG chain with retrieved context and history...")
#         response = chain_with_history.invoke(
#             {"context": context, "question": user_query},
#             config={"configurable": {"session_id": session_id}}
#         )
#         # The 'response' from the chain is the final answer string.
#         final_answer = response

#     # 5. Format the response to be OpenAI-compatible.
#     return ChatCompletionResponse(
#         choices=[
#             Choice(
#                 message=ChatResponseMessage(role="assistant", content=final_answer)
#             )
#         ]
#     )

# @app.get("/")
# def read_root():
#     return {"status": "RAG API with History is running"}

# @app.get("/v1/history/{session_id}")
# def get_history(session_id: str):
#     """A simple endpoint to inspect the history of a session."""
#     full_session_id = f"session_{session_id}"
#     if full_session_id in store:
#         return {"session_id": full_session_id, "history": store[full_session_id].messages}
#     return {"error": "Session not found"}