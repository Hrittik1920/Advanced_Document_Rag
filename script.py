# import os
# import json
# from langchain_ollama.llms import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.chat_history import BaseChatMessageHistory
# from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict
# from vector import retriever  # ← your retriever object
# import shutil
# import uuid


# # Configuration
# MODEL_NAME        = "llama3.2-vision:11b"
# HISTORY_DIR = "./chat_histories"
# MAX_CONTEXT_CHARS = 8_000        # keep context window safe

# # Create the history directory if it doesn't exist
# os.makedirs(HISTORY_DIR, exist_ok=True)

# # --- NEW: Custom Chat History Class for Per-Session Persistence ---
# class PerSessionFileChatMessageHistory(BaseChatMessageHistory):
#     """
#     Chat message history that stores messages in a unique JSON file for each session.
#     """
#     def __init__(self, session_id: str, base_path: str):
#         self.file_path = os.path.join(base_path, f"{session_id}.json")

#     @property
#     def messages(self) -> list[BaseMessage]:
#         if not os.path.exists(self.file_path):
#             return []
#         with open(self.file_path, "r", encoding="utf-8") as f:
#             try:
#                 dicts = json.load(f)
#                 return messages_from_dict(dicts)
#             except json.JSONDecodeError:
#                 return []

#     def add_messages(self, messages: list[BaseMessage]) -> None:
#         current_messages = self.messages
#         current_messages.extend(messages)
#         with open(self.file_path, "w", encoding="utf-8") as f:
#             dicts = messages_to_dict(current_messages)
#             json.dump(dicts, f, indent=2)

#     def clear(self) -> None:
#         if os.path.exists(self.file_path):
#             os.remove(self.file_path)
# # Model

# model = OllamaLLM(model=MODEL_NAME)

# # --- CHANGED: Chat-history helper and cache ---
# # In-memory cache for history objects to improve performance.
# store = {}

# # Chat-history helper
# def get_message_history(session_id: str) -> BaseChatMessageHistory:
#     """
#     Return a PerSessionFileChatMessageHistory object for the given session_id.
#     """
#     if session_id not in store:
#         store[session_id] = PerSessionFileChatMessageHistory(session_id, HISTORY_DIR)
#     return store[session_id]


# # def get_message_history(session_id: str):
# #     """
# #     Return a FileChatMessageHistory that stores the entire conversation
# #     in chat_history.json.  session_id is ignored because this version of
# #     FileChatMessageHistory does not support namespacing.
# #     """
# #     return FileChatMessageHistory(
# #         file_path=os.path.join(os.getcwd(), CHAT_HISTORY_FILE)
# #     )


# # Prompt template

# prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         "You are an AI assistant. Your task is to answer user questions "
#         "based *only* on the provided context.\n"
#         "If the context does not contain the answer, state that clearly.\n"
#         "If someone greets you always greet them properly saying about you\n"
#         "**Important** don't give Key Takeways and Assitant Response Only for greetings\n"
#         "Structure your answer in two parts:\n"
#         "1. **Key Takeaways**: A bulleted list of the most important points "
#         "from the context. (Cite the source file and page/row if available.)\n"
#         "2. **Assistant Response**: A detailed answer synthesized from the "
#         "key takeaways.\n\n"
#         "CONTEXT:\n---\n{context}\n---"
#     ),
#     MessagesPlaceholder(variable_name="history"),
#     ("human", "{question}"),
# ])


# # Helper to format retrieved documents for the prompt

# def format_documents(docs: list) -> str:
#     """
#     Turn Document objects into a single string, truncate if too long.
#     """
#     formatted = []
#     total_chars = 0

#     for idx, doc in enumerate(docs, start=1):
#         source_info = f"Source: {os.path.basename(doc.metadata.get('source', 'N/A'))}"
#         if 'page' in doc.metadata:
#             source_info += f", Page {doc.metadata['page']}"
#         if 'row' in doc.metadata:
#             source_info += f", Row {doc.metadata['row']}"

#         entry = f"[{idx}] {doc.page_content} ({source_info})"

#         # prevent context overflow
#         if total_chars + len(entry) > MAX_CONTEXT_CHARS:
#             print("Warning: context limit reached; truncating retrieved documents.")
#             break

#         formatted.append(entry)
#         total_chars += len(entry)

#     return "\n\n".join(formatted)


# # Main runnable chain (RAG)

# chain = RunnableWithMessageHistory(
#     prompt | model,
#     get_message_history,
#     input_messages_key="question",
#     history_messages_key="history"
# )


# # CLI loop

# def main() -> None:
#     print("Welcome to the Multi-Format Document Assistant!")
#     print("Ask questions about your documents. Type 'exit' or 'quit' to end.")
#     session_id = "user_session_main"

#     while True:
#         print("-" * 120)
#         question = input("Ask a question: ").strip()

#         if question.lower() in {"exit", "quit", "q"}:
#             print("Thank you for using the Document Assistant!")
#             if os.path.exists('./chroma_langchain_db'):
#                 shutil.rmtree('./chroma_langchain_db')
#             #os.remove("chat_history.json")
#             break
#         if not question:
#             continue

#         try:
#             retrieved_docs = retriever.invoke(question)
#             context = format_documents(retrieved_docs)

#             if not context:
#                 print("\nI couldn't find relevant information in the documents for your question.")
#                 continue

#             print("\nThinking...")
#             response = chain.invoke(
#                 {"context": context, "question": question},
#                 config={"configurable": {"session_id": session_id}}
#             )

#             print("\nAnswer:")
#             print(response)

#         except Exception as exc:
#             print(f"\nAn error occurred: {exc}")


# if __name__ == "__main__":
#     main()


# script.py (Corrected, Stateless Version)
import os
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --- Configuration ---
MODEL_NAME = "llama3.2-vision:11b"
MAX_CONTEXT_CHARS = 8_000

# --- Model ---
model = OllamaLLM(model=MODEL_NAME)

# --- Prompt Template ---
# The history placeholder will be filled by server.py
# script.py

# script.py
# script.py

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI assistant. Your task is to answer user questions based *only* on the provided context.\n"
        "Structure your entire response using the following Markdown format:\n\n"
        "### Assistant Response\n"
        "[Your concise, conversational answer to the user's question goes here.]\n\n"
        "---\n\n" # Use a horizontal rule as a separator
        "### Key Takeaways\n"
        "[A bulleted list of the most important points and sources from the context goes here.]\n\n"
        "**Important Rules**:\n"
        "- If the context does not contain the answer, your entire response should only be 'I could not find relevant information in the documents for that question.'\n"
        "- For simple greetings, provide only the 'Assistant Response' part without the 'Key Takeaways' or the separator.\n"
        "CONTEXT:\n---\n{context}\n---"
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])
# prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         "You are an AI assistant. Your task is to answer user questions "
#         "based *only* on the provided context. Always structure your response in two parts separated by '---' exactly as follows:\n\n"
#         "PART 1: A concise, direct, and friendly answer to the user's question. This part should be conversational and not a list.\n\n"
#         "--- \n\n" # This separator is crucial for the frontend
#         "PART 2: Start this part with the Markdown heading '### Key Takeaways'. Provide a bulleted list of the most important points from the context. (Cite the source file and page/row if available.)\n\n"
#         "**Important Rules**:\n"
#         "- If the context does not contain the answer, your entire response should only be 'I could not find relevant information in the documents for that question.'\n"
#         "- If the user greets you, your entire response should only be a friendly greeting back, introducing yourself briefly. Do not use the two-part structure for greetings.\n\n"
#         "CONTEXT:\n---\n{context}\n---"
#     ),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{question}"),
# ])
# prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         "You are an AI assistant. Your task is to answer user questions "
#         "based *only* on the provided context.\n"
#         "If the context does not contain the answer, state that clearly.\n"
#         "If someone greets you always greet them properly saying about you\n"
#         "**Important** don't give Key Takeways and Assitant Response Only for greetings\n"
#         # --- CHANGED LINES START HERE ---
#         "Structure your answer in two parts using Markdown headings:\n"
#         "### Key Takeaways\n"
#         "A bulleted list of the most important points from the context. "
#         "(Cite the source file and page/row if available.)\n\n"
#         "### Assistant Response\n"
#         "A detailed answer synthesized from the key takeaways.\n\n"
#         # --- CHANGED LINES END HERE ---
#         "CONTEXT:\n---\n{context}\n---"
#     ),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("human", "{question}"),
# ])
# prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         "You are an AI assistant. Your task is to answer user questions "
#         "based *only* on the provided context.\n"
#         "If the context does not contain the answer, state that clearly.\n"
#         "If someone greets you always greet them properly saying about you\n"
#         "**Important** don't give Key Takeways and Assitant Response Only for greetings\n"
#         "Structure your answer in two parts:\n"
#         "1. **Key Takeaways**: A bulleted list of the most important points "
#         "from the context. (Cite the source file and page/row if available.)\n"
#         "2. **Assistant Response**: A detailed answer synthesized from the "
#         "key takeaways.\n\n"
#         "CONTEXT:\n---\n{context}\n---"
#     ),
#     MessagesPlaceholder(variable_name="chat_history"), # Correctly named to match server.py
#     ("human", "{question}"),
# ])

# --- Helper to format documents ---
def format_documents(docs: list) -> str:
    """
    Turn Document objects into a single string, truncate if too long.
    """
    formatted = []
    total_chars = 0
    for idx, doc in enumerate(docs, start=1):
        source_info = f"Source: {os.path.basename(doc.metadata.get('source', 'N/A'))}"
        if 'page' in doc.metadata:
            source_info += f", Page {doc.metadata['page']}"
        if 'row' in doc.metadata:
            source_info += f", Row {doc.metadata['row']}"
        entry = f"[{idx}] {doc.page_content} ({source_info})"
        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            break
        formatted.append(entry)
        total_chars += len(entry)
    return "\n\n".join(formatted)

# --- Core RAG Chain (Stateless) ---
# This is the 'original_chain' that server.py will import.
# It is NOT wrapped with RunnableWithMessageHistory.
original_chain = (
    RunnablePassthrough.assign(
        # The 'context' is passed in from server.py, so we just pass it through
    )
    | prompt
    | model
    | StrOutputParser()
)