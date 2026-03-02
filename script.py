# script.py (Corrected, Stateless Version)
import os
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import settings
from llm_clients import query_ollama

# --- Configuration ---
MODEL_NAME = settings.LLM_MODEL_NAME
MAX_CONTEXT_CHARS = 8_000
HISTORY_KEY = settings.HISTORY_DIR
# --- Model ---
model =OllamaLLM(model=MODEL_NAME, streaming=True)
# --- Condense Prompt (rewrites follow-up questions to be standalone) ---
condense_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the chat history and the latest user question, rewrite the question "
        "as a fully self-contained, standalone question. "
        "Do NOT answer it. Only rewrite it. If it's already standalone, return it as-is."
    ),
    MessagesPlaceholder(variable_name=settings.HISTORY_DIR),
    ("human", "{question}"),
])

condense_chain = condense_prompt | model | StrOutputParser()

# --- Prompt Template ---

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
    MessagesPlaceholder(variable_name=settings.HISTORY_DIR),
    ("human", "{question}"),
])

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
    {
        "context": lambda x: x["context"],
        "question": lambda x: x["question"],
        HISTORY_KEY: lambda x: x[HISTORY_KEY]
    }
    | prompt
    | model
    | StrOutputParser()
)