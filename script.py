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
        "### Suraksha's Reply\n"
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
    and to provide all the citation list for the frontend
    """
    formatted = []
    citation = []
    total_chars = 0
    for idx, doc in enumerate(docs, start=1):
        raw_source = doc.metadata.get('source', '')
        file_name = os.path.basename(raw_source) if raw_source else 'N/A'
        source_info = f"Source: {file_name}"
        
        location_parts = [source_info]
        # Default page to 0 if not found, since your endpoint expects an integer
        page_num = doc.metadata.get('page', 0) 
        if 'page' in doc.metadata:
            location_parts.append(f"Page {page_num}")
        if 'row' in doc.metadata:
            location_parts.append(f"Row {doc.metadata['row']}")
            
        location_str = ", ".join(location_parts)
        snippet = doc.page_content.strip()
        topic = (snippet[:120] + "…") if len(snippet) > 120 else snippet
        first_sentence_end = snippet.find(". ")
        if 0 < first_sentence_end < 100:
            topic = snippet[: first_sentence_end + 1]

        entry = f"[{idx}] {doc.page_content} ({location_str})"
        if total_chars + len(entry) > MAX_CONTEXT_CHARS:
            break
        formatted.append(entry)
        total_chars += len(entry)

        citation.append({
            "id": idx,
            "display_name": file_name,
            "file_path": raw_source, 
            "page": page_num,
            "row": doc.metadata.get("row"),
            "topic": topic,
        })
        formatted.append(entry)
        total_chars += len(entry)
    return "\n\n".join(formatted), citation

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