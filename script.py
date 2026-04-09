# script.py 
import os
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from config import settings
from llm_clients import query_ollama

# --- Configuration ---
MODEL_NAME = settings.LLM_MODEL_NAME
MAX_CONTEXT_CHARS = 30_000
HISTORY_KEY = settings.HISTORY_DIR
# --- Model ---
model =OllamaLLM(model=MODEL_NAME, streaming=True)
# --- Condense Prompt (rewrites follow-up questions to be ) ---
condense_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Given the chat history and the latest user question, your task is to rewrite the question "
        "as fully self-contained, standalone questions.\n"
        "If the user asks about multiple distinct entities or concepts, break the question down into a list of separate questions.\n"
        "You MUST return ONLY a valid JSON object with two keys:\n"
        "1. 'retrieval_queries': A list of strings to be used for database search.\n"
        "2. 'generation_question': A single, comprehensive string combining the intent of the rewritten questions. This will be used by the final answering model and saved to the chat history.\n"
        "Example:\n"
        "{{\n" 
        "  \"retrieval_queries\": [\"What is the commercial tariff for company A?\", \"What is the commercial tariff for company B?\"],\n"
        "  \"generation_question\": \"What are the commercial tariffs for company A and company B?\"\n"
        "}}"
    ),
    MessagesPlaceholder(variable_name=settings.HISTORY_DIR),
    ("human", "{question}"),
])

condense_chain = condense_prompt | model | JsonOutputParser()

# --- Prompt Template ---

# script.py

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an AI assistant. Your task is to answer user questions based *only* on the provided context.\n"
        "CRITICAL INSTRUCTION: When you use information from the context to form your answer, you MUST cite the exact source index inline using brackets (e.g., [1], [3]).\n\n"
        "Structure your entire response using the following Markdown format:\n\n"
        "### Billpro Bot\n"
         "[Your concise, conversational answer to the user's question goes here. Include inline citations like [1] where applicable.]\n\n"
         "---\n\n"
        "### Key Takeaways\n"
        "[A bulleted list of the most important points from the context goes here. Include citations here too.]\n\n"
        "**Important Rules**:\n"
        "- If the context does not contain the answer, your entire response should only be 'I could not find relevant information in the documents for that question.'\n"
        "- For simple greetings, provide only the 'Suraksha's Reply' part without the 'Key Takeaways' or the separator.\n"
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
        if 'rows' in doc.metadata:
            location_parts.append(f"Row {doc.metadata['rows']}")
            
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
            "rows": doc.metadata.get("rows"),
            "topic": topic,
        })
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