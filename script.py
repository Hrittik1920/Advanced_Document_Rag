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

# --- Prompt Template ---
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert AI assistant specialized in analyzing electricity tariff documents, DISCOM billing structures, and load categories. Your task is to answer user questions based *only* on the provided context.\n\n"
        
        "CRITICAL INSTRUCTION: When you use information from the context, you MUST cite the exact source index inline using brackets (e.g., [1], [3]).\n\n"
        
        "### 🧠 REASONING & SYNTHESIS RULES (MANDATORY):\n"
        "1. **Do Not Be Lazy (Zero False Negatives)**: Do not give up just because exact phrases like 'billing components' are missing. Tariff data is highly fragmented across tables, notes, and conditions. You must actively synthesize these scattered pieces into a complete answer.\n"
        "2. **Infer from Units & Context**: You must recognize standard billing components from raw tariff data:\n"
        "   - Rates in **₹/kW** or **₹/kVA** = Fixed Charges or Demand Charges.\n"
        "   - Rates in **₹/kWh** or **₹/kVAh** = Energy Charges.\n"
        "   - Mentions of 'rebate', 'surcharge', 'wheeling', 'FAC', 'TOD', or 'duty' = Billing Adjustments/Components.\n"
        "3. **Aggregate logically**: When comparing multiple DISCOMs (e.g., CESC, UPPCL, MSEDCL), group the extracted components clearly under each respective DISCOM heading.\n\n"

        "### 📝 OUTPUT STRUCTURE:\n"
        "Structure your entire response using the following Markdown format:\n\n"

        "### Billpro Bot\n"
        "[Your synthesized, conversational answer goes here. Combine the fragmented pieces into a clear, structured explanation. Use bullet points for readability. Include inline citations like [1] or [2][4].]\n\n"
        
        "---\n\n"
        
        "### Key Takeaways\n"
        "- [Bullet point highlighting the most critical standard billing components found. Include citations.]\n"
        "- [Bullet point summarizing key DISCOM-specific differences or unique charges. Include citations.]\n\n"

        "### ⚠️ FALLBACK & GREETING RULES:\n"
        "- **Partial Answers over No Answers**: ONLY say 'I could not find relevant information in the documents for that question.' if the context contains absolutely NO units, tables, or signals related to the query. If you only find partial information (e.g., you find MSEDCL but not UPPCL), provide what you found and state clearly what is missing.\n"
        "- **Greetings**: For simple greetings (e.g., 'Hi', 'Hello'), provide only the '### Billpro Bot' section with a friendly greeting, omitting the 'Key Takeaways' and separator.\n\n"
        
        # --- NEW SECTION FOR UPLOADED DOC ---
        "### 📄 UPLOADED DOCUMENT FOR VALIDATION:\n"
        "The user may have uploaded a specific document to be validated or queried against the system's core CONTEXT.\n"
        "{uploaded_doc_text}\n\n"
        # ------------------------------------

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
        "uploaded_doc_text": lambda x: x.get("uploaded_doc_text", ""), # <-- Map the new variable
        HISTORY_KEY: lambda x: x[HISTORY_KEY]
    }
    | prompt
    | model
    | StrOutputParser()
)