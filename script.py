# script.py 
import os
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from config import settings
from llm_clients import query_ollama
import re
import json
from langchain_core.exceptions import OutputParserException
from typing import Tuple

# --- Configuration ---
MODEL_NAME = settings.LLM_MODEL_NAME
OLLAMA_BASE_URL = settings.LLM_ENDPOINT
MAX_CONTEXT_CHARS = 30_000
HISTORY_KEY = settings.HISTORY_DIR
# --- Model ---
model = OllamaLLM(model=MODEL_NAME, base_url=OLLAMA_BASE_URL, streaming=True)
# --- Condense Prompt (rewrites follow-up questions to be ) ---
_condense_model = OllamaLLM(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    streaming=False,
    temperature=0.2,      
)

_math_coding_model = OllamaLLM(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    streaming=False,    # Prevents Ollama tool-call parse errors on arithmetic output
    temperature=0.1,
)

_router_model = OllamaLLM(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    streaming=False,
    temperature=0,
)
_hyde_model = OllamaLLM(
    model=MODEL_NAME,
    base_url=OLLAMA_BASE_URL,
    streaming=False,
    temperature=0.3,
)
condense_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a query rewriting assistant for a RAG system. Output ONLY valid JSON.

Given the chat history and the user question, output this exact structure:
{{
  "retrieval_queries": ["list of standalone search queries"],
  "generation_question": "single natural question combining full intent",
  "math_intent": true
}}

Rules:
1. Rewrite the question so each retrieval query is fully self-contained (no pronouns like 'this', 'it').
2. Split into multiple retrieval_queries if the question covers multiple entities/concepts.
3. Include specific tariff classes (e.g., 'LV2', 'HT'), voltage levels (e.g., '33kV'), and consumer types (e.g., 'Industrial') when present.
4. generation_question must be natural and human-readable.
5. math_intent=true only when a calculation, formula, or numerical derivation is explicitly required.
6. If the question is a greeting or trivially simple, return the original question as-is with math_intent=false.
7. Do NOT hallucinate specificity. If vague, keep it vague.

Document context (use only to improve query specificity):
{unique_docs_map}
8 **Remember**: While writing multi-query for retrieval for document based matching always include the listed classes in it 
  --For example
  User Question: "Verify the bill?
  Document context : [uploaded document text]
  Output:
{{
  "retrieval_queries": [
    "verify calculation of energy bill for company_name", 
    "fixed price of the bill and it info from document",
    "energy charges and it info from document",
    "taxes and its info from document" #all the relavent things from the document
  ],
  "generation_question": "Can you verify the fixed charge calculation for this bill `billing info`?",
  "math_intent": true
}}"""
    ),
    MessagesPlaceholder(variable_name=settings.HISTORY_DIR),
    ("human", "{question}"),
])
condense_chain = condense_prompt | _condense_model | JsonOutputParser()


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

# --- Router Logic (For targeting uploaded files) ---
router_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a document classification API.\n"
     "Match the uploaded document snippet to the correct file(s) from AVAILABLE FILES.\n\n"
     "RULES:\n"
     "1. Exact regional match preferred (e.g., 'Poorv'=East, 'Madhya'=Central, 'Dakshin'=South, 'Paschim'=West).\n"
     "2. If State is clear but region is ambiguous, return ALL files from that state.\n"
     "3. Respond ONLY with raw JSON. No markdown.\n\n"
     "AVAILABLE FILES: {available_files}\n\n"
     "Format:\n"
     "{{\n  \"target_files\": [\"filename1.pdf\", \"filename2.pdf\"]\n}}\n\n"
     "If no match: {{\"target_files\": []}}."),
    ("human", "User Query: {question}\n\nUploaded Document Snippet:\n{uploaded_text}")
])


# Build the complete chain
router_chain = router_prompt | _router_model | JsonOutputParser()

math_coding_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert Python programmer. Solve the user's billing or math problem.\n\n"
        "STRICT OUTPUT RULES:\n"
        "1. Your ENTIRE response must be a single ```python\\n...\\n``` code block.\n"
        "2. Do NOT write any text, explanation, or arithmetic OUTSIDE the code block.\n"
        "3. The script MUST call print() with the final answer as its last action.\n"
        "4. Available libraries: math, numpy (import as np), pandas (import as pd).\n\n"
        "CORRECT example:\n"
        "```python\n"
        "fixed_charge = 500.0\n"
        "energy_charge = 3200.0\n"
        "total = fixed_charge + energy_charge\n"
        "print(f'Total Bill: Rs. {{total:.2f}}')\n"
        "```\n\n"
        "WRONG (never do this — it will crash the system):\n"
        "500.0 + 3200.0 = 3700.0\n\n"
        "### Source data (extracted from uploaded bill):\n"
        "{data}\n\n"
        "### Additional context from knowledge base:\n"
        "---\n{context}\n---"
    ),
    ("human", "{question}")
])

math_answering_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are Billpro Bot. You just ran a Python script to solve the user's billing problem.\n"
        "Use ONLY the context and execution result below. Do NOT use outside knowledge.\n\n"
        "CONTEXT:\n--\n{context}\n--\n\n"
        "Execution Output:\n---\n{result}\n---\n\n"
        "Validation Status:\n---\n{validation}\n---\n\n"
        "Provide a clear, conversational answer. Show key figures in a readable format."
    ),
    ("human", "{question}")
])

math_coding_chain = math_coding_prompt | _math_coding_model | StrOutputParser()
math_answering_chain = math_answering_prompt | model | StrOutputParser()

def extract_python_code(text: str) -> str:
    """Extracts python code from markdown code blocks."""
    match = re.search(r'```python\n(.*?)\n```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.replace("```python", "").replace("```", "").strip()


_hyde_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an official drafting For a Bill or a document provided to you
Your task is to generate a short, hypothetical excerpt from the 'Retail Supply Tariff Orde'.

Rules for Generation:
1. Format: Write in the style of an official 'Tariff Schedule' or 'Commission's Analysis' section.
2. Terminology: Use 'Petitioners', 'Distribution Licensees', 'LV-2.2', 'Aggregate Revenue Requirement (ARR)', and 'Terms and Conditions'.
3. Detail: Distinguish between 'Urban' and 'Rural' areas and include specific rates for categories like 'Non-Domestic' or 'Telecom Towers' if implied.
4. Structure: Start directly with a section number (e.g., 1.15 or Table X) and use formal regulatory language.
5. Units: Use standard MPERC units: Rs./kW/month for Fixed Charges and paise/kWh or Rs./unit for Energy Charges.

Example Style: 
'The Commission determines the Fixed Charges for Category LV-2.2 (Non-Domestic) in rural areas at Rs. 131 per kW per month. The Energy Charge for consumption exceeding 300 units shall be billed at 775 paise per unit, subject to FPPAS adjustments as per Regulation 5.'

No preamble. Under 100 words."""
    ),
    ("human", "Generate a hypothetical tariff order passage for this question: {question}"),
])

# Separate non-streaming model for HYDE — we need the full text
# synchronously before we can use it as a retrieval query.

_hyde_chain = _hyde_prompt | _hyde_model | StrOutputParser()

math_validation_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a strict billing auditor for Indian electricity DISCOMs.
Your job is to verify a Python-computed bill result against the official tariff context.

## YOUR TASK
Cross-reference EVERY numeric component in the execution result against the tariff rates in CONTEXT.
Flag any line item that:
- Uses a wrong rate (e.g., energy charge ₹7.50 but context says ₹6.50 for this load factor)
- Is missing from the computed result but present in the bill
- Is present in the computed result but not justified by context
- Has a correct rate but wrong base (e.g., wrong contract demand or wrong kWh)
- Has arithmetic that doesn't add up (e.g., components don't sum to total)

## BILLING COMPONENTS TO CHECK
For each component found in the execution result, verify:
1. Fixed Charge / Demand Charge — rate (Rs/kW or Rs/kVA) × contracted demand
2. Energy Charge — rate (paise/unit or Rs/kWh) × units consumed
3. FPPAS / FAC adjustment — should match the per-unit rate in context
4. Electricity Duty — typically a % of base amount or fixed slab
5. Surcharges / Rebates — TOD, wheeling, green energy, etc.
6. Any tax or cess items

## OUTPUT FORMAT
Respond ONLY with a valid JSON object. No preamble, no markdown.
{{
  "overall_valid": true,
  "confidence": 0.92,
  "arithmetic_check": {{
    "components_sum": 7917.56,
    "reported_total": 7911.26,
    "difference": 6.30,
    "verdict": "MINOR_DISCREPANCY"
  }},
  "line_item_checks": [
    {{
      "component": "Energy Charge",
      "computed_value": 6304.26,
      "expected_rate_from_context": "Rs 5.00/kWh for LT-2 domestic (context [3])",
      "units_used": 1260.85,
      "verdict": "VERIFIED",
      "note": ""
    }},
    {{
      "component": "Fixed Charge",
      "computed_value": 992.24,
      "expected_rate_from_context": "Rs 100/kW/month for sanctioned load (context [1])",
      "units_used": "9.92 kW",
      "verdict": "PLAUSIBLE",
      "note": "Rate matches context but sanctioned load not confirmed in uploaded bill"
    }},
    {{
      "component": "Electricity Duty",
      "computed_value": 910.00,
      "expected_rate_from_context": "Not found in context",
      "units_used": null,
      "verdict": "UNVERIFIED",
      "note": "Context has no ED rate for this tariff category. Cannot confirm."
    }}
  ],
  "missing_components": ["TOD surcharge not computed but context mentions it for HT consumers"],
  "extra_components": [],
  "summary": "Energy and fixed charges broadly align with tariff context. Electricity Duty cannot be verified — rate not present in retrieved chunks. Arithmetic difference of Rs 6.30 is within rounding tolerance.",
  "recommendation": "ACCEPT_WITH_CAUTION"
}}

Verdict values: VERIFIED | PLAUSIBLE | WRONG_RATE | WRONG_BASE | UNVERIFIED | MISSING
Recommendation values: ACCEPT | ACCEPT_WITH_CAUTION | REJECT | INSUFFICIENT_CONTEXT

CONTEXT (tariff documents):
---
{context}
---
""",
    ),
    (
        "human",
        "Question: {question}\n\nExecution Result (Python output):\n{result}\n\nUploaded Bill Snippet:\n{uploaded_doc_text}"
    )
])

math_validation_chain = math_validation_prompt | _math_coding_model | JsonOutputParser()
def generate_hyde_query(question: str) -> str:
    """
    Generates a hypothetical document passage for a given question.

    The passage is intentionally written to sound like the *answer*
    would appear in an actual tariff document. Embedding this passage
    instead of the raw question dramatically improves retrieval recall
    for sparse, jargon-heavy queries (e.g. "what is the TOD rebate
    for HT Industrial consumers in MSEDCL?").

    Args:
        question: The user's standalone question (post-condense).

    Returns:
        A short hypothetical passage string ready to be passed to
        retriever.ainvoke().  Falls back to the original question
        if generation fails so the caller never gets None.
    """
    try:
        hyde_passage = _hyde_chain.invoke({"question": question}).strip()
        return hyde_passage or question
    except Exception as e:
        # Graceful degradation — log in server.py, fall back here
        print(f"[HYDE] generate_hyde_query failed: {e}")
        return question


# ─────────────────────────────────────────────────────────────
# FORMAT DOCUMENTS
# Turns retrieved Document objects into:
#   - A single context string for the LLM prompt
#   - A citation list for the frontend
# ─────────────────────────────────────────────────────────────

def format_documents(docs: list) -> Tuple[str, list]:
    """
    Format retrieved documents into an LLM-ready context string
    and a structured citation list for the frontend.

    Changes vs original:
    - Score-aware ordering: docs with a 'score' in metadata are
      sorted descending before formatting, so the highest-quality
      chunks get lower citation indices (and therefore appear first
      in the LLM's context window, where attention is strongest).
    - HYDE-tagged docs are placed after scored originals so the
      LLM sees the most reliable sources first.
    - Returns Tuple[str, list] (was previously untyped).

    Args:
        docs: List of LangChain Document objects.  Each doc may
              optionally carry metadata keys: source, page, rows,
              score, source_type ("hyde" | "original").

    Returns:
        (context_string, citation_list)
    """
    # ── Sort by score (desc), with HYDE docs ranked lower ──────
    def sort_key(doc):
        score = doc.metadata.get("score", 0.0)
        is_hyde = 1 if doc.metadata.get("source_type") == "hyde" else 0
        # Primary: prefer non-HYDE; secondary: prefer higher score
        return (is_hyde, -score)

    sorted_docs = sorted(docs, key=sort_key)

    formatted = []
    citation = []
    total_chars = 0

    for idx, doc in enumerate(sorted_docs, start=1):
        raw_source = doc.metadata.get("source", "")
        file_name = os.path.basename(raw_source) if raw_source else "N/A"
        page_num = doc.metadata.get("page", 0)

        location_parts = [f"Source: {file_name}"]
        if "page" in doc.metadata:
            location_parts.append(f"Page {page_num}")
        if "rows" in doc.metadata:
            location_parts.append(f"Row {doc.metadata['rows']}")

        location_str = ", ".join(location_parts)
        snippet = doc.page_content.strip()

        # Derive a short topic label for the citation card
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
            "score": doc.metadata.get("score"),
            "source_type": doc.metadata.get("source_type", "original"),
        })

    return "\n\n".join(formatted), citation


original_chain = (
    {
        "context": lambda x: x["context"],
        "question": lambda x: x["question"],
        "uploaded_doc_text": lambda x: x.get("uploaded_doc_text", ""), # <-- Map the new variable
        HISTORY_KEY: lambda x: x[HISTORY_KEY],
    }
    | prompt
    | model
    | StrOutputParser()
)