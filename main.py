#main.py
"""Orchestration of LLM generation and agent execution using LangGraph."""
import os
import json
import asyncio
from typing import TypedDict, List, Any, Dict
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from script import condense_chain, unified_chain, math_coding_chain, math_answering_chain, format_documents, prompt, model, HISTORY_KEY, extract_python_code, generate_hyde_query, math_validation_chain
from retriever import retriever
from logger_utils import log_timing, log_debug, timed
import time 
from sandbox import run_code_in_sandbox 
import re
# --- State Definition ---
class AgentState(TypedDict):
    question: str
    chat_histories: List[Any]
    uploaded_doc_text: str
    available_files: List[str]
    target_files: List[str]
    generation_question: str
    retrieval_queries: List[str]
    math_intent: bool
    context: str
    citations: List[Dict]
    generated_code: str
    execution_result: str
    validation_result: Dict
    validation_passed: bool
    
    # Final Output
    final_response: str

# --- Graph Nodes ---
class AgentNodes:
    def __init__(self, condense_chain, unified_chain, math_coding_chain, math_answering_chain, retriever, prompt, model):
        self.condense_chain = condense_chain
        self.unified_chain = unified_chain
        self.math_coding_chain = math_coding_chain
        self.math_answering_chain = math_answering_chain
        self.retriever = retriever
        self.prompt = prompt
        self.model = model

    async def classify_and_retrieve(self, state: AgentState) -> dict:
        """Runs the math intent classifier and retrieves documents."""
        node_start=time.perf_counter()
        user_query = state["question"]
        chat_histories = state.get("chat_histories", [])
        uploaded_doc_text = state.get("uploaded_doc_text", "")
        available_files = state.get("available_files", [])
        ui_target_files=state.get("target_files",[])
        log_debug(f"Starting unified prompt: {user_query}")
        # 1. Math Intent Classifier (Condense Chain)
        step_start = time.perf_counter()
        
        condense_task = await self.condense_chain.ainvoke({
            "question": user_query,     
            HISTORY_KEY: chat_histories,
        })
        unified_result_raw = {}
        condensed_result_raw ={}
        if uploaded_doc_text:
            try:
                unified_result_raw = await self.unified_chain.ainvoke({
                "question": user_query,
                HISTORY_KEY: chat_histories,
                "uploaded_doc_text": uploaded_doc_text[:2000], 
                "available_files": str(available_files)
                })
            except Exception as e:
                unified_result_raw = e
            log_timing(f"[LATENCY] unified refine with docs: {(time.perf_counter() - step_start) * 1000:.2f} ms")
        else:
            log_debug("[ROUTER] No document uploaded. Skipping router chain.")
            # Only run condense and set a default empty list for router            
            try:
                condensed_result_raw = await condense_task
            except Exception as e:
                condensed_result_raw = e

        log_timing(f"[LATENCY] Pre-processing: {(time.perf_counter() - step_start) * 1000:.2f} ms")

        if isinstance(unified_result_raw, Exception):
            log_debug(f"[WARN] Router chain failed: {unified_result_raw}. Defaulting to no file filter.")
            llm_target_files = []
        else:
            llm_target_files = unified_result_raw.get("target_files", [])
        #combining the router llm and ui initiated document target
        # ── Safely unpack condense result ────────────────────────────────────
        if isinstance(condensed_result_raw, Exception):
            log_debug(f"[WARN] Condense chain failed: {condensed_result_raw}. Using original query as fallback.")
            retrieval_queries = [user_query]
            generation_question = user_query
            math_intent = False
            log_debug(f"Condensed Result: {condensed_result_raw}")
        else:
            retrieval_queries = condensed_result_raw.get("retrieval_queries", [user_query])
            generation_question = condensed_result_raw.get("generation_question", user_query)
            math_intent = condensed_result_raw.get("math_intent", False)
            log_debug(f"Condensed Result: {condensed_result_raw}")
        
        if isinstance(unified_result_raw, Exception):
            log_debug(f"[WARN] unified process failed:{unified_result_raw}. using original details as fallback")
            query_list=uploaded_doc_text[:500].split("\n")
            doc_snippet = "\n".join(line for line in query_list[:5] if line.strip())
            retrieval_queries = [user_query, doc_snippet] if doc_snippet else [user_query]
            generation_question = user_query
            math_intent = True
            llm_target_files = []
            log_debug(f"Condensed Result: {unified_result_raw}")
        else:
            query_list=uploaded_doc_text[:500].split("\n")
            doc_snippet = "\n".join(line for line in query_list[:5] if line.strip())
            fallback_queries = [user_query, doc_snippet] if doc_snippet else [user_query]
            retrieval_queries = unified_result_raw.get('retrieval_queries',fallback_queries)
            generation_question = unified_result_raw.get('generation_question',user_query)
            math_intent = unified_result_raw.get("math_intent", False)
            llm_target_files=unified_result_raw.get('target_files',[])
        combined_target_files=list(set(llm_target_files+ui_target_files))
        log_debug(f"Router matched: {combined_target_files} | Math Intent: {math_intent}")
        log_debug(f"Condensed Result: {unified_result_raw}")

        # ── Step 2: Retrieve ONCE ────────────────────────────────────────────
        step_start = time.perf_counter()
        retrieval_tasks = [
            self.retriever.ainvoke(q, target_sources=combined_target_files if combined_target_files else None) 
            for q in retrieval_queries
        ]
        results = await asyncio.gather(*retrieval_tasks)
        log_timing(f"[LATENCY] Vector Retrieval: {(time.perf_counter() - step_start) * 1000:.2f} ms")
        step_start = time.perf_counter()
        unique_docs_map = {}
        for doc_list in results:
            for doc in doc_list:
                if doc.page_content not in unique_docs_map:
                    unique_docs_map[doc.page_content] = doc
        retrieved_docs = list(unique_docs_map.values())

        # ── Step 4: HYDE fallback ────────────────────────────────────────────
        if len(retrieved_docs) < 2 and len(generation_question.split()) < 5:
            log_debug("[HYDE] Weak retrieval detected, retrying with HYDE...")
            try:
                hyde_query = generate_hyde_query(generation_question)
                log_debug(f"[HYDE] Generated passage: {hyde_query}...")

                hyde_results = await self.retriever.ainvoke(
                    hyde_query,
                    target_sources=combined_target_files if combined_target_files else None
                )

                log_debug(f"[HYDE] Retrieval returned {len(hyde_results)} docs")

                # merge with tagging
                hyde_doc_count = 0
                for doc in hyde_results:
                    if doc.page_content not in unique_docs_map:
                        doc.metadata["source_type"] = "hyde"
                        unique_docs_map[doc.page_content] = doc
                        hyde_doc_count += 1

                retrieved_docs = list(unique_docs_map.values())
                log_debug(f"[HYDE] Added {hyde_doc_count} HYDE docs. Total now: {len(retrieved_docs)}")

                if len(retrieved_docs) < 5:
                    log_debug("[HYDE] Still weak after retry - proceeding anyway")

            except Exception as e:
                log_debug(f"[HYDE ERROR] {type(e).__name__}: {e}")
                log_debug("Continuing with original retrieval results")

        retrieved_docs = retrieved_docs[:20]
        log_debug(f"Unique Retrieved Docs: {len(retrieved_docs)}")  
                  
        context, citations = format_documents(retrieved_docs)
        if not context:
            context = "No relevant documents found in the knowledge base."

        if uploaded_doc_text:
            with open("upload_content.txt", "w", encoding="utf-8") as f:
                f.write(uploaded_doc_text)

        with open("chunk.txt", "w", encoding="utf-8") as f:
            f.write(context)
        log_timing(f"[LATENCY] Dedup & Format: {(time.perf_counter() - step_start) * 1000:.2f} ms")
        
        log_timing(f"[TOTAL] classify_and_retrieve node took {(time.perf_counter() - node_start) * 1000:.2f} ms")
        return {
            "generation_question": generation_question,
            "retrieval_queries": retrieval_queries,
            "math_intent": math_intent,
            "context": context,
            "citations": citations,
        }

    async def text_path(self, state: AgentState) -> dict:
        """Standard LLM generation for non-math queries."""
        # Using the standard prompt and model
        chain = self.prompt | self.model | StrOutputParser()
        response = await chain.ainvoke({
            "context": state["context"],
            "question": state["generation_question"],
            "uploaded_doc_text": state.get("uploaded_doc_text", ""), # <-- ADDED THIS LINE
            HISTORY_KEY: state.get("chat_histories", [])
        })
        log_debug(f"Text Path Response: {response}")
        return {"final_response": response}

    async def math_generate(self, state: AgentState) -> dict:
        """Python code generator: LLM writes typed SymPy + arithmetic."""
        try:
            response = await self.math_coding_chain.ainvoke({
                "data": state.get("uploaded_doc_text",""),
                "context": state["context"],
                "question": state["generation_question"]
            })
            code = extract_python_code(response if isinstance(response, str) else response.get("code", ""))
            log_debug(f"[MATH] Generated code:\n{code}")
        except Exception as e:
            if "error parsing tool call" in str(e):
                # Extract the 'raw' part from the error message using regex
                raw_math = re.search(r"raw='(.*?)'", str(e)).group(1)
                # Manually construct a valid tool call or fallback response
                print(f"Rescued raw math: {raw_math}")
        return {"generated_code": code}

    async def math_execute(self, state: AgentState) -> dict:
        """Sandbox executor: RestrictedPython + timeout guard."""
        code = state["generated_code"]
        # FIX: Utilize the secure docker sandbox logic
        result = await run_code_in_sandbox(code)
        log_debug(f"[MATH] Execution result: {result}")
        return {"execution_result": str(result)}

    async def math_validate(self, state: AgentState) -> dict:
        """
        Cross-references every computed billing component against
        the tariff context. Returns structured discrepancies so
        synthesize_response can explain exactly what's wrong.
        """
        # Guard: if execution itself failed, skip deep validation
        result_text = state.get("execution_result", "")
        if result_text.startswith("Sandbox Error") or not result_text.strip():
            log_debug("[MATH VALIDATE] Skipping — no valid execution result.")
            return {
                "validation_result": {
                    "overall_valid": False,
                    "confidence": 0.0,
                    "summary": "Execution failed — no result to validate.",
                    "recommendation": "REJECT",
                    "line_item_checks": [],
                    "missing_components": [],
                    "extra_components": [],
                    "arithmetic_check": {}
                },
                "validation_passed": False
            }

        try:
            validation_result = await math_validation_chain.ainvoke({
                "question":          state["generation_question"],
                "context":           state["context"],
                "result":            result_text,
                "uploaded_doc_text": state.get("uploaded_doc_text", "")[:2000]
            })
            log_debug(f"[MATH VALIDATE] Result: {json.dumps(validation_result, indent=2)}")
            
            # Routing flag: only hard-reject if explicitly REJECT
            validation_passed = validation_result.get("recommendation", "") != "REJECT"

        except Exception as e:
            log_debug(f"[MATH VALIDATE] Chain failed: {e}. Defaulting to ACCEPT_WITH_CAUTION.")
            validation_result = {
                "overall_valid": True,
                "confidence": 0.5,
                "summary": f"Validation chain failed ({e}). Result unverified.",
                "recommendation": "ACCEPT_WITH_CAUTION",
                "line_item_checks": [],
                "missing_components": [],
                "extra_components": [],
                "arithmetic_check": {}
            }
            validation_passed = True

        return {
            "validation_result": validation_result,
            "validation_passed": validation_passed
        }

    async def synthesize_response(self, state: AgentState) -> dict:
        """
        Merges execution result + structured validation into a 
        readable response. Surfaces specific discrepancies to user.
        """
        v = state.get("validation_result", {})
        
        # Build a human-readable validation summary for the answering LLM
        line_items = v.get("line_item_checks", [])
        
        verdict_lines = []
        for item in line_items:
            icon = {
                "VERIFIED":      "✅",
                "PLAUSIBLE":     "🟡",
                "WRONG_RATE":    "❌",
                "WRONG_BASE":    "❌",
                "UNVERIFIED":    "⚠️",
                "MISSING":       "🔴",
            }.get(item.get("verdict", ""), "❓")
            
            note = f" — {item['note']}" if item.get("note") else ""
            verdict_lines.append(
                f"{icon} **{item['component']}**: "
                f"₹{item.get('computed_value', '?')} "
                f"[{item.get('verdict', '?')}]{note}"
            )

        arith = v.get("arithmetic_check", {})
        arith_line = ""
        if arith:
            diff = arith.get("difference", 0)
            arith_line = (
                f"\n**Arithmetic check**: "
                f"Components sum ₹{arith.get('components_sum', '?')} vs "
                f"reported ₹{arith.get('reported_total', '?')} "
                f"(diff: ₹{diff}) — {arith.get('verdict', '')}"
            )

        missing = v.get("missing_components", [])
        missing_line = ""
        if missing:
            missing_line = "\n**Potentially missing**: " + "; ".join(missing)

        formatted_validation = "\n".join(verdict_lines) + arith_line + missing_line
        recommendation = v.get("recommendation", "ACCEPT_WITH_CAUTION")
        confidence = v.get("confidence", 0.0)

        response = await self.math_answering_chain.ainvoke({
            "question":   state["generation_question"],
            "context":    state["context"],
            "result":     state["execution_result"],
            "validation": (
                f"Recommendation: {recommendation} (confidence: {confidence:.0%})\n\n"
                f"{formatted_validation}\n\n"
                f"Summary: {v.get('summary', '')}"
            )
        })

        log_debug(f"MATH Synthesized Response: {response}")
        return {"final_response": response}

# --- Graph Construction ---
def build_graph() -> StateGraph:
    nodes = AgentNodes(condense_chain, unified_chain, math_coding_chain, math_answering_chain, retriever, prompt, model)
    
    workflow = StateGraph(AgentState)
    
    # Add Nodes
    workflow.add_node("classify_and_retrieve", nodes.classify_and_retrieve)
    workflow.add_node("text_path", nodes.text_path)
    workflow.add_node("math_generate", nodes.math_generate)
    workflow.add_node("math_execute", nodes.math_execute)
    workflow.add_node("math_validate", nodes.math_validate)
    workflow.add_node("synthesize_response", nodes.synthesize_response)
    
    # Set Entry Point
    workflow.set_entry_point("classify_and_retrieve")
    
    # Conditional Routing based on Math Intent Classifier
    workflow.add_conditional_edges(
        "classify_and_retrieve",
        lambda state: "math_path" if state.get("math_intent") else "text_path",
        {
            "math_path": "math_generate",
            "text_path": "text_path"
        }
    )
    
    # Build Math Agent Pipeline
    # workflow.add_edge("math_extract", "math_generate")
    workflow.add_edge("math_generate", "math_execute")
    workflow.add_edge("math_execute", "math_validate")
    # workflow.add_edge("math_validate", "synthesize_response")
    workflow.add_conditional_edges(
        "math_validate",
        lambda state: "synthesize_response" if state.get("validation_passed", True) else "text_path",
        {
            "synthesize_response": "synthesize_response",
            # On hard REJECT, fall back to the text path which will explain
            # what went wrong using the context without fake numbers
            "text_path": "text_path"
        }
    )
    
    # Both paths lead to END
    workflow.add_edge("text_path", END)
    workflow.add_edge("synthesize_response", END)
    
    return workflow.compile()

app = build_graph()