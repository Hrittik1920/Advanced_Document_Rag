import json
import time
import datetime
import os
import asyncio
import re
import sys
from typing import Dict, List, Any
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from retriever import retriever
from script import original_chain, condense_chain
from config import settings
from langchain_core.messages import AIMessage, HumanMessage

# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC TEST RUNNER - RAG Pipeline Quality Assessment
# ═══════════════════════════════════════════════════════════════════════════════

class RAGDiagnosticTester:
    """
    Tests the RAG pipeline component-by-component and tracks:
    - Retrieval quality (BM25, Dense, Knowledge Graph)
    - Fusion quality (RRF scoring)
    - Reranking effectiveness
    - LLM generation quality
    - Citation accuracy
    """
    
    def __init__(self):
        self.retriever = retriever
        self.results = []
        self.current_test = {}
    
    async def run_tests(self, questions: List[Dict]):
        """Run diagnostic tests on each question."""
        print("🚀 Starting RAG Diagnostic Tests...\n")
        
        for idx, q_data in enumerate(questions, 1):
            start_total = time.perf_counter()
            question_id = q_data.get("id", idx)
            question = q_data.get("question", "")
            
            print(f"\n{'='*80}")
            print(f"📌 Test {idx}/{len(questions)}: ID={question_id}")
            print(f"   Question: {question}")
            print(f"{'='*80}")
            
            self.current_test = {
                "id": question_id,
                "question": question,
                "timestamp": datetime.datetime.now().isoformat(),
                "retrieval": {},
                "generation": {},
                "citations": {},
                "quality_metrics": {}
            }
            
            try:
                # Stage 1: Retrieval
                await self._test_retrieval(question)
                
                # Stage 2: Generation
                await self._test_generation(question)
                
                # Stage 3: Citation analysis
                self._analyze_citations()
                
                # Stage 4: Quality metrics
                # self._compute_quality_metrics()
                
                self.results.append(self.current_test)
                print(f"\n✅ Test {question_id} completed successfully")
                
            except Exception as e:
                print(f"\n❌ Test {question_id} failed: {str(e)}")
                self.current_test["error"] = str(e)
                total_time_ms = (time.perf_counter() - start_total) * 1000

                self.current_test["timings"] = {
                    "retrieval_ms": self.current_test.get("retrieval", {}).get("latency_ms", 0),
                    "generation_ms": self.current_test.get("generation", {}).get("latency_ms", 0),
                    "total_time_ms": round(total_time_ms, 2)
                }
                self.results.append(self.current_test)
                import traceback
                traceback.print_exc()
    
    async def _test_retrieval(self, question: str):
        """Test retrieval pipeline: BM25 + Dense + KG → RRF → Reranking."""
        print("\n🔍 RETRIEVAL STAGE")
        print("-" * 80)
        
        # Measure retrieval time
        start_retrieval = time.perf_counter()
        
        try:
            # Call the retriever (which uses HybridRetriever internally)
            retrieved_docs = await asyncio.to_thread(
                self.retriever.get_relevant_documents, 
                question
            )
            
            retrieval_time_ms = (time.perf_counter() - start_retrieval) * 1000
            num_docs = len(retrieved_docs)
            
            print(f"   ⏱️  Retrieval Time: {retrieval_time_ms:.2f} ms")
            print(f"   📄 Documents Retrieved: {num_docs}")
            
            # Extract scores from metadata if available
            scores = []
            for doc in retrieved_docs:
                meta = doc.metadata
                if "score" in meta:
                    scores.append(meta["score"])
            
            if scores:
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                min_score = min(scores)
                print(f"   📊 Score Stats:")
                print(f"      - Max: {max_score:.4f}")
                print(f"      - Min: {min_score:.4f}")
                print(f"      - Avg: {avg_score:.4f}")
                self.current_test["retrieval"]["scores"] = {
                    "max": round(max_score, 4),
                    "min": round(min_score, 4),
                    "avg": round(avg_score, 4)
                }
            
            # Document diversity (deduplication check)
            contents = [doc.page_content[:100] for doc in retrieved_docs]
            unique_contents = len(set(contents))
            diversity_ratio = unique_contents / max(num_docs, 1)
            print(f"   🔄 Diversity: {unique_contents}/{num_docs} unique docs ({diversity_ratio*100:.1f}%)")
            
            # Store retrieval metadata
            self.current_test["retrieval"] = {
                "latency_ms": round(retrieval_time_ms, 2),
                "num_docs": num_docs,
                "diversity_ratio": round(diversity_ratio, 3),
                "top_sources": [doc.metadata.get("source", "unknown") for doc in retrieved_docs[:5]],
                
            }
            
            # Print top 5 documents
            print(f"\n   📌 Top 5 Retrieved Documents:")
            for i, doc in enumerate(retrieved_docs[:5], 1):
                source = doc.metadata.get("source", "unknown")
                section = doc.metadata.get("section", "N/A")
                preview = doc.page_content[:60].replace("\n", " ")
                print(f"      {i}. [{source}] {section}")
                print(f"         → {preview}...")
            
            # Store full docs for later use in generation
            self.current_test["_retrieved_docs"] = retrieved_docs
            
        except Exception as e:
            print(f"   ❌ Retrieval failed: {e}")
            self.current_test["retrieval"]["error"] = str(e)
            raise
    
    async def _test_generation(self, question: str):
        """Test LLM generation using retrieved context."""
        print("\n🧠 GENERATION STAGE")
        print("-" * 80)
        
        retrieved_docs = self.current_test.get("_retrieved_docs", [])
        
        if not retrieved_docs:
            print("   ⚠️  No documents retrieved; skipping generation")
            return
        
        # Format context from retrieved documents
        context = self._format_context(retrieved_docs)
        
        print(f"   📝 Context Size: {len(context)} chars")
        
        # Call LLM with context
        start_gen = time.perf_counter()
        
        try:
            # Simple LLM call without history
            response = await original_chain.ainvoke({
                "question": question,
                "context": context,
                settings.HISTORY_DIR: [],  # Empty chat history for diagnostics
                "uploaded_doc_text": ""
            })
            
            gen_time_ms = (time.perf_counter() - start_gen) * 1000
            
            print(f"   ⏱️  Generation Time: {gen_time_ms:.2f} ms")
            print(f"   📄 Response Length: {len(response)} chars")
            
            # Extract cited document indices from response
            citation_pattern = r'[\[【](\d+)[\]】]'
            cited_ids = set()
            for match in re.finditer(citation_pattern, response):
                try:
                    cited_ids.add(int(match.group(1)))
                except ValueError:
                    pass
            
            print(f"   🔗 Citations Found: {sorted(cited_ids) if cited_ids else 'None'}")
            
            # Store generation metadata
            self.current_test["generation"] = {
                "latency_ms": round(gen_time_ms, 2),
                "response_length": len(response),
                "response": response,
                "cited_ids": sorted(list(cited_ids)),
            }
            
            # Store full response for citation analysis
            self.current_test["_response"] = response
            
        except Exception as e:
            print(f"   ❌ Generation failed: {e}")
            self.current_test["generation"]["error"] = str(e)
            import traceback
            traceback.print_exc()
    
    def _format_context(self, docs: List) -> str:
        """Format retrieved documents as context for the LLM."""
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            section = doc.metadata.get("section", "")
            content = doc.page_content
            
            context_parts.append(f"[{i}] ({source} - {section})\n{content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def _analyze_citations(self):
        """Analyze citation accuracy and coverage."""
        print("\n🔗 CITATION ANALYSIS")
        print("-" * 80)
        
        response = self.current_test.get("_response", "")
        retrieved_docs = self.current_test.get("_retrieved_docs", [])
        
        if not response or not retrieved_docs:
            print("   ⚠️  No response or documents to analyze")
            return
        
        # Extract all citations from response
        citation_pattern = r'[\[【](\d+)[\]】]'
        cited_ids = []
        for match in re.finditer(citation_pattern, response):
            try:
                cited_ids.append(int(match.group(1)))
            except ValueError:
                pass
        
        num_citations = len(cited_ids)
        num_unique_citations = len(set(cited_ids))
        
        print(f"   📊 Citation Stats:")
        print(f"      - Total Citations: {num_citations}")
        print(f"      - Unique Sources: {num_unique_citations}")
        
        # Check for invalid citation IDs
        max_valid_id = len(retrieved_docs)
        invalid_ids = [cid for cid in set(cited_ids) if cid > max_valid_id or cid < 1]
        
        if invalid_ids:
            print(f"   ⚠️  Invalid Citation IDs: {invalid_ids} (max valid: {max_valid_id})")
        else:
            print(f"   ✅ All citations valid (range: 1-{max_valid_id})")
        
        # Citation density
        citation_density = num_citations / max(len(response.split()), 1)
        print(f"   📈 Citation Density: {citation_density:.4f} citations per word")
        
        self.current_test["citations"] = {
            "total_citations": num_citations,
            "unique_sources": num_unique_citations,
            "invalid_ids": invalid_ids,
            "citation_density": round(citation_density, 4)
        }
    
    # def _compute_quality_metrics(self):
    #     """Compute overall RAG quality metrics."""
    #     print("\n📊 QUALITY METRICS")
    #     print("-" * 80)
        
    #     retrieval = self.current_test.get("retrieval", {})
    #     generation = self.current_test.get("generation", {})
    #     citations = self.current_test.get("citations", {})
        
    #     metrics = {}
        
    #     # Retrieval quality score (0-100)
    #     if "num_docs" in retrieval:
    #         num_docs = retrieval["num_docs"]
    #         diversity = retrieval.get("diversity_ratio", 0)
    #         retrieval_score = min(100, (num_docs / 10) * 50 + diversity * 50)
    #         metrics["retrieval_score"] = round(retrieval_score, 1)
    #         print(f"   🔍 Retrieval Quality: {retrieval_score:.1f}/100")
        
    #     # Generation quality (based on response length)
    #     if "response_length" in generation:
    #         response_len = generation["response_length"]
    #         gen_score = min(100, (response_len / 500) * 100)
    #         metrics["generation_score"] = round(gen_score, 1)
    #         print(f"   🧠 Generation Quality: {gen_score:.1f}/100")
        
    #     # Citation quality (based on validity and density)
    #     if citations.get("invalid_ids"):
    #         citation_score = max(0, 100 - len(citations["invalid_ids"]) * 20)
    #     else:
    #         citation_score = 100
    #     metrics["citation_score"] = citation_score
    #     print(f"   🔗 Citation Quality: {citation_score:.1f}/100")
        
    #     # Overall score
    #     scores = [v for k, v in metrics.items() if k.endswith("_score")]
    #     if scores:
    #         overall_score = sum(scores) / len(scores)
    #         metrics["overall_score"] = round(overall_score, 1)
    #         print(f"   ⭐ Overall Score: {overall_score:.1f}/100")
        
    #     self.current_test["quality_metrics"] = metrics
    
    def save_results(self):
        """Save test results to JSON file."""
        folder = "TestAnswerLog"
        os.makedirs(folder, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d---%H-%M-%S")
        file_path = os.path.join(folder, f"{timestamp}_diagnostic_results.json")
        
        # Clean up temporary keys before saving
        for result in self.results:
            result.pop("_retrieved_docs", None)
            # result.pop("_response", None)
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*80}")
        print(f"📁 Results saved to: {file_path}")
        print(f"{'='*80}")
        
        # Print summary statistics
        self._print_summary()
    
    def _print_summary(self):
        """Print summary statistics of all tests."""
        if not self.results:
            return
        
        print("\n📈 TEST SUMMARY")
        print("-" * 80)
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if "error" not in r)
        failed_tests = total_tests - successful_tests
        
        print(f"   Total Tests: {total_tests}")
        print(f"   ✅ Successful: {successful_tests}")
        print(f"   ❌ Failed: {failed_tests}")
        
        # Average metrics
        retrieval_times = []
        gen_times = []
        overall_scores = []
        
        for result in self.results:
            if "retrieval" in result and "latency_ms" in result["retrieval"]:
                retrieval_times.append(result["retrieval"]["latency_ms"])
            if "generation" in result and "latency_ms" in result["generation"]:
                gen_times.append(result["generation"]["latency_ms"])
            if "quality_metrics" in result and "overall_score" in result["quality_metrics"]:
                overall_scores.append(result["quality_metrics"]["overall_score"])
        
        if retrieval_times:
            avg_retrieval = sum(retrieval_times) / len(retrieval_times)
            print(f"\n   ⏱️  Average Retrieval Time: {avg_retrieval:.2f} ms")
        
        if gen_times:
            avg_gen = sum(gen_times) / len(gen_times)
            print(f"   ⏱️  Average Generation Time: {avg_gen:.2f} ms")
        
        if overall_scores:
            avg_score = sum(overall_scores) / len(overall_scores)
            print(f"   ⭐ Average Overall Score: {avg_score:.1f}/100")


async def main():
    """Main entry point for diagnostic testing."""
    try:
        # Load test questions
        with open("test_questions.json") as f:
            questions = json.load(f)
        
        print(f"\n🔄 Loaded {len(questions)} test questions\n")
        
        # Create tester and run tests
        tester = RAGDiagnosticTester()
        await tester.run_tests(questions)
        
        # Save results
        tester.save_results()
        
    except FileNotFoundError:
        print("❌ Error: test_questions.json not found")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())