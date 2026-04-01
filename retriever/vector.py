import os
import json
import pickle
import hashlib
import re
import asyncio
from dataclasses import dataclass, field
from typing import List, Optional

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from tqdm import tqdm
import uuid

from data_loader import MultiFormatDocumentLoader, dump_chunks_to_file
from config import settings
from .knowledge_graph import build_graph_retriever, GraphRetriever   # ← new

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

DOCUMENTS_DIRECTORY = settings.DOCUMENTS_DIR
QDRANT_URL = settings.QDRANT_URL
DB_LOCATION         = "./hybrid_db"
FILE_HASH_DB        = os.path.join(DB_LOCATION, "file_hashes.json")
BM25_INDEX_FILE     = os.path.join(DB_LOCATION, "bm25_index.pkl")
COLLECTION_NAME     = "multi_format_documents"

EMBEDDINGS      = OllamaEmbeddings(model=settings.LLM_EMBEDDING_MODEL)
RERANKER_MODEL  = settings.CROSS_ENCODER_MODEL

# Stage 1 — how many candidates each retriever returns before fusion on the basis of thresolds
BM25_SCORE_RATIO   = 0.4
VECTOR_SIMILARITY_THRESOLD = 0.6
GRAPH_SCORE_THRESHOLD  = 0.4      # ← new
# Stage 3 — final docs after reranking
MAX_CANDIDATES_PER_RETRIEVER = 100
RERANKER_THRESHOLD=0.1
FINAL_TOP_K = 15


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def get_file_hash(path: str) -> Optional[str]:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                h.update(chunk)
        return h.hexdigest()
    except (IOError, FileNotFoundError):
        return None


def doc_id(doc) -> str:
    key = f"{doc.metadata.get('source','')}{doc.metadata.get('chunk_number',0)}{doc.page_content}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))  # Using UUID5 for consistent hashing


def corpus_doc_id(c: dict) -> str:
    key = f"{c['metadata'].get('source','')}{c['metadata'].get('chunk_number',0)}{c['content']}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))  # Using UUID5 for consistent hashing


# ─── File-hash persistence ───────────────────

def load_file_hashes() -> dict:
    if os.path.exists(FILE_HASH_DB):
        with open(FILE_HASH_DB) as f:
            return json.load(f)
    return {}


def save_file_hashes(h: dict):
    os.makedirs(DB_LOCATION, exist_ok=True)
    with open(FILE_HASH_DB, "w") as f:
        json.dump(h, f, indent=4)


# ─── BM25 index persistence ──────────────────

def load_bm25_index() -> Optional[dict]:
    if os.path.exists(BM25_INDEX_FILE):
        with open(BM25_INDEX_FILE, "rb") as f:
            return pickle.load(f)
    return None


def save_bm25_index(data: dict):
    os.makedirs(DB_LOCATION, exist_ok=True)
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(data, f)


# ─────────────────────────────────────────────
# Minimal LangChain-compatible Document
# ─────────────────────────────────────────────

@dataclass
class _Document:
    page_content: str
    metadata: dict = field(default_factory=dict)


# ─────────────────────────────────────────────
# Hybrid Retriever
# ─────────────────────────────────────────────

@dataclass
class HybridRetriever:
    """
    Four-stage retrieval pipeline:
      1a. BM25 (sparse)     ──┐
      1b. Dense (Qdrant)    ──┼─→ Reciprocal Rank Fusion ─→ candidate pool
      1c. Knowledge Graph   ──┘
      2.  Cross-encoder reranker ─→ final top-k
    """
    corpus: List[dict]            # [{"content": str, "metadata": dict}]
    bm25: BM25Okapi
    vector_store: QdrantVectorStore
    reranker: CrossEncoder
    graph_retriever: GraphRetriever         # ← new
    k: int                  = FINAL_TOP_K
    bm25_score_ratio: float    = BM25_SCORE_RATIO
    vector_similarity_thresold: float  = VECTOR_SIMILARITY_THRESOLD
    graph_score_thresold: float   = GRAPH_SCORE_THRESHOLD  # ← new
    max_candidates_per_retriever: int =MAX_CANDIDATES_PER_RETRIEVER

    # ── Public interface (LangChain-compatible) ──

    def invoke(self, query: str) -> List[_Document]:
        return self._retrieve(query)

    async def ainvoke(self, query: str) -> List[_Document]:
        return await asyncio.to_thread(self._retrieve, query)

    def get_relevant_documents(self, query: str) -> List[_Document]:
        return self._retrieve(query)

    async def aget_relevant_documents(self, query: str) -> List[_Document]:
        return await asyncio.to_thread(self._retrieve, query)

    # ── Internal pipeline ────────────────────────

    def _retrieve(self, query: str) -> List[_Document]:
        bm25_results   = self._bm25_search(query)
        dense_results  = self._dense_search(query)
        graph_results  = self._graph_search(query)          # ← new

        # 3-way RRF
        fused = self._reciprocal_rank_fusion(bm25_results, dense_results, graph_results)

        if not fused:
            return []

        # Stage 3: Cross-encoder reranking
        return self._rerank(query, fused)

    def _bm25_search(self, query: str) -> List[_Document]:
        if not self.corpus:
            return []
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        
        if not len(scores):
            return []
            
        max_score = max(scores)
        if max_score <= 0:
            return []

        # Dynamic relative threshold
        threshold = max_score * self.bm25_score_ratio
        
        # Filter and sort
        valid_indices = [i for i, s in enumerate(scores) if s >= threshold]
        valid_indices.sort(key=lambda i: scores[i], reverse=True)
        
        # Apply a generous upper limit for safety before RRF
        valid_indices = valid_indices[:self.max_candidates_per_retriever]

        return [
            _Document(
                page_content=self.corpus[i]["content"],
                metadata=self.corpus[i]["metadata"]
            )
            for i in valid_indices
        ]

    def _dense_search(self, query: str) -> List[_Document]:
        results = self.vector_store.similarity_search_with_relevance_scores(query, k=self.max_candidates_per_retriever)
        return [
            _Document(page_content=r.page_content, metadata=r.metadata)
            for r, score in results
            if score>= self.vector_similarity_thresold
        ]

    # ── Stage 1c: Knowledge graph  ──────────────────────────────────────────  ← new
    #
    #  The GraphRetriever returns chunks reachable within hop_depth hops of any
    #  entity mentioned in the query.  This gives us chunks that BM25 and dense
    #  both miss because they don't share keywords or embedding neighbourhood
    #  with the query — they're connected only through a shared entity.
    #
    #  Examples of what this recovers:
    #   • A chunk that defines "Protocol X" when the query asks what company
    #     invented "Protocol X" — linked via the ORG entity.
    #   • Cross-document summaries: chunk A mentions Person P; chunk B (different
    #     doc) also mentions Person P — both bubble up together.
    #   • Multi-hop: query about "ACME–Globex partnership" surfaces chunks that
    #     each mention only one of the two orgs but co-occur with the same third
    #     entity (a product), linking them through 2 hops.

    def _graph_search(self, query: str) -> List[_Document]:
        if self.graph_retriever is None:
            return []
        try:
            # Assuming your graph_retriever can return scores (e.g., PageRank or edge weight)
            # If it only returns raw docs, you may have to threshold by hop-distance instead.
            results = self.graph_retriever.invoke_with_scores(query) 
            
            valid_docs = [
                doc for doc, score in results 
                if score >= self.graph_score_thresold
            ]
            return valid_docs[:self.max_candidates_per_retriever]
            
        except AttributeError:
             # Fallback if your GraphRetriever doesn't support scores
             print("[GraphRetriever] Warning: No score mechanism found. Returning top-K.")
             return self.graph_retriever.invoke(query)[:30]
        except Exception as exc:
            # Graph retrieval is best-effort; never crash the pipeline
            print(f"[GraphRetriever] warning: {exc}")
            return []

    # ── Stage 2: Reciprocal Rank Fusion ────────────────────────────────────

    @staticmethod
    def _reciprocal_rank_fusion(
        *ranked_lists: List[_Document], k: int = 60
    ) -> List[_Document]:
        """
        Merge multiple ranked lists into one using RRF.
        Score = Σ  1 / (k + rank)    (k=60 is the RRF standard default)
        Higher k → less penalty for lower-ranked docs.
        """
        scores: dict = {}
        docs:   dict = {}

        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked, start=1):
                key = hashlib.sha256(doc.page_content.encode()).hexdigest()
                scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                docs[key]   = doc

        merged = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [docs[key] for key in merged]

    def _rerank(self, query: str, candidates: List[_Document]) -> List[_Document]:
        if not candidates:
            return []
            
        pairs = [[query, d.page_content] for d in candidates]
        r_scores = self.reranker.predict(pairs, show_progress_bar=False)
        
        # Filter purely by cross-encoder threshold
        ranked = sorted(zip(r_scores, candidates), key=lambda x: x[0], reverse=True)
        
        final_docs = [doc for score, doc in ranked if score >= RERANKER_THRESHOLD]
        
        # Optional: Apply absolute cap to avoid flooding the LLM context window
        return final_docs[:FINAL_TOP_K]
    
def initialize_retriever() -> HybridRetriever:
    print("Initializing Hybrid Retriever (BM25 + Dense + KG +Reranker)...")

    loader       = MultiFormatDocumentLoader()
    file_hashes  = load_file_hashes()
    bm25_data    = load_bm25_index()

    corpus:       List[dict] = bm25_data["corpus"] if bm25_data else []
    existing_ids: set        = bm25_data["ids"]    if bm25_data else set()

    updated_hashes = file_hashes.copy()
    docs_to_add    = []
    current_files  = set()

    # ── Scan document directory ───────────────
    all_files = [
        os.path.join(root, fname)
        for root, _, files in os.walk(DOCUMENTS_DIRECTORY)
        for fname in files
        if not fname.startswith(".")
    ]
    print(f"Found {len(all_files)} files to check...")

    for fpath in tqdm(all_files, desc="Checking file status"):
        current_files.add(fpath)
        new_hash = get_file_hash(fpath)
        if new_hash and file_hashes.get(fpath) != new_hash:
            print(f"\n  Change detected: {fpath}")
            docs_to_add.extend(loader.load_document(fpath))
            updated_hashes[fpath] = new_hash
    dump_chunks_to_file(docs_to_add, output_path="vector.txt")

    # ── Handle deleted files ──────────────────
    deleted = set(file_hashes.keys()) - current_files
    deleted_chunk_ids = []
    
    if deleted:
        print(f"Pruning chunks from {len(deleted)} deleted file(s)...")
        # Identify which chunks belong to the deleted files to remove them from Qdrant later
        deleted_chunk_ids = [corpus_doc_id(c) for c in corpus if c["metadata"].get("source") in deleted]
        
        corpus = [c for c in corpus if c["metadata"].get("source") not in deleted]
        existing_ids = {corpus_doc_id(c) for c in corpus}
        for fpath in deleted:
            del updated_hashes[fpath]

    # ── Qdrant vector store (Safely Isolated) ──
    # qdrant_path = os.path.join(DB_LOCATION, "qdrant_data")
    
    # Using local Qdrant. Adjust url/api_key here if you use Qdrant Cloud/Docker.
    client = QdrantClient(url=QDRANT_URL)
    size = len(EMBEDDINGS.embed_query("test"))
    print(f"Embedding dimension detected: {size}")
    try:
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if not exists:
            print(f"Creating collection '{COLLECTION_NAME}'...")
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=size, # Ensure this matches your Ollama model dimensions (e.g., 768 for mxbai-embed-large)
                    distance=Distance.COSINE
                ),
            )
    except Exception as e:
        print(f"Error checking/creating Qdrant collection: {e}")

    # Now initialize the store safely
    try:
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=EMBEDDINGS,
        )
    except Exception as e:
        print(f"Critical error: Could not initialize QdrantVectorStore: {e}")
        # Re-raise or exit because we cannot proceed without the vector_store
        raise e
    # Safely delete orphaned vectors from Qdrant if files were removed
    if deleted_chunk_ids:
        print(f"Removing {len(deleted_chunk_ids)} orphaned vectors from Qdrant collection '{COLLECTION_NAME}'...")
        try:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=deleted_chunk_ids
            )
        except Exception as e:
            print(f"Warning: Could not delete old points from Qdrant: {e}")

    # ── Add new documents ─────────────────────
    if docs_to_add:
        print(f"\nIndexing {len(docs_to_add)} new chunks...")

        # 1. Filter out duplicates WITHIN the new docs_to_add list itself
        unique_docs_to_add = {}
        for d in docs_to_add:
            did = doc_id(d)
            if did not in unique_docs_to_add:
                unique_docs_to_add[did] = d
        
        # 2. Convert back to list and prepare for batching
        final_docs = list(unique_docs_to_add.values())
        final_ids = list(unique_docs_to_add.keys())

        # Dense: batch-embed into Chroma
        batch_size = 100
        for i in tqdm(range(0, len(final_docs), batch_size), desc="Embedding (dense)"):
            batch = final_docs[i : i + batch_size]
            batch_ids = final_ids[i : i + batch_size]
            
            # Use upsert instead of add_documents to be safer
            vector_store.add_documents(documents=batch, ids=batch_ids)

        # Sparse: add new chunks to BM25 corpus (using the same unique list)
        added_count = 0
        for did, d in unique_docs_to_add.items():
            if did not in existing_ids:
                corpus.append({"content": d.page_content, "metadata": d.metadata})
                existing_ids.add(did)
                added_count += 1

        print(f"  {added_count} unique chunks added to BM25 corpus.")
        save_bm25_index({"corpus": corpus, "ids": existing_ids})
        save_file_hashes(updated_hashes)
        print("Indexes saved to disk.")
    else:
        print("Indexes are up to date.")

    if not corpus:
        raise ValueError("No documents indexed. Add files to the documents directory.")

    # ── Build in-memory BM25 ──────────────────
    print("Building BM25 index...")
    tokenized = [tokenize(c["content"]) for c in corpus]
    bm25 = BM25Okapi(tokenized)

    # ── Build / update knowledge graph ───────     ← new
    print("Building Knowledge Graph …")
    graph_ret = build_graph_retriever(corpus)

    # ── Load cross-encoder reranker ───────────
    print(f"Loading reranker: {RERANKER_MODEL} ...")
    reranker = CrossEncoder(RERANKER_MODEL)

    print("\nHybrid retriever ready!")
    print(f"  Corpus        : {len(corpus)} chunks")
    print(f"  BM25 pool     : top {BM25_SCORE_RATIO}")
    print(f"  Vector pool   : top {VECTOR_SIMILARITY_THRESOLD}")
    print(f"  Graph pool    : top {GRAPH_SCORE_THRESHOLD}")
    print(f"  Final top-k   : {FINAL_TOP_K} (after reranking)\n")

    return HybridRetriever(
        corpus=corpus,
        bm25=bm25,
        vector_store=vector_store,
        reranker=reranker,
        graph_retriever=graph_ret,      # ← new
    )


# ─────────────────────────────────────────────
# Module-level retriever — drop-in for your app
# ─────────────────────────────────────────────

retriever = initialize_retriever()