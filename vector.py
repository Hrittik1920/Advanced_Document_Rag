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
from langchain_chroma import Chroma
from tqdm import tqdm

from data_loader import MultiFormatDocumentLoader
from config import settings

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

DOCUMENTS_DIRECTORY = settings.DOCUMENTS_DIR
DB_LOCATION         = "./hybrid_db"
FILE_HASH_DB        = os.path.join(DB_LOCATION, "file_hashes.json")
BM25_INDEX_FILE     = os.path.join(DB_LOCATION, "bm25_index.pkl")
COLLECTION_NAME     = "multi_format_documents"

EMBEDDINGS      = OllamaEmbeddings(model=settings.LLM_EMBEDDING_MODEL)
RERANKER_MODEL  = settings.CROSS_ENCODER_MODEL

# Stage 1 — how many candidates each retriever returns before fusion
BM25_CANDIDATES   = 50
VECTOR_CANDIDATES = 50
# Stage 3 — final docs returned after reranking
FINAL_TOP_K = 10


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
    return hashlib.sha256(key.encode()).hexdigest()


def corpus_doc_id(c: dict) -> str:
    key = f"{c['metadata'].get('source','')}{c['metadata'].get('chunk_number',0)}{c['content']}"
    return hashlib.sha256(key.encode()).hexdigest()


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
    Three-stage retrieval pipeline:
      1. BM25 (sparse)  ──┐
                          ├─→ Reciprocal Rank Fusion ─→ candidate pool
      2. Dense (Chroma) ──┘
      3. Cross-encoder reranker ─→ final top-k
    """
    corpus: List[dict]            # [{"content": str, "metadata": dict}]
    bm25: BM25Okapi
    vector_store: Chroma
    reranker: CrossEncoder
    k: int                  = FINAL_TOP_K
    bm25_candidates: int    = BM25_CANDIDATES
    vector_candidates: int  = VECTOR_CANDIDATES

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
        # Stage 1a: BM25 candidates
        bm25_results  = self._bm25_search(query)

        # Stage 1b: Dense vector candidates
        dense_results = self._dense_search(query)

        # Stage 2: Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(bm25_results, dense_results)

        if not fused:
            return []

        # Stage 3: Cross-encoder reranking
        return self._rerank(query, fused)

    def _bm25_search(self, query: str) -> List[_Document]:
        if not self.corpus:
            return []
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[
            : self.bm25_candidates
        ]
        return [
            _Document(
                page_content=self.corpus[i]["content"],
                metadata=self.corpus[i]["metadata"],
            )
            for i in top_idx
            if scores[i] > 0
        ]

    def _dense_search(self, query: str) -> List[_Document]:
        results = self.vector_store.similarity_search(query, k=self.vector_candidates)
        return [
            _Document(page_content=r.page_content, metadata=r.metadata)
            for r in results
        ]

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
        pairs    = [[query, d.page_content] for d in candidates]
        r_scores = self.reranker.predict(pairs, show_progress_bar=False)
        ranked   = sorted(zip(r_scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[: self.k]]


# ─────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────

def initialize_retriever() -> HybridRetriever:
    print("Initializing Hybrid Retriever (BM25 + Dense + Reranker)...")

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

    # ── Handle deleted files ──────────────────
    deleted = set(file_hashes.keys()) - current_files
    if deleted:
        print(f"Pruning chunks from {len(deleted)} deleted file(s)...")
        corpus = [c for c in corpus if c["metadata"].get("source") not in deleted]
        existing_ids = {corpus_doc_id(c) for c in corpus}
        for fpath in deleted:
            del updated_hashes[fpath]

    # ── Chroma vector store ───────────────────
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_LOCATION,
        embedding_function=EMBEDDINGS,
    )

    # ── Add new documents ─────────────────────
    if docs_to_add:
        print(f"\nIndexing {len(docs_to_add)} new chunks...")

        # Dense: batch-embed into Chroma
        batch_size = 100
        for i in tqdm(range(0, len(docs_to_add), batch_size), desc="Embedding (dense)"):
            batch = docs_to_add[i : i + batch_size]
            ids   = [doc_id(d) for d in batch]
            vector_store.add_documents(documents=batch, ids=ids)

        # Sparse: add new chunks to BM25 corpus
        added = 0
        for d in tqdm(docs_to_add, desc="Indexing  (BM25)"):
            did = doc_id(d)
            if did not in existing_ids:
                corpus.append({"content": d.page_content, "metadata": d.metadata})
                existing_ids.add(did)
                added += 1

        print(f"  {added} unique chunks added to BM25 corpus.")
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

    # ── Load cross-encoder reranker ───────────
    print(f"Loading reranker: {RERANKER_MODEL} ...")
    reranker = CrossEncoder(RERANKER_MODEL)

    print("\nHybrid retriever ready!")
    print(f"  Corpus        : {len(corpus)} chunks")
    print(f"  BM25 pool     : top {BM25_CANDIDATES}")
    print(f"  Vector pool   : top {VECTOR_CANDIDATES}")
    print(f"  Final top-k   : {FINAL_TOP_K} (after reranking)\n")

    return HybridRetriever(
        corpus=corpus,
        bm25=bm25,
        vector_store=vector_store,
        reranker=reranker,
    )


# ─────────────────────────────────────────────
# Module-level retriever — drop-in for your app
# ─────────────────────────────────────────────

retriever = initialize_retriever()