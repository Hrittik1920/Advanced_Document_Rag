"""
retriever/knowledge_graph.py
──────────────────
Builds an entity–relation graph over your document corpus and exposes a
LangChain-compatible retriever that returns chunks reachable from entities
mentioned in the query.

Dependencies
------------
pip install spacy networkx
python -m spacy download en_core_web_md  # swap for en_core_web_trf for accuracy
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import spacy
from .models import _Document
# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

KG_DIR        = "./hybrid_db"
KG_GRAPH_FILE = os.path.join(KG_DIR, "knowledge_graph.pkl")
KG_INDEX_FILE = os.path.join(KG_DIR, "kg_entity_index.json")

# spaCy entity labels we care about – tune to your domain
RELEVANT_LABELS: Set[str] = {
    "PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT",
    "WORK_OF_ART", "LAW", "FAC", "NORP",
    # add domain-specific ones, e.g. "DISEASE", "DRUG" for medical corpora
}

# How many hops to expand from a matched entity node
DEFAULT_HOP_DEPTH  = 3
# Max chunks returned by the graph retriever per query
GRAPH_CANDIDATES   = 30


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chunk_id(source: str, chunk_number: int, content: str) -> str:
    key = f"{source}{chunk_number}{content}"
    return hashlib.sha256(key.encode()).hexdigest()


def _normalise(text: str) -> str:
    """Lower-case, strip punctuation → canonical entity name."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Graph schema
# ─────────────────────────────────────────────────────────────────────────────
#
#  Node types
#  ──────────
#  "entity"  – a named entity extracted by spaCy
#              attrs: label (NER type), name (raw text), norm (normalised)
#
#  "chunk"   – a document chunk
#              attrs: content, metadata, cid (hash id)
#
#  Edge types
#  ──────────
#  "mentions"      – chunk  → entity  (chunk mentions that entity)
#  "co_occurs"     – entity → entity  (both mentioned in the same chunk)
#  "same_as"       – entity → entity  (same normalised form, different raw text)
#
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_kg() -> Optional[Tuple[nx.DiGraph, Dict[str, str]]]:
    """
    Returns (graph, entity_norm_index) or None if no persisted graph exists.
    entity_norm_index maps normalised entity name → canonical node id.
    """
    if os.path.exists(KG_GRAPH_FILE) and os.path.exists(KG_INDEX_FILE):
        with open(KG_GRAPH_FILE, "rb") as f:
            G = pickle.load(f)
        with open(KG_INDEX_FILE) as f:
            idx = json.load(f)
        return G, idx
    return None


def save_kg(G: nx.DiGraph, entity_index: Dict[str, str]) -> None:
    os.makedirs(KG_DIR, exist_ok=True)
    with open(KG_GRAPH_FILE, "wb") as f:
        pickle.dump(G, f)
    with open(KG_INDEX_FILE, "w") as f:
        json.dump(entity_index, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class KnowledgeGraphBuilder:
    """
    Incrementally builds a NetworkX DiGraph from a corpus of chunks.

    Usage
    -----
    builder = KnowledgeGraphBuilder()
    builder.add_chunks(new_chunks)   # list of {"content": str, "metadata": dict}
    builder.remove_source("path/to/deleted/file.pdf")
    G, idx = builder.G, builder.entity_index
    save_kg(G, idx)
    """

    _nlp: Any               = field(init=False, repr=False)
    G: nx.DiGraph           = field(init=False, repr=False)
    entity_index: Dict[str, str] = field(init=False, repr=False)  # norm → node_id

    def __post_init__(self) -> None:
        print("Loading spaCy model …")
        try:
            self._nlp = spacy.load("en_core_web_md")
        except OSError as e:
            print(f"⚠️ SpaCy model not found: {e}")
            print("Try running: python -m spacy download en_core_web_md")
            # Fallback to smaller model
            try:
                print("Attempting fallback to en_core_web_sm...")
                self._nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("❌ No spaCy model available. Knowledge graph will be disabled.")
                self._nlp = None

        loaded = load_kg()
        if loaded:
            self.G, self.entity_index = loaded
            print(f"  ✓ Loaded existing KG: {self.G.number_of_nodes()} nodes, "
                  f"{self.G.number_of_edges()} edges.")
        else:
            self.G = nx.DiGraph()
            self.entity_index = {}
            print("  Starting fresh KG.")

    # ── Public API ────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Dict]) -> int:
        """
        Process a list of corpus chunks and add them to the graph.
        Returns the number of new chunk nodes added.
        Handles large batches with progress tracking and error resilience.
        """
        added = 0
        failed = 0
        
        # Process in batches to avoid memory issues with large corpus
        batch_size = 50
        for batch_idx in range(0, len(chunks), batch_size):
            batch = chunks[batch_idx : batch_idx + batch_size]
            batch_num = batch_idx // batch_size + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            print(f"  Processing batch {batch_num}/{total_batches} ({len(batch)} chunks)...")
            
            for chunk in batch:
                try:
                    cid = _chunk_id(
                        chunk["metadata"].get("source", ""),
                        chunk["metadata"].get("chunk_number", 0),
                        chunk["content"],
                    )
                    if self.G.has_node(cid):
                        continue  # already indexed

                    # Add chunk node
                    self.G.add_node(
                        cid,
                        node_type="chunk",
                        content=chunk["content"],
                        metadata=chunk["metadata"],
                        cid=cid,
                    )

                    # Extract entities from this chunk
                    entities = self._extract_entities(chunk["content"])
                    entity_ids_in_chunk: List[str] = []

                    for ent_text, ent_label in entities:
                        ent_id = self._get_or_create_entity(ent_text, ent_label)
                        # chunk → entity
                        if not self.G.has_edge(cid, ent_id):
                            self.G.add_edge(cid, ent_id, rel="mentions")
                        entity_ids_in_chunk.append(ent_id)

                    # entity ↔ entity co-occurrence within same chunk
                    for i, eid_a in enumerate(entity_ids_in_chunk):
                        for eid_b in entity_ids_in_chunk[i + 1:]:
                            if eid_a != eid_b:
                                if not self.G.has_edge(eid_a, eid_b):
                                    self.G.add_edge(eid_a, eid_b, rel="co_occurs", weight=1)
                                else:
                                    # Increment co-occurrence weight
                                    self.G[eid_a][eid_b]["weight"] = (
                                        self.G[eid_a][eid_b].get("weight", 1) + 1
                                    )

                    added += 1
                    
                except Exception as e:
                    failed += 1
                    print(f"    ⚠️ Error processing chunk: {str(e)[:100]}")
                    continue

        if failed > 0:
            print(f"  ⚠️ {failed} chunks failed to process")
        print(f"  ✓ {added} chunks successfully added to KG")
        return added

    def remove_source(self, source_path: str) -> int:
        """Remove all chunk nodes (and dangling entity nodes) for a deleted file."""
        to_remove = [
            n for n, d in self.G.nodes(data=True)
            if d.get("node_type") == "chunk"
            and d.get("metadata", {}).get("source") == source_path
        ]
        self.G.remove_nodes_from(to_remove)

        # Prune entity nodes with no remaining edges
        dangling = [
            n for n, d in self.G.nodes(data=True)
            if d.get("node_type") == "entity" and self.G.degree(n) == 0
        ]
        self.G.remove_nodes_from(dangling)

        # Rebuild entity_index
        self.entity_index = {
            d["norm"]: n
            for n, d in self.G.nodes(data=True)
            if d.get("node_type") == "entity"
        }

        return len(to_remove)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Returns [(entity_text, label), …] filtered to RELEVANT_LABELS."""
        if self._nlp is None:
            return []  # No spaCy model available
        
        try:
            doc = self._nlp(text[:self._nlp.max_length])
            seen: Set[str] = set()
            results: List[Tuple[str, str]] = []
            for ent in doc.ents:
                if ent.label_ not in RELEVANT_LABELS:
                    continue
                norm = _normalise(ent.text)
                if not norm or norm in seen:
                    continue
                seen.add(norm)
                results.append((ent.text, ent.label_))
            return results
        except Exception as e:
            print(f"⚠️ Error extracting entities: {str(e)[:100]}")
            return []

    def _get_or_create_entity(self, ent_text: str, ent_label: str) -> str:
        """
        Look up or create an entity node.
        Entities that normalise to the same string share a node (same_as merge).
        Returns the node id.
        """
        norm = _normalise(ent_text)
        if norm in self.entity_index:
            node_id = self.entity_index[norm]
            # If this raw text is new, record it as an alias
            existing = self.G.nodes[node_id]
            if ent_text not in existing.get("aliases", []):
                existing.setdefault("aliases", []).append(ent_text)
            return node_id

        # New entity
        node_id = f"ent:{norm.replace(' ', '_')}"
        self.G.add_node(
            node_id,
            node_type="entity",
            label=ent_label,
            name=ent_text,
            norm=norm,
            aliases=[ent_text],
        )
        self.entity_index[norm] = node_id
        return node_id


# ─────────────────────────────────────────────────────────────────────────────
# Retriever
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GraphRetriever:
    """
    LangChain-compatible retriever that:
      1. Extracts named entities from the query via spaCy.
      2. Looks up matching entity nodes in the KG.
      3. Expands hop_depth hops along any edge.
      4. Collects all reachable chunk nodes.
      5. Ranks them by graph centrality × reachability score.

    Drop this into HybridRetriever._graph_search().
    """

    G: nx.DiGraph
    entity_index: Dict[str, str]
    nlp: Any
    hop_depth: int   = DEFAULT_HOP_DEPTH
    top_k: int       = GRAPH_CANDIDATES

    # ── LangChain shim ────────────────────────────────────────────────────────

    def invoke(self, query: str) -> List[Any]:
        return self._retrieve(query)

    async def ainvoke(self, query: str) -> List[Any]:
        import asyncio
        return await asyncio.to_thread(self._retrieve, query)

    def get_relevant_documents(self, query: str) -> List[Any]:
        return self._retrieve(query)

    # Add this new method that the HybridRetriever is looking for:
    def invoke_with_scores(self, query: str) -> List[Tuple[Any, float]]:
        return self._retrieve_with_scores(query)

    # ── Core logic ────────────────────────────────────────────────────────────

    def _retrieve(self, query: str) -> List[Any]:
        """LangChain compatible method (drops the scores)."""
        return [doc for doc, score in self._retrieve_with_scores(query)]

    def _retrieve_with_scores(self, query: str) -> List[Tuple[Any, float]]:
        """
        Returns _Document-like objects along with their graph proximity score.
        """

        if self.G.number_of_nodes() == 0:
            return []

        # Step 1: entity extraction from query
        query_entities = self._extract_query_entities(query)
        if not query_entities:
            return []

        # Step 2: seed nodes — entity nodes that match query entities
        seed_nodes: List[str] = []
        for norm in query_entities:
            if norm in self.entity_index:
                seed_nodes.append(self.entity_index[norm])
            else:
                # Partial / substring match
                for idx_norm, node_id in self.entity_index.items():
                    if norm in idx_norm or idx_norm in norm:
                        seed_nodes.append(node_id)

        if not seed_nodes:
            return []

        # Step 3: hop expansion — collect all nodes within hop_depth
        reachable: Dict[str, float] = {}  # node_id → proximity score

        for seed in seed_nodes:
            # BFS up to hop_depth, scoring by 1 / (hop + 1)
            visited = {seed: 0}
            frontier = [seed]
            for hop in range(self.hop_depth):
                next_frontier = []
                for node in frontier:
                    neighbors = list(self.G.successors(node)) + list(self.G.predecessors(node))
                    for nb in neighbors:
                        if nb not in visited:
                            visited[nb] = hop + 1
                            next_frontier.append(nb)
                frontier = next_frontier

            for node, depth in visited.items():
                score = 1.0 / (depth + 1)
                reachable[node] = reachable.get(node, 0.0) + score

        # Step 4: collect chunk nodes from the reachable set
        chunk_scores: Dict[str, float] = {}
        for node_id, score in reachable.items():
            d = self.G.nodes[node_id]
            if d.get("node_type") == "chunk":
                chunk_scores[node_id] = score

        # Fallback: for reachable entity nodes, follow "mentions" edges back to chunks
        for node_id, score in reachable.items():
            d = self.G.nodes[node_id]
            if d.get("node_type") == "entity":
                for pred in self.G.predecessors(node_id):
                    pd = self.G.nodes.get(pred, {})
                    if pd.get("node_type") == "chunk":
                        chunk_scores[pred] = chunk_scores.get(pred, 0.0) + score * 0.5

        if not chunk_scores:
            return []

        # Step 5: rank and return top_k ALONG WITH SCORES
        ranked = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)[: self.top_k]

        results = []
        for node_id, score in ranked:
            nd = self.G.nodes[node_id]
            doc = _Document(
                page_content=nd["content"],
                metadata=nd["metadata"],
            )
            results.append((doc, score)) # Append the tuple!
            
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_query_entities(self, query: str) -> List[str]:
        """Returns normalised entity strings from the query."""
        if self.nlp is None:
            return []  # No spaCy model available
        
        try:
            doc = self.nlp(query)
            norms = []
            for ent in doc.ents:
                if ent.label_ in RELEVANT_LABELS:
                    n = _normalise(ent.text)
                    if n:
                        norms.append(n)
            # Also add important noun chunks as fallback (in case NER misses things)
            if not norms:
                for chunk in doc.noun_chunks:
                    n = _normalise(chunk.text)
                    if len(n) > 2:
                        norms.append(n)
            return norms
        except Exception as e:
            print(f"⚠️ Error extracting query entities: {str(e)[:100]}")
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Factory — used by retriever.py
# ─────────────────────────────────────────────────────────────────────────────

def build_graph_retriever(corpus: List[Dict]) -> GraphRetriever:
    """
    Build (or update) the KG from corpus, persist it, return a GraphRetriever.
    Called once from initialize_retriever().
    Handles large corpus gracefully with progress tracking and error recovery.
    """
    try:
        builder = KnowledgeGraphBuilder()
    except Exception as e:
        print(f"❌ Failed to initialize KnowledgeGraphBuilder: {e}")
        print("⚠️ Knowledge Graph will be disabled for this session")
        # Return a disabled GraphRetriever
        return GraphRetriever(
            G=nx.DiGraph(),
            entity_index={},
            nlp=None,
        )

    # Identify which chunks are new (not yet in the graph)
    new_chunks = [
        c for c in corpus
        if not builder.G.has_node(
            _chunk_id(
                c["metadata"].get("source", ""),
                c["metadata"].get("chunk_number", 0),
                c["content"],
            )
        )
    ]

    if new_chunks:
        print(f"Building KG: indexing {len(new_chunks)} new chunks …")
        try:
            added = builder.add_chunks(new_chunks)
            save_kg(builder.G, builder.entity_index)
            print(f"  ✓ KG saved: {added} chunks, "
                  f"{builder.G.number_of_nodes()} nodes, "
                  f"{builder.G.number_of_edges()} edges.")
        except Exception as e:
            print(f"❌ Error building KG: {e}")
            print("⚠️ Continuing without KG updates")
    else:
        print("KG is up to date.")

    return GraphRetriever(
        G=builder.G,
        entity_index=builder.entity_index,
        nlp=builder._nlp,
    )