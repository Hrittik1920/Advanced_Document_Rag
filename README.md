# BillPro Multi-Docs Chatbot - RAG System

A sophisticated **Retrieval-Augmented Generation (RAG)** system designed for intelligent document analysis and context-aware answering. This system powers a multi-user chatbot that processes electricity tariff documents and complex PDFs using a hybrid retrieval pipeline combining BM25, dense embeddings, and knowledge graphs.

---

## Table of Contents

1. [Overview](#overview)
2. [Core Architecture](#core-architecture)
3. [Project Directory Structure](#project-directory-structure)
4. [Installation & Setup](#installation--setup)
5. [Configuration](#configuration)
6. [How It Works](#how-it-works)
7. [API Endpoints](#api-endpoints)
8. [Technology Stack](#technology-stack)
9. [Running the Application](#running-the-application)
10. [Key Features](#key-features)

---

## Overview

**BillPro Multi-Docs Chatbot** is a production-ready RAG system that answers questions by retrieving relevant document chunks and generating context-aware responses using LLMs. It's specifically optimized for electricity tariff documents and regulatory PDFs.

### Key Capabilities
- 📄 Process **100+ multi-format documents** (~10,000 pages)
- 🔍 **Hybrid retrieval**: BM25 + Dense vectors + Knowledge graphs
- 🤖 **Streaming responses** with real-time citations
- 💬 **Multi-user support** with persistent chat history
- 🎯 **Context-aware chunking** that respects semantic boundaries
- 📊 **Web interface** with document preview and source tracking

---

## Core Architecture

### Four-Stage Retrieval Pipeline

The system employs a sophisticated retrieval strategy to ensure accuracy and relevance:

```
┌─────────────────────── Stage 1: Parallel Retrieval ──────────────────────┐
│                                                                            │
│  Query Input                                                              │
│    │                                                                       │
│    ├─→ BM25 Search          → Top 40% by score                          │
│    ├─→ Dense Embeddings     → Qdrant vectors (similarity > 0.6)         │
│    └─→ Knowledge Graph      → Entity extraction + graph traversal       │
│                                                                            │
│    All return MAX_CANDIDATES_PER_RETRIEVER (100) candidates             │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                   ┌───────────────┴───────────────┐
                   │ Stage 2: Reciprocal Rank      │
                   │ Fusion (RRF)                  │
                   │ Combine rank scores           │
                   └───────────────┬───────────────┘
                                   │
                   ┌───────────────┴───────────────┐
                   │ Stage 3: Cross-Encoder       │
                   │ Reranking                     │
                   │ Score each candidate with     │
                   │ semantic relevance            │
                   └───────────────┬───────────────┘
                                   │
                   ┌───────────────┴────────────────┐
                   │ Stage 4: Final Selection       │
                   │ FINAL_TOP_K = 30 documents   │
                   └───────────────┬────────────────┘
                                   │
                                   ↓
                           Context + Citations
```

### Component Breakdown

#### **Stage 1a: BM25 (Sparse Text Search)**
- **Algorithm**: BM25Okapi from `rank_bm25` library
- **Tokenization**: Custom lowercase + punctuation removal
- **Threshold**: Returns documents where score ≥ `max_score × BM25_SCORE_RATIO (0.4)`
- **Use Case**: Excellent for keyword-heavy queries and exact term matching

#### **Stage 1b: Dense Vector Search (Qdrant)**
- **Embedding Model**: `mxbai-embed-large` (768-dimensional vectors)
- **Vector Store**: Qdrant (high-performance HNSW index)
- **Similarity Metric**: Cosine distance
- **Threshold**: Only returns vectors with similarity ≥ 0.6
- **Use Case**: Semantic similarity and meaning-based retrieval

#### **Stage 1c: Knowledge Graph**
- **Entity Extraction**: spaCy NER (Named Entity Recognition)
- **Graph Structure**:
  - Nodes: Named entities + document chunks
  - Edges: Co-occurrence relations and entity mentions
- **Retrieval**: Entity-based graph traversal (3-hop depth)
- **Threshold**: Score ≥ 0.4
- **Use Case**: Relationship-driven queries and entity linking

#### **Stage 2: Reciprocal Rank Fusion (RRF)**
- **Formula**: Combines BM25, dense, and knowledge graph rankings
- **Benefit**: Single, normalized score integrating three independent signals
- **Output**: Candidate pool (typically 50-150 docs)

#### **Stage 3: Cross-Encoder Reranking**
- **Model**: `cross-encoder/ms-marco-MiniLM-L-12-v2`
- **Task**: Learn-to-rank with human preference data
- **Threshold**: Filters out low-confidence matches (≥ 0.1)
- **Benefit**: Semantic relevance scoring using bidirectional attention

#### **Stage 4: Final Selection**
- Returns top `FINAL_TOP_K = 30` documents
- Each document includes source, page, and citation info

---

## Project Directory Structure

```
billpro_multi_docs_chatbot/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Project metadata
├── .env                               # Environment variables (REQUIRED)
│
├── 📁 config/                         # Configuration management
│   ├── __init__.py
│   └── settings.py                    # Pydantic BaseSettings for env vars
│
├── 📁 llm_clients/                    # LLM integration
│   ├── __init__.py
│   └── ollama_client.py              # Async Ollama API wrapper
│
├── 📁 retriever/                      # Core retrieval pipeline
│   ├── __init__.py
│   ├── vector.py                      # HybridRetriever (4-stage pipeline)
│   └── knowledge_graph.py             # Entity extraction + graph building
│
├── 📁 templates/                      # Web UI templates
│   └── index.html                     # Main chatbot interface
│
├── 📁 static/                         # Frontend assets
│   ├── style.css                      # UI styling
│   └── script.js                      # Client-side logic & Socket.IO
│
├── 📁 test_documents/                 # Document corpus (~100 PDFs)
│   ├── AEML.pdf
│   ├── APCPDCL.pdf
│   ├── AVVNL.pdf
│   └── ... (70+ electricity tariff PDFs)
│
├── 📁 chat_histories/                 # Persistent chat storage
│   ├── session_id_1.json
│   ├── session_id_2.json
│   └── ... (per-user message history)
│
├── 📁 hybrid_db/                      # Vectorization & search indices
│   ├── file_hashes.json              # SHA256 hashes (change detection)
│   ├── bm25_index.pkl                # BM25 index + corpus
│   ├── knowledge_graph.pkl           # spaCy NER graph
│   └── kg_entity_index.json          # Entity name → node ID mapping
│
├── 📁 logs/                           # Performance & debug logs
│   ├── log01042026.txt               # Day-based log rotation
│   ├── log02042026.txt
│   └── ... (one per day)
│
├── main.py                            # Placeholder entry point
├── server.py                          # FastAPI + Socket.IO web server ⭐
├── script.py                          # LangChain pipelines & prompts
├── data_loader.py                     # Document parsing & chunking
├── direct_embeddings.py               # Utility for manual embedding
│
├── chunk.txt                          # Last retrieved context (debug)
├── vector.txt                         # Debug file
├── knowledge.txt                      # Debug file
├── improvement.md                     # Development notes
└── .venv/                             # Virtual environment
```

### Key Directories Explained

#### **`retriever/`** - The Heart of the System
- **`vector.py`**: Implements `HybridRetriever` dataclass
  - Orchestrates all 4 retrieval stages
  - Manages BM25, Qdrant, and knowledge graph integration
  - Handles LLM streaming and response formatting

- **`knowledge_graph.py`**: Entity-based retrieval
  - Builds a NetworkX directed graph from document chunks
  - Performs Named Entity Recognition (NER) via spaCy
  - Implements graph traversal for entity-driven retrieval

#### **`config/`** - Environment & Settings
- Single source of truth for all configuration
- Reads from `.env` file using Pydantic
- Required variables: `LLM_MODEL_NAME`, `QDRANT_URL`, `DOCUMENTS_DIR`, etc.

#### **`llm_clients/`** - LLM Communication
- Async HTTP wrapper around Ollama API
- Supports streaming responses
- Handles image encoding for vision models

#### **`chat_histories/`** - Persistent State
- Per-session JSON files (one per user)
- Stores message history with role/content/metadata
- Automatically truncated to last 6 messages for context
- Trimmed to 300 chars per message for efficiency

#### **`hybrid_db/`** - Vector & Index Storage
- BM25 index pickled for fast loading
- Qdrant collections stored in local database
- Knowledge graph serialized as NetworkX pickle
- File hashes tracked to detect document changes

#### **`test_documents/`** - Document Corpus
- 100+ electricity tariff PDFs from Indian power distribution companies
- Each document processed through semantic chunking pipeline
- Stored vectors, BM25 indices, and entity graphs

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Ollama installed and running locally (default: `http://localhost:11434`)
- Qdrant vector database running (default: `http://localhost:6333`)

### Step 1: Clone & Install Dependencies
```bash
git clone <repo-url>
cd billpro_multi_docs_chatbot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Set Up Ollama & Qdrant

**Ollama** (for embeddings and LLM):
```bash
# Install from https://ollama.ai
# Pull required models
ollama pull mxbai-embed-large      # Embedding model
ollama pull llama2                 # LLM model (or your choice)
ollama serve                       # Runs on localhost:11434
```

**Qdrant** (for vector storage):
```bash
# Option 1: Docker
docker run -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest

# Option 2: Download from https://qdrant.tech/documentation/quick-start/
```

### Step 3: Configure Environment Variables
Create a `.env` file in the project root:

```env
# LLM Configuration
LLM_MODEL_NAME=llama2                          # Or mistral, neural-chat, etc.
LLM_ENDPOINT=http://localhost:11434
LLM_EMBEDDING_MODEL=mxbai-embed-large
CONTEXT_MODEL=cross-encoder/ms-marco-minilm-l-12-v2

# Database & Storage
QDRANT_URL=http://localhost:6333
DOCUMENTS_DIR=./test_documents
HISTORY_DIR=./chat_histories

# Reranking Model
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2  # For semantic reranking
```

### Step 4: Download spaCy Models
```bash
python -m spacy download en_core_web_md
# For higher accuracy, use: en_core_web_trf (requires more compute)
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL_NAME` | - | Ollama model name (e.g., llama2, mistral) |
| `LLM_ENDPOINT` | `http://localhost:11434` | Ollama API base URL |
| `LLM_EMBEDDING_MODEL` | `mxbai-embed-large` | Embedding model (768-dim vectors) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant vector DB URL |
| `DOCUMENTS_DIR` | `./test_documents` | Path to documents folder |
| `HISTORY_DIR` | `./chat_histories` | Where to store chat files |
| `CROSS_ENCODER_MODEL` | - | Reranker model name |

### Retrieval Tuning Parameters (`retriever/vector.py`)

```python
# Stage 1 thresholds
BM25_SCORE_RATIO = 0.4                      # Return top 40% by BM25 score
VECTOR_SIMILARITY_THRESHOLD = 0.6           # Min cosine similarity
GRAPH_SCORE_THRESHOLD = 0.4                 # Min graph relevance

# Stage 2
MAX_CANDIDATES_PER_RETRIEVER = 100          # Per retriever before fusion

# Stage 3
RERANKER_THRESHOLD = 0.1                    # Min cross-encoder score

# Stage 4
FINAL_TOP_K = 30                            # Final documents returned
```

### Chunking Parameters (`data_loader.py`)

```python
MIN_CHUNK_CHARS = 400                       # Minimum chunk size
BASE_CHUNK_CHARS = 1_200                    # Target size
MAX_CHUNK_CHARS = 2_000                     # Hard limit
BOUNDARY_THRESHOLD = 0.35                   # TF-IDF distance threshold
BRIDGE_SENTENCES = 1                        # Overlap sentences
```

---

## How It Works

### 1. **Document Ingestion Pipeline**

```
Raw Document (.pdf, .docx, .txt)
        ↓
    [data_loader.py]
        ├─ PDF → PyMuPDF (fitz) for text extraction
        ├─ DOCX → python-docx for structured parsing
        ├─ TXT → Direct read
        └─ OCR fallback (Tesseract) for scanned PDFs
        ↓
    Semantic-Aware Chunking
        ├─ Sentence tokenization with regex protection
        ├─ TF-IDF semantic boundary detection
        ├─ Dynamic sizing based on content density
        └─ Cross-boundary sentence bridges
        ↓
    Chunk Metadata Enhancement
        ├─ Chunk ID (SHA256 hash)
        ├─ Source file path
        ├─ Page number
        ├─ Section header (for hierarchical docs)
        └─ Content hash (change detection)
        ↓
    Persistence
        ├─ BM25 index created → `hybrid_db/bm25_index.pkl`
        ├─ Vectors → Qdrant collection
        ├─ Knowledge graph → `hybrid_db/knowledge_graph.pkl`
        └─ Entity index → `hybrid_db/kg_entity_index.json`
```

**Key Classes**:
- `SemanticAwareTextSplitter`: Intelligent chunking respecting topic boundaries
- `MultiFormatDocumentLoader`: Unified interface for PDF/DOCX/TXT
- Document tracking prevents re-processing unchanged files

### 2. **Query Processing Workflow**

```
User Question
        ↓
    [server.py - handle_chat_request]
        ├─ Load session history (last 6 messages, 300 chars each)
        ├─ Call condense_chain to rewrite multi-part questions
        │  (breaks "Compare A and B" → ["What is A?", "What is B?"])
        └─ Result: List of standalone questions
        ↓
    [retriever/vector.py - HybridRetriever._retrieve]
        ├─ Stage 1: Parallel retrieval (async)
        │   ├─ BM25._bm25_search(q) → 100 docs
        │   ├─ Dense._dense_search(q) → 100 docs (Qdrant)
        │   └─ Graph._graph_search(q) → 100 docs (entity-driven)
        │
        ├─ Stage 2: Reciprocal Rank Fusion
        │   └─ Combine rankings → score each doc by all 3 methods
        │
        ├─ Stage 3: Cross-Encoder Reranking
        │   ├─ Load cross-encoder model
        │   ├─ Score query-document pairs
        │   └─ Filter by RERANKER_THRESHOLD
        │
        └─ Stage 4: Final Selection
            └─ Return FINAL_TOP_K (30) docs with metadata
        ↓
    [script.py - format_documents]
        ├─ Build citation index: [1] source_file (page X)
        ├─ Concatenate all doc chunks
        └─ Truncate to MAX_CONTEXT_CHARS (30,000)
        ↓
    [server.py - run_llm_logic]
        ├─ Build prompt with context
        ├─ Stream response from LLM (Ollama)
        ├─ Extract inline citations [1], [2], etc.
        ├─ Filter citations by actual usage in response
        └─ Return streaming chunks + final citations
        ↓
    Socket.IO Emit (to frontend)
        ├─ "chat_stream_chunk" (streaming text)
        ├─ "chat_stream_end" (completion + citations)
        └─ "error" (if failed)
```

### 3. **Prompt Architecture**

#### **Condense Chain** (`script.py`)
Rewrites follow-up questions to be standalone:
```
Input:  "What about commercial tariff?"
Context: Previous messages about "residential tariff for Company A"
Output: ["What is the commercial tariff for Company A?"]
```

#### **Main RAG Chain** (`script.py`)
Generates answer with structured output:
```markdown
### Billpro Bot
[Your answer with inline citations like [1].]

---

### Key Takeaways
- Bullet 1 [2]
- Bullet 2 [3]
```

---

## API Endpoints

### WebSocket Events (Socket.IO)

#### **Client → Server**

- **`chat_request`** (data: dict)
  - Payload: `{"message": "user question"}`
  - Emits response stream and citations

  ```python
  emit('chat_request', {"message": "What is the tariff for Delhi?"},
       namespace='/socket.io/')
  ```

- **`stop_generation`**
  - Cancels ongoing LLM generation
  - Server emits `chat_stopped_manually`

#### **Server → Client**

- **`chat_stream_chunk`** (chunk: str)
  - Streamed response text in real-time

- **`chat_stream_end`** (message, citation: list)
  - Final completion signal
  - Includes citation array:
    ```json
    [{
      "id": 1,
      "display_name": "AEML.pdf",
      "file_path": "./test_documents/AEML.pdf",
      "page": 5,
      "rows": "N/A",
      "topic": "Commercial tariff rates..."
    }]
    ```

- **`error`** (message: str)
  - Error description

### REST Endpoints (FastAPI)

#### **GET `/`**
Returns `index.html` (web UI)

#### **GET `/v1/history/{session_id}`**
Returns chat message history for a session (debugging)
```bash
curl http://localhost:8010/v1/history/abc123xyz
```

#### **GET `/v1/document-page?source=<path>&page=<int>`**
Renders a document page as PNG image
- Supports: PDF, DOCX, PNG, JPG
- Auto-converts DOCX → PDF via LibreOffice
- Returns 2x zoom for readability

```bash
# Example: Get page 3 of AEML.pdf
curl "http://localhost:8010/v1/document-page?source=./test_documents/AEML.pdf&page=3" \
  > page3.png
```

---

## Technology Stack

### Core Framework
- **FastAPI** (0.115.7) - Async web framework
- **uvicorn** (0.34.2) - ASGI server
- **Socket.IO** (5.13.0) - Real-time bidirectional communication

### LLM & Embeddings
- **Ollama** (client: 0.5.1) - Local LLM & embedding inference
- **LangChain** (0.3.27) - LLM orchestration
  - `langchain-ollama` (0.3.6)
  - `langchain-qdrant` - Vector store integration
  - `langchain-core` (0.3.72) - Chains & prompts

### Vector Search
- **Qdrant** (client: 1.14.3) - HNSW vector database
- **sentence-transformers** (4.1.0) - Cross-encoder reranking

### Document Processing
- **PyMuPDF** (1.20.1, `fitz`) - PDF extraction
- **python-docx** (1.2.0) - DOCX parsing
- **pdf2image** (1.17.1) - PDF → PNG conversion
- **Tesseract** (pytesseract) - OCR fallback
- **PIL** (Pillow 11.3.0) - Image manipulation

### Search & Indexing
- **rank_bm25** (0.2.2) - BM25Okapi sparse search
- **NetworkX** (3.5) - Knowledge graph management
- **spaCy** (3.7.x) - Named Entity Recognition (NER)

### Data Processing
- **pandas** (2.2.3) - Table handling
- **numpy** (1.26.4) - Numerical operations
- **scikit-learn** (1.7.1) - TF-IDF for semantic boundaries

### Utilities
- **pydantic** (2.11.7) - Configuration validation
- **loguru** (0.7.3) - Logging
- **tqdm** (4.67.1) - Progress bars
- **asyncio** + **aiohttp** - Async HTTP

---

## Running the Application

### Quick Start

```bash
# 1. Ensure Ollama & Qdrant are running
ollama serve                    # Terminal 1
docker run -p 6333:6333 qdrant/qdrant  # Terminal 2 (or equivalent)

# 2. Activate environment
source .venv/bin/activate

# 3. Start the server
python server.py                # Production mode on 0.0.0.0:8010

# Development mode (hot reload, verbose logging)
python server.py dev            # Dev mode on 0.0.0.0:8100
```

### Access the UI
- Open browser: `http://localhost:8010`
- Type questions, see real-time streaming responses
- Click citations to view source documents

### Processing Documents (One-time Setup)

The system auto-loads documents on startup from `DOCUMENTS_DIR`. To manually process documents:

```python
# Direct embedding script
python direct_embeddings.py     # Processes all docs in DOCUMENTS_DIR
```

This will:
1. Scan new/modified files (hash-based change detection)
2. Parse text via semantic chunking
3. Generate embeddings for Qdrant
4. Build BM25 index
5. Extract entities for knowledge graph
6. Save indices to `hybrid_db/`

---

## Key Features

### ✅ Hybrid Retrieval (Four-Stage Pipeline)
- **BM25**: Fast, keyword-based retrieval
- **Dense vectors**: Semantic meaning matching
- **Knowledge graphs**: Entity relationships
- **Cross-encoder reranking**: Learned-to-rank scoring

### ✅ Context-Aware Chunking
- **Semantic boundaries**: Respects topic shifts (TF-IDF)
- **Dynamic sizing**: Adjusts chunk size by content density
- **Sentence bridges**: Maintains cross-chunk context
- **Robust**: Handles abbreviations, decimals, ellipses

### ✅ Multi-Format Document Support
- ✓ PDF (text + scanned/OCR fallback)
- ✓ DOCX (structured parsing)
- ✓ TXT (plain text)
- **Metadata**: Page numbers, table detection, section headers

### ✅ Streaming Responses
- Real-time text generation via Ollama
- Per-chunk Socket.IO emission
- Inline citations with source tracking
- Graceful stop signal handling

### ✅ Multi-User Architecture
- **Per-session chat history** (file-based JSON)
- **Session isolation** (in-memory history cache)
- **Message truncation** (last 6 msgs, 300 chars each)
- **Concurrent request handling** (async/await)

### ✅ Comprehensive Logging
- **Daily log rotation** (`logs/log{DDMMYYYY}.txt`)
- **Performance timing** (stage-by-stage)
- **Debug output** (for retrieval pipeline)

### ✅ Document Visualization
- **Interactive preview**: Click citations to view pages
- **Image rendering**: 2x zoom for readability
- **Format support**: PDF, DOCX, PNG, JPG

---

## Example Workflow

### User asks: "Compare commercial tariffs between Delhi and Mumbai"

**1. Condense Chain**
```
Input:  "Compare commercial tariffs between Delhi and Mumbai"
Output: [
  "What are commercial tariffs in Delhi?",
  "What are commercial tariffs in Mumbai?"
]
```

**2. Parallel Retrieval (for each question)**
```
Query 1: "What are commercial tariffs in Delhi?"
├─ BM25 Search     → [Delhi tariff doc 1, Delhi tariff doc 2, ...]
├─ Dense Search    → [doc 3, doc 5, ...]
└─ Graph Search    → [doc 1, doc 7, ...]   (via "Delhi" entity)

Query 2: "What are commercial tariffs in Mumbai?"
├─ BM25 Search     → [Mumbai tariff doc, ...]
├─ Dense Search    → [doc 2, doc 8, ...]
└─ Graph Search    → [doc 2, doc 6, ...]   (via "Mumbai" entity)
```

**3. RRF Fusion**
```
Merged & ranked by normalized scores from all 3 retrievers
→ Top 50-100 candidate docs
```

**4. Cross-Encoder Reranking**
```
Score each candidate pair: (query, doc_chunk)
filter by score ≥ 0.1
→ Top 30 docs
```

**5. Context Formatting**
```
[1] Delhi commercial tariff: ₹8.50/kWh (Source: TPDDL.pdf, page 12)
[2] Additional charges for Delhi: ₹0.50/kWh (TPDDL.pdf, page 13)
[3] Mumbai commercial tariff: ₹7.80/kWh (Source: MSEDCL.pdf, page 8)
...
```

**6. LLM Generation (Ollama, streaming)**
```
### Billpro Bot
Delhi's commercial tariff stands at ₹8.50/kWh [1] with an additional
charge of ₹0.50/kWh [2], bringing the total to ₹9.00/kWh. In comparison,
Mumbai's commercial tariff is ₹7.80/kWh [3], making it approximately
13.6% cheaper than Delhi...

### Key Takeaways
- Delhi: ₹9.00/kWh total [1][2]
- Mumbai: ₹7.80/kWh [3]
- Difference: ₹1.20/kWh (Delhi higher) [1][3]
```

**7. Citation Extraction**
```json
[
  {"id": 1, "display_name": "TPDDL.pdf", "page": 12, "topic": "Delhi commercial tariff..."},
  {"id": 2, "display_name": "TPDDL.pdf", "page": 13, "topic": "Additional charges..."},
  {"id": 3, "display_name": "MSEDCL.pdf", "page": 8, "topic": "Mumbai commercial tariff..."}
]
```

---

## Performance Metrics

| Operation | Time | Notes |
|-----------|------|-------|
| BM25 Search (100 candidates) | ~50ms | Single-threaded |
| Dense Vector Search (Qdrant) | ~100ms | Includes embedding |
| Graph Search (entity extraction) | ~200ms | spaCy NER + traversal |
| RRF Fusion | ~10ms | Simple ranking combination |
| Cross-Encoder Reranking | ~500ms | 30 doc pairs scoring |
| **Total Retrieval** | **~860ms** | For typical query |
| LLM Generation (streaming) | ~2-5s | Per response (model-dependent) |
| **E2E User Response** | **~3-6s** | Including overhead |

---

## Troubleshooting

### Issue: "Failed to connect to Ollama"
```
Solution: Ensure Ollama is running
$ ollama serve
```

### Issue: "Qdrant connection refused"
```
Solution: Start Qdrant
$ docker run -p 6333:6333 qdrant/qdrant
```

### Issue: Document not appearing in search
```
Solution: Check file hash & manual reprocess
1. Delete hybrid_db/file_hashes.json
2. Run: python direct_embeddings.py
3. Restart server
```

### Issue: Slow queries
```
Solution: Tune thresholds in retriever/vector.py
- Increase BM25_SCORE_RATIO to get fewer BM25 results
- Increase VECTOR_SIMILARITY_THRESHOLD for tighter filtering
- Reduce FINAL_TOP_K if not needing 30 docs
```

---

## Future Enhancements

- [ ] Add RAG evaluation metrics (BLEU, ROUGE, F1)
- [ ] Implement parent document retriever (retrieve parent → cite specific child)
- [ ] Multi-language support (auto-detect + translate)
- [ ] Fact verification (cross-check LLM output against docs)
- [ ] Analytics dashboard (query popularity, success rates)
- [ ] Fine-tuned cross-encoder models for domain
- [ ] Hybrid BM25 + semantic query expansion

---

## License

[Add your license here]

---

## Contact & Support

For issues, feature requests, or questions:
- 📧 Email: support@billpro.io
- 🐛 GitHub Issues: [repo-link]/issues
- 💬 Discussions: [repo-link]/discussions
