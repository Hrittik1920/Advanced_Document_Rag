# Multi-Docs Chatbot

Multi-Docs Chatbot is a FastAPI, Socket.IO, LangGraph, and LangChain based RAG application for answering questions over a corpus of electricity tariff documents. It supports multi-document retrieval, citations, document page previews, uploaded bill extraction, and a separate math-agent path for billing calculations.

The application is optimized for Indian DISCOM tariff PDFs and related billing documents, but the ingestion pipeline also supports DOCX, CSV, Excel, TXT, and image files.

## What The System Does

- Indexes documents from `DOCUMENTS_DIR` into a hybrid retrieval store.
- Retrieves relevant chunks with BM25, dense vector search, and a knowledge graph.
- Fuses and reranks retrieved candidates before building the final prompt context.
- Streams answers to a browser UI through Socket.IO.
- Persists per-session chat history on disk.
- Lets users target specific source files with `@file` mentions from the UI.
- Extracts uploaded bill/document text with Surya layout OCR.
- Routes calculation-heavy questions through a generated Python math workflow.
- Runs generated math code in a Docker-hosted sandbox API.
- Returns structured citations that can open page previews through the web app.

## Runtime Architecture

```text
Browser UI
  |
  | Socket.IO: chat_request, stop_generation
  v
server.py
  |
  | optional upload extraction
  | history load/save
  | available document list
  v
main.py LangGraph app
  |
  +--> classify_and_retrieve
  |      |
  |      +--> script.py condense/unified router chains
  |      +--> retriever/vector.py HybridRetriever
  |      +--> script.py format_documents
  |
  +--> text_path
  |      |
  |      +--> tariff QA prompt -> Ollama LLM -> streamed response
  |
  +--> math path, when math_intent=true
         |
         +--> math_generate
         +--> sandbox/tools.py -> Docker sandbox -> sandbox/runner.py
         +--> math_validate
         +--> synthesize_response
```

## Core Modules

| Path | Responsibility |
| --- | --- |
| `server.py` | FastAPI app, Socket.IO events, static UI serving, history persistence, upload handling, document preview endpoints. |
| `main.py` | LangGraph orchestration. Decides text vs math path and coordinates retrieval, code execution, validation, and final response synthesis. |
| `script.py` | Ollama models, prompts, LangChain chains, query rewriting, document routing, HYDE generation, citation formatting. |
| `retriever/vector.py` | Hybrid retriever initialization, document change detection, Qdrant indexing, BM25 persistence, RRF fusion, cross-encoder reranking. |
| `retriever/knowledge_graph.py` | spaCy/NetworkX entity graph builder and graph retriever. |
| `extraction/data_loader.py` | Multi-format document ingestion, table extraction, OCR fallback, token-aware chunking, metadata generation. |
| `extraction/upload_extraction.py` | Surya layout/OCR extractor for uploaded PDFs and images. |
| `sandbox/runner.py` | Restricted execution service for generated Python calculations. |
| `sandbox/tools.py` | Async client that sends generated code to the sandbox API. |
| `config/settings.py` | Pydantic settings loaded from `.env`. |
| `templates/index.html` | Main browser chat UI, file upload, mentions, citation preview modal. |
| `static/` | Additional frontend CSS/JS assets. |
| `run_tests.py` | Diagnostic runner for retrieval, generation, and citation quality. |

## Directory Layout

```text
billpro_multi_docs_chatbot/
|-- main.py
|-- server.py
|-- script.py
|-- run_tests.py
|-- pyproject.toml
|-- requirements.txt
|-- config/
|   `-- settings.py
|-- extraction/
|   |-- data_loader.py
|   `-- upload_extraction.py
|-- retriever/
|   |-- models.py
|   |-- vector.py
|   `-- knowledge_graph.py
|-- llm_clients/
|   `-- ollama_client.py
|-- sandbox/
|   |-- Dockerfile
|   |-- runner.py
|   `-- tools.py
|-- templates/
|   `-- index.html
|-- static/
|   |-- script.js
|   `-- style.css
|-- test_documents/
|-- hybrid_db/
|-- chat_histories/
`-- logs/
```

Generated/debug files such as `chunk.txt`, `vector.txt`, `upload_content.txt`, and files under `logs/`, `hybrid_db/`, and `chat_histories/` are written during runtime.

## Configuration

The app reads required settings from `.env` through `config/settings.py`.

```env
LLM_MODEL_NAME=your-ollama-chat-model
LLM_ENDPOINT=http://localhost:11434
LLM_EMBEDDING_MODEL=your-ollama-embedding-model
DOCUMENTS_DIR=test_documents
HISTORY_DIR=chat_histories
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
QDRANT_URL=http://localhost:6333
CONTEXT_MODEL=your-context-generation-model
COLLECTION_NAME=billpro_documents
```

Optional:

```env
DEBUG_HISTORY_ENDPOINT=false
```

Set `DEBUG_HISTORY_ENDPOINT=true` only for local debugging, because `/v1/history/{session_id}` exposes stored conversation history.

## External Services

The app expects these services to be reachable:

- Ollama at `LLM_ENDPOINT` for chat, query rewriting, contextual chunk summaries, embeddings, and optional vision/OCR fallback.
- Qdrant at `QDRANT_URL` for dense vector storage.
- Docker sandbox API at `http://localhost:9999/run` for generated Python execution.

The sandbox is built from `sandbox/Dockerfile` and runs `sandbox/runner.py` as a FastAPI service.

## Setup

This project targets Python `>=3.12,<3.13`.

```bash
uv sync
```

Or, with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install or pull the Ollama models referenced in `.env`, then start Ollama and Qdrant.

Build and run the calculation sandbox:

```bash
docker build -t billpro-sandbox ./sandbox
docker run --rm -p 9999:9999 billpro-sandbox
```

Run the app:

```bash
python server.py
```

Default runtime:

- Production-style: `python server.py` starts on port `8010`.
- Development mode: `python server.py dev` starts on port `8100` with reload enabled.

Open:

```text
http://localhost:8010
```

## Startup Workflow

The retriever is initialized at import time through `retriever/__init__.py`, which imports the module-level `retriever` from `retriever/vector.py`.

Startup steps:

1. `server.py` imports `main.app`.
2. `main.py` imports `retriever`.
3. `retriever/vector.py` calls `initialize_retriever()`.
4. Files under `DOCUMENTS_DIR` are scanned.
5. File hashes in `hybrid_db/file_hashes.json` are compared with current files.
6. New or changed files are loaded and chunked with `MultiFormatDocumentLoader`.
7. Dense embeddings are inserted into Qdrant.
8. BM25 corpus is saved in `hybrid_db/bm25_index.pkl`.
9. The knowledge graph is built or updated in `hybrid_db/knowledge_graph.pkl`.
10. The cross-encoder reranker is loaded.
11. The FastAPI and Socket.IO app starts accepting requests.

This means the first startup can be slow if many documents need OCR, contextualization, embedding, or graph indexing. Later startups are faster when hashes and indexes are up to date.

## Document Ingestion Workflow

`extraction/data_loader.py` owns persistent corpus ingestion.

Supported formats:

- PDF
- DOCX
- CSV
- XLS/XLSX
- TXT
- PNG/JPG/JPEG/BMP/TIFF

PDF workflow:

1. Read text and page metadata with PyMuPDF.
2. Skip table-of-contents-like pages.
3. Extract tables with PyMuPDF table detection where possible.
4. Convert tables to compact Markdown with source/section context.
5. Split text with token-aware sentence and section splitting.
6. For scanned PDFs, render pages to images and run OCR.
7. Use VLM fallback when OCR confidence is low or tables are detected.
8. Store chunk metadata such as `source`, `file_type`, `page`, `section`, `rows`, and `chunk_id`.

DOCX, CSV, and Excel files are split into text sections and table chunks. Image files go through OCR and then the same text splitting pipeline.

## Retrieval Workflow

`HybridRetriever` combines three candidate retrievers and a reranking stage.

```text
User retrieval query
  |
  +--> BM25 keyword search
  +--> Qdrant dense similarity search
  +--> Knowledge graph entity search
          |
          v
Reciprocal Rank Fusion
          |
          v
Cross-encoder reranking
          |
          v
Top chunks -> context string + citation objects
```

Retrieval details:

- BM25 uses tokenized sparse search and keeps candidates above a fraction of the best score.
- Dense search uses Ollama embeddings stored in Qdrant.
- Knowledge graph search extracts query entities, expands graph hops, and returns nearby chunks.
- Reciprocal Rank Fusion merges the three ranked lists.
- CrossEncoder reranking sorts candidates by query/document relevance.
- `script.py::format_documents()` builds the final prompt context and citation payload.

If retrieval is weak for short questions, `main.py` can generate a HYDE passage with `script.py::generate_hyde_query()` and retry retrieval.

## Chat Request Workflow

The browser sends `chat_request` through Socket.IO.

Payload shape:

```json
{
  "message": "How is the fixed charge calculated?",
  "target_files": ["MP_EAST.pdf"],
  "file": {
    "name": "uploaded_bill.pdf",
    "data": "<binary payload from browser>"
  }
}
```

Server workflow:

1. `server.py::handle_chat_request()` creates an async task for the session.
2. Uploaded files are written to a temporary file.
3. `SuryaLayoutExtractor` extracts structured text from uploaded PDFs/images.
4. Available corpus file names are loaded from `DOCUMENTS_DIR`.
5. Recent session history is loaded from `chat_histories/{session_id}.json`.
6. Initial LangGraph state is built.
7. `langgraph_app.astream_events()` streams node updates and LLM tokens.
8. Final messages are persisted to history.
9. `chat_stream_end` sends citations back to the UI.

The frontend can also emit `stop_generation`, which cancels the active session task.

## LangGraph Workflow

`main.py` builds this graph:

```text
classify_and_retrieve
  |
  +-- math_intent=false --> text_path --> END
  |
  +-- math_intent=true  --> math_generate
                            |
                            v
                         math_execute
                            |
                            v
                         math_validate
                            |
                            +-- validation passed --> synthesize_response --> END
                            |
                            +-- validation rejected -> text_path -----------> END
```

`classify_and_retrieve` performs:

- Query rewriting with `condense_chain`.
- Uploaded-document-aware routing with `unified_chain`.
- Combination of LLM-selected target files and UI-selected target files.
- Parallel retrieval for each rewritten query.
- Deduplication by page content.
- Optional HYDE fallback.
- Context and citation formatting.

`text_path` uses the main tariff QA prompt and streams the final answer.

## Math Agent Workflow

When `math_intent=true`, the graph uses a calculation workflow:

1. `math_generate` asks the LLM to write a single Python code block.
2. `extract_python_code()` strips Markdown fences.
3. `math_execute` sends code to the sandbox API at `localhost:9999/run`.
4. `sandbox/runner.py` executes code with a restricted builtin set and optional `math`, `numpy`, and `pandas`.
5. `math_validate` asks the LLM to audit the computed result against retrieved tariff context.
6. If accepted or accepted with caution, `synthesize_response` creates the final user-facing answer.
7. If rejected, the graph falls back to `text_path` to answer from context without relying on bad computed numbers.

The sandbox client has a 10 second timeout to avoid hanging the main app.

## Upload Workflow

Uploaded files are handled per request and are not permanently added to the retrieval corpus.

1. The UI sends the uploaded file bytes in the Socket.IO payload.
2. `server.py` writes the bytes to a temporary file.
3. `SuryaLayoutExtractor` loads detection, recognition, and layout predictors.
4. The extractor routes PDFs and images to layout-aware OCR.
5. Extracted text is passed into the LangGraph state as `uploaded_doc_text`.
6. The router chain uses the uploaded text to infer target tariff files.
7. The main answer prompt receives the uploaded text for validation and comparison.
8. The temporary file is removed.

## Citations And Document Preview

Retrieved chunks are formatted as numbered context entries:

```text
[1] chunk text... (Source: MP_EAST.pdf, Page 4)
```

The final citation payload includes:

- `id`
- `display_name`
- `file_path`
- `page`
- `rows`
- `topic`
- `score`
- `source_type`

The frontend renders these under the answer. Clicking a citation calls:

```text
GET /v1/document-page?source=<path-or-file>&page=<zero-based-page>
```

`server.py` restricts previews to files inside `DOCUMENTS_DIR`, then renders PDFs with PyMuPDF. DOC/DOCX preview attempts conversion through LibreOffice. Image files are returned directly.

## API And Socket Events

HTTP endpoints:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Serves `templates/index.html`. |
| `GET` | `/v1/available-files` | Lists visible files in `DOCUMENTS_DIR`. |
| `GET` | `/v1/document-page` | Renders a PDF/DOC/DOCX/image page for citation preview. |
| `GET` | `/v1/history/{session_id}` | Debug-only history endpoint gated by `DEBUG_HISTORY_ENDPOINT=true`. |

Socket.IO events:

| Event | Direction | Description |
| --- | --- | --- |
| `chat_request` | client -> server | Starts a chat request with message, optional file, and optional target files. |
| `chat_stream_chunk` | server -> client | Sends status text or streamed LLM chunks. |
| `chat_stream_end` | server -> client | Ends the stream and includes citations. |
| `stop_generation` | client -> server | Cancels the active task for the Socket.IO session. |
| `chat_stopped_manually` | server -> client | Confirms manual cancellation. |
| `error` | server -> client | Sends request or graph errors. |

## Persistent State

| Location | Purpose |
| --- | --- |
| `hybrid_db/file_hashes.json` | Tracks corpus file hashes for change detection. |
| `hybrid_db/bm25_index.pkl` | Stores BM25 corpus and chunk ids. |
| `hybrid_db/knowledge_graph.pkl` | Stores the NetworkX knowledge graph. |
| `hybrid_db/kg_entity_index.json` | Maps normalized entity names to graph node ids. |
| Qdrant collection | Stores dense vectors and metadata. |
| `chat_histories/*.json` | Stores per-session chat history. |
| `logs/` | Runtime debug and timing logs. |
| `chunk.txt` | Last formatted retrieval context, useful for debugging. |
| `upload_content.txt` | Last uploaded document text, useful for debugging. |
| `vector.txt` | Chunk dump produced during indexing. |

## Running Diagnostics

`run_tests.py` loads questions from `test_questions.json` and evaluates retrieval, generation, and citations.

```bash
python run_tests.py
```

Because importing `retriever` initializes the index, diagnostics require the same `.env`, Ollama, Qdrant, and model availability as the main server.

## Operational Notes

- Startup indexing is eager. Importing `retriever` can trigger document scanning, embedding, KG updates, and reranker loading.
- `retriever/vector.py` stores source metadata as basenames for indexed documents, while file hashes track full paths.
- Uploaded documents affect only the current request. To make a document part of the permanent corpus, place it in `DOCUMENTS_DIR` and restart or reinitialize the app.
- The math path depends on the sandbox container. If `localhost:9999` is unavailable, math requests return a sandbox connection error.
- The knowledge graph requires a spaCy English model. It tries `en_core_web_md` first and falls back to `en_core_web_sm`.
- The app writes debug artifacts in the repo root during normal execution.
- The browser UI has both inline scripts in `templates/index.html` and additional assets under `static/`.

## Common Commands

```bash
# Start the web app
python server.py

# Start in development mode
python server.py dev

# Build the sandbox image
docker build -t billpro-sandbox ./sandbox

# Run the sandbox API
docker run --rm -p 9999:9999 billpro-sandbox

# Run diagnostics
python run_tests.py
```
