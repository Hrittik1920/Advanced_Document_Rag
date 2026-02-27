import os
import json
import hashlib
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from data_loader import MultiFormatDocumentLoader
from tqdm import tqdm # Import tqdm for the progress bar

# --- Configuration ---
DOCUMENTS_DIRECTORY = "./test_documents"
DB_LOCATION = "./chroma_langchain_db"
FILE_HASH_DB = os.path.join(DB_LOCATION, "file_hashes.json")
EMBEDDINGS = OllamaEmbeddings(model="mxbai-embed-large")
COLLECTION_NAME = "multi_format_documents"

def get_file_hash(file_path):
    hasher = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        return hasher.hexdigest()
    except (IOError, FileNotFoundError):
        return None

def load_file_hashes():
    if os.path.exists(FILE_HASH_DB):
        with open(FILE_HASH_DB, 'r') as f:
            return json.load(f)
    return {}

def save_file_hashes(hashes):
    os.makedirs(DB_LOCATION, exist_ok=True)
    with open(FILE_HASH_DB, 'w') as f:
        json.dump(hashes, f, indent=4)

def initialize_vector_store():
    print("Initializing vector store...")
    loader = MultiFormatDocumentLoader()
    file_hashes = load_file_hashes()
    
    docs_to_add = []
    current_files = set()
    updated_hashes = file_hashes.copy()

    # Scan directory for new or modified files
    all_files_in_dir = []
    for root, _, files in os.walk(DOCUMENTS_DIRECTORY):
        for fname in files:
            if not fname.startswith('.'): # ignore hidden files
                 all_files_in_dir.append(os.path.join(root, fname))

    print(f"Found {len(all_files_in_dir)} files to check...")
    for fpath in tqdm(all_files_in_dir, desc="Checking file status"):
        current_files.add(fpath)
        new_hash = get_file_hash(fpath)
        if new_hash and file_hashes.get(fpath) != new_hash:
            print(f"\nDetected change in: {fpath}. Loading...")
            docs_to_add.extend(loader.load_document(fpath))
            updated_hashes[fpath] = new_hash

    deleted_files = set(file_hashes.keys()) - current_files
    if deleted_files:
        print(f"Detected {len(deleted_files)} deleted files. A full rebuild is needed to remove them.")
        for fpath in deleted_files:
            del updated_hashes[fpath]

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_LOCATION,
        embedding_function=EMBEDDINGS
    )

    if docs_to_add:
        print(f"\nFound {len(docs_to_add)} new document chunks to process.")
        # Add documents in batches to be more efficient
        batch_size = 100 
        for i in tqdm(range(0, len(docs_to_add), batch_size), desc="Embedding documents"):
            batch = docs_to_add[i:i+batch_size]
            doc_ids = [hashlib.sha256(f"{doc.metadata['source']}{doc.metadata.get('chunk_number', 0)}{doc.page_content}".encode()).hexdigest() for doc in batch]
            vector_store.add_documents(documents=batch, ids=doc_ids)
        
        save_file_hashes(updated_hashes)
        print("Documents embedded successfully!")
    else:
        print("Vector store is up to date. No new documents to add.")

    return vector_store

# --- Main Execution ---
vector_store = initialize_vector_store()
retriever = vector_store.as_retriever(
    search_kwargs={"k": 8}
)

print("Retriever is ready.")