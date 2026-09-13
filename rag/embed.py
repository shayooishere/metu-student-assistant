"""
METU Student Assistant — Embeddings + Vector DB (Step 3 of the RAG pipeline)

Short Summary:
-----------------------------------------
1. Reads text chunks from data/processed/chunks.jsonl
2. Converts each chunk into a vector using BAAI BGE-M3 embedding model
3. Stores vectors + text + metadata in Chroma vector database locally

"""

import json
import shutil
import sys
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from rag.hybrid import build_bm25_index, save_bm25_index

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

INPUT_FILE = PROJECT_FOLDER / "data" / "processed" / "chunks.jsonl"
CHROMA_FOLDER = PROJECT_FOLDER / "chroma_db"

# HuggingFace model ID 
EMBEDDING_MODEL = "BAAI/bge-m3"

# Name of the collection inside Chroma (like a table name in a database)
COLLECTION_NAME = "metu_student_assistant"

# Process this many chunks at a time when embedding (lower = less RAM usage)
EMBED_BATCH_SIZE = 32


# ---------------------------------------------------------------------------
# STEP 1 — Load chunks.jsonl
# ---------------------------------------------------------------------------

def load_chunks(file_path):
    """
    Read chunks.jsonl and return a list of dicts.

    Each dict has: chunk_id, text, url, title, section, language, etc.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}\n"
            "Run chunking first"
        )

    chunks = []
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))

    return chunks


# ---------------------------------------------------------------------------
# STEP 2 — Convert chunks to LangChain Documents
# ---------------------------------------------------------------------------

def clean_metadata_value(value):
    """
    Chroma only accepts metadata values that are str, int, float, or bool.
    Convert None to empty string so we don't get errors.
    """
    if value is None:
        return ""
    return value


def chunks_to_documents(chunks):
    """
    Turn each chunk dict into a LangChain Document.

    Document = page_content (the text) + metadata (url, title, etc.)
    """
    documents = []

    for chunk in chunks:
        metadata = {
            "chunk_id": clean_metadata_value(chunk.get("chunk_id")),
            "chunk_index": int(chunk.get("chunk_index", 0)),
            "url": clean_metadata_value(chunk.get("url")),
            "title": clean_metadata_value(chunk.get("title")),
            "section": clean_metadata_value(chunk.get("section")),
            "language": clean_metadata_value(chunk.get("language")),
            "source_id": clean_metadata_value(chunk.get("source_id")),
            "header_1": clean_metadata_value(chunk.get("header_1")),
            "header_2": clean_metadata_value(chunk.get("header_2")),
            "header_3": clean_metadata_value(chunk.get("header_3")),
            "header_4": clean_metadata_value(chunk.get("header_4")),
            "header_5": clean_metadata_value(chunk.get("header_5")),
        }

        documents.append(
            Document(
                page_content=chunk["text"],
                metadata=metadata,
            )
        )

    return documents


# ---------------------------------------------------------------------------
# STEP 3 — Create the embedding model
# ---------------------------------------------------------------------------

def create_embedding_model():
    """
    Load BGE-M3 from HuggingFace.

    normalize_embeddings=True makes vector length consistent,
    which improves similarity search quality.

    device='cpu' uses your laptop CPU.
    """
    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": EMBED_BATCH_SIZE,
        },
    )


# ---------------------------------------------------------------------------
# STEP 4 — Build Chroma vector database
# ---------------------------------------------------------------------------

def build_vector_database(documents, embeddings, chroma_folder):
    """
    Embed all documents and save them to Chroma on disk.

    Chroma stores:
      - the original text (page_content)
      - metadata (url, title, ...)
      - the embedding vector (for similarity search)

    If chroma_folder already exists, we delete it first so re-running
    this script gives a fresh index from the latest chunks.jsonl.
    """
    if chroma_folder.exists():
        print(f"Removing old database at: {chroma_folder}")
        shutil.rmtree(chroma_folder)

    print(f"Embedding {len(documents)} chunks and saving to Chroma...")

    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(chroma_folder),
    )

    return vector_db


# ---------------------------------------------------------------------------
# STEP 5 — Quick test search (optional sanity check)
# ---------------------------------------------------------------------------

def test_search(vector_db, query):
    """
    Run one test query to verify the database works.

    Returns the top matching chunk so you can see if retrieval makes sense.
    """
    results = vector_db.similarity_search(query, k=1)

    if not results:
        return None

    top = results[0]
    return {
        "query": query,
        "title": top.metadata.get("title"),
        "url": top.metadata.get("url"),
        "preview": top.page_content[:200] + "...",
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("METU Student Assistant - Embeddings (Step 3)")
    print("=" * 60)
    print(f"\nInput:  {INPUT_FILE}")
    print(f"Output: {CHROMA_FOLDER}\n")

    # Load chunks
    chunks = load_chunks(INPUT_FILE)
    print(f"Loaded {len(chunks)} chunks from chunks.jsonl")

    documents = chunks_to_documents(chunks)
    print(f"Converted to {len(documents)} LangChain documents\n")

    # Create embeddings model and vector DB
    embeddings = create_embedding_model()
    vector_db = build_vector_database(documents, embeddings, CHROMA_FOLDER)

    # Build BM25 keyword index (used with Chroma for hybrid search)
    print("\nBuilding BM25 keyword index...")
    bm25_payload = build_bm25_index(documents)
    bm25_path = save_bm25_index(bm25_payload, CHROMA_FOLDER)
    print(f"BM25 index saved to: {bm25_path}")

    # Sanity check: try one English and one Turkish query
    print("=" * 60)
    print("Sanity check — test retrieval:")
    print("=" * 60)

    test_queries = [
        "How many books can undergraduate students borrow?",
        "Yurt basvurusu nasil yapilir?", #How do I apply for a dormitory?
    ]

    for query in test_queries:
        result = test_search(vector_db, query)
        if result:
            print(f"\nQuery: {result['query']}")
            print(f"Top match: {result['title']}")
            print(f"URL: {result['url']}")
            print(f"Preview: {result['preview']}")

    print("\n" + "=" * 60)
    print("Done! Vector database saved to:")
    print(f"  {CHROMA_FOLDER}")
    print(f"Collection name: {COLLECTION_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    main()
