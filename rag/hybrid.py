"""
Hybrid retrieval helpers: BM25 (keyword) + dense fusion via RRF.

Used by:
  - rag/embed.py  → build & save the BM25 index next to Chroma
  - rag/engine.py → hybrid search at question time
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# How many candidates each retriever contributes before fusion
CANDIDATE_K = 10

# RRF constant (standard default from the original paper)
RRF_K = 60

BM25_INDEX_NAME = "bm25_index.pkl"


def bm25_index_path(chroma_folder: Path) -> Path:
    return Path(chroma_folder) / BM25_INDEX_NAME


def tokenize(text: str) -> list[str]:
    """Simple tokenizer that keeps English + Turkish letters and digits."""
    text = (text or "").lower()
    # Normalize common add-drop spellings so calendar rows and FAQs align
    text = (
        text.replace("add/drop", " adddrop ")
        .replace("add-drop", " adddrop ")
        .replace("add - drop", " adddrop ")
        .replace("add drop", " adddrop ")
    )
    return re.findall(r"[a-z0-9çğıöşüâîû]+", text)


def documents_from_chunks(chunks: list[dict]) -> list[Document]:
    """Convert chunk dicts (from chunks.jsonl) into LangChain Documents."""
    documents = []
    for chunk in chunks:
        metadata = {
            "chunk_id": chunk.get("chunk_id") or "",
            "chunk_index": int(chunk.get("chunk_index", 0)),
            "url": chunk.get("url") or "",
            "title": chunk.get("title") or "",
            "section": chunk.get("section") or "",
            "language": chunk.get("language") or "",
            "source_id": chunk.get("source_id") or "",
        }
        documents.append(
            Document(page_content=chunk.get("text") or "", metadata=metadata)
        )
    return documents


def build_bm25_index(documents: list[Document]) -> dict:
    """
    Build a picklable BM25 index from documents.

    Stored payload:
      - bm25: BM25Okapi model
      - documents: list[Document] in the same order as the corpus
    """
    tokenized_documents = [tokenize(doc.page_content) for doc in documents]
    return {
        "bm25": BM25Okapi(tokenized_documents),
        "documents": documents,
    }


def save_bm25_index(index_payload: dict, chroma_folder: Path) -> Path:
    path = bm25_index_path(chroma_folder)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(index_payload, file)
    return path


def load_bm25_index(chroma_folder: Path) -> dict:
    path = bm25_index_path(chroma_folder)
    if not path.exists():
        raise FileNotFoundError(
            f"BM25 index not found: {path}\n"
            "Run embedding (rag/embed.py) to build Chroma + BM25."
        )
    with open(path, "rb") as file:
        return pickle.load(file)


def bm25_search(index_payload: dict, query: str, k: int = CANDIDATE_K) -> list[Document]:
    """Return top-k Documents by BM25 score."""
    bm25: BM25Okapi = index_payload["bm25"]
    documents: list[Document] = index_payload["documents"]

    if not documents:
        return []

    scores = bm25.get_scores(tokenize(query))
    # argsort descending without numpy dependency
    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for index in ranked_indices[:k]:
        if scores[index] <= 0:
            break
        results.append(documents[index])
    return results


def reciprocal_rank_fusion(
    result_lists: list[list[Document]],
    top_n: int,
    rrf_k: int = RRF_K,
    weights: list[float] | None = None,
) -> list[Document]:
    """
    Merge ranked lists with Reciprocal Rank Fusion.

    score(doc) = sum weight / (rrf_k + rank)
    """
    scores: dict[str, float] = {}
    docs_by_key: dict[str, Document] = {}

    if weights is None:
        weights = [1.0] * len(result_lists)
    if len(weights) != len(result_lists):
        raise ValueError("weights must match the number of result lists")

    for results, weight in zip(result_lists, weights):
        for rank, doc in enumerate(results, start=1):
            key = doc.metadata.get("chunk_id") or doc.page_content[:240]
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank)
            docs_by_key[key] = doc

    ranked_keys = sorted(scores, key=scores.get, reverse=True)
    return [docs_by_key[key] for key in ranked_keys[:top_n]]


def hybrid_search(
    vector_db,
    bm25_index: dict,
    query: str,
    top_k: int,
    candidate_k: int = CANDIDATE_K,
) -> list[Document]:
    """Dense (Chroma) + sparse (BM25) retrieval, fused with RRF."""
    dense_docs = vector_db.similarity_search(query, k=candidate_k)
    sparse_docs = bm25_search(bm25_index, query, k=candidate_k)
    # Slight BM25 bias helps exact phrases/dates beat near-topic FAQ pages but still working this out
    return reciprocal_rank_fusion(
        [dense_docs, sparse_docs],
        top_n=top_k,
        weights=[1.0, 1.0],
    )
