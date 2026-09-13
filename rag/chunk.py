"""
METU Student Assistant — Chunking (Step 2 of the RAG pipeline)

Short Summary:
-----------------------------------------
1. Reads scraped pages from data/processed/pages.jsonl
2. Splits each page into smaller overlapping chunks
3. Saves chunks to data/processed/chunks.jsonl
"""

import hashlib
import json
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
INPUT_FILE = PROJECT_FOLDER / "data" / "processed" / "pages.jsonl"
OUTPUT_FILE = PROJECT_FOLDER / "data" / "processed" / "chunks.jsonl"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_CHUNK_LENGTH = 100

HEADERS_TO_SPLIT_ON = [
    ("#", "header_1"),
    ("##", "header_2"),
    ("###", "header_3"),
    ("####", "header_4"),
    ("#####", "header_5"),
]


# ---------------------------------------------------------------------------
# STEP 1 — Load pages.jsonl
# ---------------------------------------------------------------------------

def load_pages(file_path):
    """
    Read pages.jsonl and return successfully scraped pages that have content.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {file_path}\n"
            "Run the scraper first"
        )

    pages = []
    with open(file_path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            if record.get("status") != "success":
                print(f"  [skip] Line {line_number}: scrape failed — {record.get('url')}")
                continue

            if not record.get("content", "").strip():
                print(f"  [skip] Line {line_number}: empty content — {record.get('url')}")
                continue

            pages.append(record)

    return pages


# ---------------------------------------------------------------------------
# STEP 2 — Metadata helpers
# ---------------------------------------------------------------------------

def build_base_metadata(page_record):
    return {
        "source_id": page_record["id"],
        "url": page_record["url"],
        "title": page_record["title"],
        "section": page_record["section"],
        "language": page_record["expected_language"],
    }


def make_chunk_id(source_id, chunk_index):
    raw = f"{source_id}:{chunk_index}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# STEP 3 — Splitters + chunk one page
# ---------------------------------------------------------------------------

def create_splitters():
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADERS_TO_SPLIT_ON,
        strip_headers=False, #include the headers in the chunk themselves
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""], #seperate on paragraphs first, then line breaks, then sentences, then words, then forcibly through middle of the string
    )
    return markdown_splitter, text_splitter


def is_catalog_course_page(page_record) -> bool:
    """METU academic catalog single-course pages are kept whole 
    and chunked only using the recursive character text splitter if necessary"""
    
    url = page_record.get("url") or ""
    return "catalog.metu.edu.tr/course.php" in url


def chunk_catalog_course(page_record, text_splitter):
    """
    Keep catalog course pages intact so metadata and Course Content stay together.

    Skips MarkdownHeaderTextSplitter (which would split ## title from ### Course Content
    with no overlap).
    """
    base_metadata = build_base_metadata(page_record)
    content = (page_record.get("content") or "").strip()
    if len(content) < MIN_CHUNK_LENGTH:
        return []

    doc = Document(page_content=content, metadata=base_metadata)

    return [doc]


def chunk_one_page(page_record, markdown_splitter, text_splitter):
    """
    Split one page into Documents.

    Catalog course pages: whole page (no header split).
    Everything else: markdown headers + recursive splitter.
    """
    if is_catalog_course_page(page_record):
        return chunk_catalog_course(page_record, text_splitter)

    base_metadata = build_base_metadata(page_record)
    content = page_record["content"]

    # Split by markdown headings
    # Returns a list of Document objects (one per section under a heading)
    header_docs = markdown_splitter.split_text(content)

    # Attach page metadata to every section document
    for doc in header_docs:
        doc.metadata = {**base_metadata, **doc.metadata}

    # Further split long sections into overlapping chunks
    final_docs = text_splitter.split_documents(header_docs)

    # Drop tiny leftovers
    return [
        doc for doc in final_docs
        if len(doc.page_content.strip()) >= MIN_CHUNK_LENGTH
    ]


# ---------------------------------------------------------------------------
# STEP 4 — Save chunks to JSONL
# ---------------------------------------------------------------------------

def document_to_record(doc, chunk_index):
    source_id = doc.metadata.get("source_id", "unknown")
    text = doc.page_content.strip()

    return {
        "chunk_id": make_chunk_id(source_id, chunk_index),
        "chunk_index": chunk_index,
        "text": text,
        "char_count": len(text),
        "url": doc.metadata.get("url"),
        "title": doc.metadata.get("title"),
        "section": doc.metadata.get("section"),
        "language": doc.metadata.get("language"),
        "source_id": source_id,
        "header_1": doc.metadata.get("header_1"),
        "header_2": doc.metadata.get("header_2"),
        "header_3": doc.metadata.get("header_3"),
        "header_4": doc.metadata.get("header_4"),
        "header_5": doc.metadata.get("header_5"),
    }


def save_chunks(documents, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        for index, doc in enumerate(documents):
            record = document_to_record(doc, index)
            file.write(json.dumps(record, ensure_ascii=False))
            file.write("\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("METU Student Assistant - Chunking (Step 2)")
    print("=" * 60)
    print(f"\nInput:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Settings: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}\n")

    pages = load_pages(INPUT_FILE)
    print(f"Loaded {len(pages)} pages from pages.jsonl\n")

    markdown_splitter, text_splitter = create_splitters()
    all_chunks = []

    for page_index, page in enumerate(pages):
        title = (page.get("title") or "")[:50]
        print(f"[{page_index + 1}/{len(pages)}] {page['section']} | {title}")

        page_chunks = chunk_one_page(page, markdown_splitter, text_splitter)
        print(f"         -> {len(page_chunks)} chunks")

        all_chunks.extend(page_chunks)

    save_chunks(all_chunks, OUTPUT_FILE)

    if all_chunks:
        char_counts = [len(doc.page_content) for doc in all_chunks]
        avg_size = sum(char_counts) // len(char_counts)
        print("\n" + "=" * 60)
        print(f"Done!  {len(all_chunks)} total chunks from {len(pages)} pages")
        print(f"Average chunk size: {avg_size} characters")
        print(f"Saved to: {OUTPUT_FILE}")
        print("=" * 60)
    else:
        print("\nNo chunks created. Check pages.jsonl content.")


if __name__ == "__main__":
    main()
