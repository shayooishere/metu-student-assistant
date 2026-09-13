"""
METU Student Assistant — Web Scraper (Step 1 of the RAG pipeline)

Short Summary:
-----------------------------------------
1. Reads a list of METU web page URLs from urls.txt
2. Downloads each page 
3. Extracts the main text content (ignores menus, footers, ads)
4. Saves everything to a JSONL file in data/processed/pages.jsonl

"""

import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import trafilatura
from langdetect import LangDetectException, detect

# ---------------------------------------------------------------------------
# SETTINGS 
# ---------------------------------------------------------------------------

# How long to wait between requests (10 seconds crawl delay according to robot.txt on most metu webpages)
SECONDS_BETWEEN_REQUESTS = 10.0

# Ignore pages with less than this many characters (usually empty or broken)
MIN_CONTENT_LENGTH = 100

# Where to find urls.txt and where to save results
PROJECT_FOLDER = Path(__file__).resolve().parent.parent
URLS_FILE = Path(__file__).resolve().parent / "urls.txt"
OUTPUT_FILE = PROJECT_FOLDER / "data" / "processed" / "pages.jsonl"

# Browser-like header so METU servers accept our request
REQUEST_HEADERS = {
    "User-Agent": "METU-Student-Assistant (internship project)",

    #removable since our urls already directly refer the en version, does make our scraper look more like a regular browser request and less likely to be flagged by antibot checks
    "Accept-Language": "en-US,en;q=0.9,tr-TR;q=0.8,tr;q=0.7", 
}


# ---------------------------------------------------------------------------
# STEP 1 — Read urls.txt
# ---------------------------------------------------------------------------

def load_url_list(file_path):
    """
    Read urls.txt and return a list of dicts.

    """
    pages_to_scrape = []

    with open(file_path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            # Skip empty lines and comment lines 
            if not line or line.startswith("#"):
                continue

            #makes sure that only the first 2 | are split on so if the url contains a |, it is safe
            parts = line.split("|", maxsplit=2) 

            #further safeguard
            if len(parts) != 3:
                print(f"  [skip] Bad line format: {line}")
                continue

            language, section, url = parts
            pages_to_scrape.append({
                "language": language.strip(),
                "section": section.strip(),
                "url": url.strip(),
            })

    return pages_to_scrape


# ---------------------------------------------------------------------------
# STEP 2 — Download one web page
# ---------------------------------------------------------------------------

def download_page(url):
    """
    Send an HTTP GET request to the URL and return the HTML as text.

    Returns:
        (html_text, error_message)
        If successful: ("<html>...", None)
        If failed:     (None, "some error")
    """
    try:
        response = requests.get(
            url,
            headers=REQUEST_HEADERS,

            #if the METU server hangs, our script will throw an error after 30 secs instead of just freezing infinitely
            timeout=30,
        )
        response.raise_for_status()  # Raise an error for 404, 500, etc.

        # Fix Turkish character encoding if the server guessed wrong
        response.encoding = response.apparent_encoding or "utf-8"
        return response.text, None

    except requests.RequestException as error:
        return None, str(error)


# ---------------------------------------------------------------------------
# STEP 3 — Extract clean text from HTML
# ---------------------------------------------------------------------------

def extract_main_content(html, url):
    """
    Use trafilatura to pull out the main article/content from raw HTML.

    We try markdown first (keeps headings like # Title).
    If that fails, we fall back to plain text.

    favor_recall=True means "get more text" — better for METU pages.
    """
    settings = {
        "url": url,
        "include_links": True,
        "include_tables": True,
        "favor_recall": True,
    }

    # Try markdown format
    text = trafilatura.extract(html, output_format="markdown", **settings)
    if text and len(text.strip()) >= MIN_CONTENT_LENGTH:
        return text.strip(), "markdown"

    # Fallback: plain text
    text = trafilatura.extract(html, output_format="txt", **settings)
    if text and len(text.strip()) >= MIN_CONTENT_LENGTH:
        return text.strip(), "txt"

    return None, None


def get_page_title(html, url):
    """Get the page title from HTML metadata (or use the URL as fallback)."""
    metadata = trafilatura.extract_metadata(html, default_url=url)
    if metadata and metadata.title:
        return metadata.title.strip()
    return url


def guess_content_language(text):
    """
    Auto-detect Turkish vs English from the text itself.
    This is a backup check — we already know the expected language from urls.txt.
    """
    sample = text[:2000].strip()
    if len(sample) < 50:
        return None
    try:
        return detect(sample)
    except LangDetectException:
        return None


# ---------------------------------------------------------------------------
# STEP 4 — Scrape one URL and build a record
# ---------------------------------------------------------------------------

def make_page_id(url):
    """Create a short unique ID from the URL (used as a key in the database later)."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]


def scrape_one_page(language, section, url):
    """
    Full pipeline for a single URL:
    download -> extract text -> build a record dict
    """
    record = {
        "id": make_page_id(url),
        "url": url,
        "section": section,
        "expected_language": language,
        "title": "",
        "content": "",
        "content_format": "markdown",
        "detected_language": None,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
        "status": "error",
        "error": None,
        "char_count": 0,
    }

    # 1. Download
    html, error = download_page(url)
    if error:
        record["error"] = error
        return record

    # 2. Get title
    record["title"] = get_page_title(html, url)

    # 3. Extract main content
    content, content_format = extract_main_content(html, url)
    if not content:
        record["error"] = "No extractable content (page may be empty or JavaScript-only)"
        return record

    # 4. Success — fill in the rest
    record["content"] = content
    record["content_format"] = content_format or "txt"
    record["detected_language"] = guess_content_language(content)
    record["char_count"] = len(content)
    record["status"] = "success"
    record["error"] = None

    return record


# ---------------------------------------------------------------------------
# STEP 5 — Save all records to JSONL file
# ---------------------------------------------------------------------------

def save_to_jsonl(records, output_path):
    """
    Write each record as one line of JSON.

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        for record in records:
            line = json.dumps(record, ensure_ascii=False)
            file.write(line + "\n")


# ---------------------------------------------------------------------------
# MAIN — ties everything together
# ---------------------------------------------------------------------------

def main():
    # Windows terminals sometimes fail on Turkish characters, force UTF-8 output
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 60)
    print("METU Student Assistant - Web Scraper")
    print("=" * 60)

    # Load the URL list
    pages = load_url_list(URLS_FILE)
    print(f"\nLoaded {len(pages)} URLs from urls.txt\n")

    records = []
    success_count = 0
    error_count = 0

    # Scrape each URL one by one
    for index, page in enumerate(pages):
        url = page["url"]
        print(f"[{index + 1}/{len(pages)}] {page['language'].upper()} | {page['section']}")
        print(f"         {url}")

        record = scrape_one_page(page["language"], page["section"], url)
        records.append(record)

        if record["status"] == "success":
            success_count += 1
            print(f"         OK - {record['char_count']} characters")
        else:
            error_count += 1
            print(f"         FAILED - {record['error']}")

        # Wait between requests (skip wait after the last URL)
        if index < len(pages) - 1:
            time.sleep(SECONDS_BETWEEN_REQUESTS)

    # Save results
    save_to_jsonl(records, OUTPUT_FILE)

    print("\n" + "=" * 60)
    print(f"Done!  Success: {success_count}  |  Errors: {error_count}")
    print(f"Saved to: {OUTPUT_FILE}")
    print("=" * 60)


#running python scrape.py vs importing a certain function from scrape.py later on
if __name__ == "__main__":
    main()
