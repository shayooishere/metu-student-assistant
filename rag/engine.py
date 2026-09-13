"""
METU Student Assistant — RAG Engine (Step 4 of the RAG pipeline)

Short Summary:
----------------------
1. Loads the Chroma vector database + BM25 keyword index
2. Optionally rewrites follow-up questions using short chat history
3. Hybrid-retrieves top-K chunks (dense + BM25, RRF fusion)
4. Sends those chunks + the question (+ short history) to Groq (LLM)
5. Returns an answer + source URLs

This file has no user interface — app.py (Streamlit) calls it.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langdetect import DetectorFactory, LangDetectException, detect
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from rag.hybrid import hybrid_search, load_bm25_index

# Make langdetect deterministic across runs
DetectorFactory.seed = 0

# ---------------------------------------------------------------------------
# SETTINGS 
# ---------------------------------------------------------------------------

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
CHROMA_FOLDER = PROJECT_FOLDER / "chroma_db"
COLLECTION_NAME = "metu_student_assistant"
EMBEDDING_MODEL = "BAAI/bge-m3"

# Final chunks sent to the LLM (hybrid dense + BM25, RRF-ranked)
TOP_K = 5

# How many unique source URLs to show in the UI 
MAX_SOURCES = 5

# Short-term chat memory: how much prior conversation to use
HISTORY_MAX_MESSAGES = 6  # last 3 user+assistant exchanges
HISTORY_MAX_CHARS = 2000
REWRITE_MAX_TOKENS = 256

# The groq free tier model we will be using
GROQ_MODEL = "openai/gpt-oss-20b"

# Keep room for long procedure answers (residence permit, etc.)
# Note: gpt-oss-20b also spends some of this budget on hidden reasoning tokens.
MAX_OUTPUT_TOKENS = 4096

# Load GROQ_API_KEY from .env file
load_dotenv(PROJECT_FOLDER / ".env")


# ---------------------------------------------------------------------------
# Our LLM Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the METU Student Assistant for Middle East Technical University (Ankara).

Rules:
1. Answer ONLY using the context below. Do not invent university rules, dates, or external websites.
2. LANGUAGE (critical): Answer entirely in {language}.
3. Recent conversation is only for continuity (pronouns, follow-ups like "what about spring?").
   Do NOT take university facts from the conversation — only from the context.
4. If the context is not enough, OR the question is outside METU student-support scope:
   - Acknowledge the question briefly and warmly.
   - Explain politely that it is outside what you are designed to help with, or that the METU documents you have do not contain enough information.
   - Invite a follow-up about METU topics (dorms, library, registration, student affairs, etc.).
   - Do NOT invent links or facts that are not in the context.
5. Answer length should match the question:
   - Simple fact questions (e.g. borrowing limits, a single date): keep it brief.
   - Procedures and requirements (residence permit, registration, documents, applications):
     be complete and precise. Include every required step and document from the context.
     Do not omit details that could cause a student to miss a requirement.
   - Prefer clear structure (numbered steps / bullet lists) over long prose.
   - Do not invent extra steps; only include what the context supports.
6. Do NOT list, mention, or append source URLs, titles, or a "Sources" section in your answer.
   Sources are shown separately in the app UI.

Context:
{context}
"""

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            "Recent conversation (may be empty):\n{history}\n\n"
            "User question:\n{question}\n\n"
            "Reply entirely in {language}.",
        ),
    ]
)

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite follow-up questions into standalone search queries for a METU "
            "student-info assistant.\n"
            "Rules:\n"
            "- Use the recent conversation only to resolve references "
            "(e.g. \"what about spring?\", \"ya yaz okulu?\", \"and for graduates?\").\n"
            "- Output ONE standalone question in the SAME language as the latest user message.\n"
            "- If the latest message is already standalone, return it unchanged.\n"
            "- Do not answer the question. Output only the rewritten question, no quotes or preamble.",
        ),
        (
            "human",
            "Recent conversation:\n{history}\n\n"
            "Latest user message:\n{question}\n\n"
            "Standalone question:",
        ),
    ]
)


# ---------------------------------------------------------------------------
# Load embeddings + Chroma
# ---------------------------------------------------------------------------

def create_embedding_model():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vector_database():
    """
    Open the existing Chroma database from disk.

    """
    if not CHROMA_FOLDER.exists():
        raise FileNotFoundError(
            f"Vector database not found: {CHROMA_FOLDER}\n"
            "Run embedding first"
        )

    embeddings = create_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_FOLDER),
    )


def create_llm():
    """
    Connect to Groq using the API key.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing."
        )

    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0.2,  # low = more factual, less creative
        max_tokens=MAX_OUTPUT_TOKENS,
        # gpt-oss uses reasoning tokens from the total max_tokens allowed.
        # "low" leaves more room for the visible answer in long procedures.
        reasoning_effort="low",
        api_key=api_key,
    )


# ---------------------------------------------------------------------------
# Format retrieved chunks / chat history
# ---------------------------------------------------------------------------

def format_context(docs):
    """
    Turn retrieved Document objects into readable text for the LLM.

    URLs are kept out of the prompt so the model does not
    copy them into the answer. The UI shows sources from metadata of retrieved chunks instead.
    """
    parts = []
    for i, doc in enumerate(docs, start=1):
        title = doc.metadata.get("title", "Untitled")
        parts.append(f"[Document {i}] {title}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def format_history(history: list[dict] | None) -> str:
    """
    Format prior chat turns for prompts. Caps message count and total length.
    Expects items like {"role": "user"|"assistant", "content": "..."}.
    """
    if not history:
        return "(none)"

    lines = []
    for message in history[-HISTORY_MAX_MESSAGES:]:
        role = (message.get("role") or "").strip().lower()
        content = (message.get("content") or "").strip()
        if not content or role not in {"user", "assistant"}:
            continue
        label = "User" if role == "user" else "Assistant"
        lines.append(f"{label}: {content}")

    if not lines:
        return "(none)"

    text = "\n".join(lines)
    if len(text) > HISTORY_MAX_CHARS:
        text = text[-HISTORY_MAX_CHARS:]
    return text


def unique_sources(docs, max_sources: int = MAX_SOURCES):
    """
    Return unique (title, url) pairs for display under the answer.

    Docs are expected in hybrid/RRF order (best first). Only the first
    'max_sources' distinct URLs are kept so the UI stays focused.
    """
    seen = set() #faster lookups than list
    sources = []
    for doc in docs:
        if len(sources) >= max_sources:
            break
        url = doc.metadata.get("url", "")
        title = doc.metadata.get("title", "Source")
        if url and url not in seen:
            seen.add(url)
            sources.append({"title": title, "url": url})
    return sources


def detect_query_language(question: str) -> str:
    """
    Detect whether the user question is Turkish or English with langdetect.

    Returns "Turkish" or "English". Defaults to English if detection fails
    or the result is neither tr nor en.
    """
    text = (question or "").strip()
    if not text:
        return "English"

    try:
        code = detect(text)
    except LangDetectException:
        return "English"

    if code == "tr":
        return "Turkish"
    return "English"


# ---------------------------------------------------------------------------
# Main RAG function: question -> answer + sources
# ---------------------------------------------------------------------------

class MetuAssistant:
    """
    Usage:
        assistant = MetuAssistant()
        result = assistant.ask("How many books can I borrow?")
        print(result["answer"])
        print(result["sources"])

    """

    def __init__(self):
        print("Loading vector database, BM25 index, and Groq model...")
        self.vector_db = load_vector_database()
        self.bm25_index = load_bm25_index(CHROMA_FOLDER)
        self.llm = create_llm()
        print("Ready.")

    def rewrite_query(self, question: str, history: list[dict] | None) -> str:
        """
        Turn a follow-up into a standalone search query using recent chat history.
        If there is no history, returns the original question.
        """
        history_text = format_history(history)
        if history_text == "(none)":
            return question

        messages = REWRITE_PROMPT.format_messages(
            history=history_text,
            question=question,
        )
        try:
            response = self.llm.invoke(
                messages,
                max_tokens=REWRITE_MAX_TOKENS,
                reasoning_effort="low",
            )
            rewritten = (response.content or "").strip()
            # Drop wrapping quotes / accidental labels
            if rewritten.startswith(("Standalone question:", "Question:")):
                rewritten = rewritten.split(":", 1)[1].strip()
            rewritten = rewritten.strip().strip('"').strip("'")
            if rewritten:
                return rewritten
        except Exception as error:
            print(f"Query rewrite failed, using original question: {error}")

        return question

    def ask(self, question: str, history: list[dict] | None = None) -> dict:
        """
        Full RAG pipeline for one question:

        1. Optional: rewrite follow-ups using short chat history
        2. Hybrid retrieve top-K chunks (Chroma dense + BM25, fused with RRF)
        3. Ask Groq to answer using context (+ short history for continuity)
        """
        question = (question or "").strip() #if question is None, it defaults to an empty string
        if not question:
            return {
                "answer": "Please enter a question.",
                "sources": [],
            }

        history_text = format_history(history)
        search_query = self.rewrite_query(question, history)

        docs = hybrid_search(
            self.vector_db,
            self.bm25_index,
            search_query,
            top_k=TOP_K,
        )

        if not docs:
            return {
                "answer": "I could not find any related information in the METU documents.",
                "sources": [],
            }

        language = detect_query_language(question)
        context = format_context(docs)
        messages = PROMPT.format_messages(
            context=context,
            history=history_text,
            question=question,
            language=language,
        )
        response = self.llm.invoke(
            messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            reasoning_effort="low",
        )

        return {
            "answer": response.content,
            "sources": unique_sources(docs),
        }
