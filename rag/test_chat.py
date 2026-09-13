"""
Quick terminal test for the RAG + Groq pipeline (no Streamlit).

HOW TO RUN:
-----------
    .venv\\Scripts\\python rag\\test_chat.py
"""

import sys
from pathlib import Path

PROJECT_FOLDER = Path(__file__).resolve().parent.parent
if str(PROJECT_FOLDER) not in sys.path:
    sys.path.insert(0, str(PROJECT_FOLDER))

from rag.engine import MetuAssistant


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    assistant = MetuAssistant()

    test_questions = [
        "How many books can undergraduate students borrow from the library?",
        "Yurt basvurusu nasil yapilir?",
    ]

    for question in test_questions:
        print("\n" + "=" * 60)
        print(f"Q: {question}")
        print("-" * 60)
        result = assistant.ask(question)
        print(result["answer"])
        print("\nSources:")
        for source in result["sources"]:
            print(f"  - {source['title']}: {source['url']}")


if __name__ == "__main__":
    main()
