"""
=============================================================================
CLI Mode — The Original Terminal Interface
=============================================================================
This is the original command-line interface for Novel RAG, preserved from
before the web UI was added. It provides the same functionality through
an interactive terminal prompt.

USAGE:
    python -m src.cli

This module is separate from the web API (src/main.py) so you can choose
your preferred interface without them interfering with each other.
=============================================================================
"""

import os
import sys

# Add the root directory of the project to the Python path.
# This allows us to use 'import src.core...' regardless of where we run from.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import google.generativeai as genai
import src.core.config as config
from src.core.document_processor import DocumentProcessor
from src.core.vector_db import VectorDB
from src.core.utils import get_logger, time_it

logger = get_logger("CLI")


def setup_gemini():
    """Configures the Gemini API and returns a GenerativeModel instance."""
    api_key = config.settings.get("gemini_api_key", config.GEMINI_API_KEY)
    if not api_key:
        logger.error("Please add your GEMINI_API_KEY to the .env file")
        sys.exit(1)
    genai.configure(api_key=api_key)
    model_name = config.settings.get("generation_model", config.GENERATION_MODEL)
    return genai.GenerativeModel(model_name)


def select_novel() -> str:
    """Provides a terminal UI to let the user select which novel to query."""
    novels = config.get_available_novels()

    if not novels:
        logger.error(f"No novel directories found in {config.DATA_DIR}.")
        logger.error("Please create a folder for your novel containing markdown files.")
        sys.exit(1)

    print("\n📚 Available Novels:")
    for i, novel in enumerate(novels, 1):
        print(f"  {i}. {novel}")

    while True:
        try:
            choice = input(f"\nSelect a novel (1-{len(novels)}): ")
            index = int(choice) - 1
            if 0 <= index < len(novels):
                return novels[index]
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


@time_it
def ingest_data(novel_name: str, db: VectorDB):
    """Runs the ingestion pipeline for a specific novel."""
    logger.info(f"Initializing Data Ingestion Pipeline for '{novel_name}'...")
    novel_dir = os.path.join(config.DATA_DIR, novel_name)

    processor = DocumentProcessor(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    documents = processor.process_directory(novel_dir)

    if not documents:
        logger.error(f"No markdown documents found in {novel_dir}. Exiting.")
        sys.exit(1)

    db.store_documents(documents)
    logger.info("Data Ingestion Complete! You can now start asking questions.")


def format_context_for_prompt(retrieved_results: dict) -> str:
    """Formats ChromaDB results into a readable string for the LLM prompt."""
    docs = retrieved_results['documents']
    metas = retrieved_results['metadatas']

    formatted_context = ""
    for idx, (doc, meta) in enumerate(zip(docs, metas)):
        source = meta.get('source', 'Unknown File')
        formatted_context += f"--- Excerpt {idx + 1} (Source: {source}) ---\n"
        formatted_context += f"{doc}\n\n"

    return formatted_context


@time_it
def generate_answer(model, question: str, context: str) -> str:
    """Generates an answer using the RAG prompt template."""
    prompt = f"""You are an expert lore-keeper and character historian for a specific novel. 
The user is asking a question about the novel's characters, relationships, or plot.

I have performed a search of the novel chapters and retrieved the following highly specific excerpts that are mathematically relevant to the user's question.

YOUR INSTRUCTIONS:
1. Carefully read the excerpts provided below. 
2. Answer the user's question based ONLY on the facts present in these excerpts.
3. If the excerpts do not hold enough information to answer the question fully, explicitly state exactly what you do know, and then state what information is missing. DO NOT guess, DO NOT hallucinate, and DO NOT invent lore.
4. If asked about a character, focus heavily on describing who that character is, their relationships with other characters (if possible describe who those characters are as well), their actions as described in the text and when were they first introduced.
5. In your answer, you MUST cite the 'Source' (e.g., Chapter file) where you found the information.

====================
RETRIEVED NOVEL EXCERPTS:
{context}
====================

USER QUESTION: {question}

LORE-KEEPER ANSWER:
"""
    logger.info("Sending Prompt + Context to Gemini for generation...")
    response = model.generate_content(prompt)
    return response.text


def interactive_chat():
    """The main CLI chat loop."""
    selected_novel = select_novel()
    db = VectorDB(collection_name=selected_novel)

    if not db.has_documents():
        print(f"\n⚠️  No embeddings found for '{selected_novel}'.")
        print("⚙️  Automatically starting ingestion. This may take a few minutes...\n")
        ingest_data(selected_novel, db)

    model = setup_gemini()

    print("\n" + "=" * 50)
    print(f"📚 Novel RAG System Initialized for: {selected_novel}")
    print("Type your questions below. Type 'exit' or 'quit' to quit.")
    print("Type '!reingest' if you have added new chapters.")
    print("=" * 50 + "\n")

    while True:
        try:
            question = input("\n👤 Question: ")

            if question.lower().strip() in ['exit', 'quit']:
                print("Goodbye!")
                break

            if question.lower().strip() == '!reingest':
                ingest_data(selected_novel, db)
                continue

            if not question.strip():
                continue

            retrieval_results = db.query(question, n_results=7)
            context_string = format_context_for_prompt(retrieval_results)
            answer = generate_answer(model, question, context_string)

            print(f"\n🤖 Answer:\n{answer}\n")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logger.error(f"An error occurred during chat: {str(e)}")


if __name__ == "__main__":
    interactive_chat()
