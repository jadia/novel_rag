"""
=============================================================================
Document Processor — Chapter Chunking with Sliding-Window Overlap
=============================================================================
This module is responsible for reading raw markdown novel chapters and
preparing them for the Vector Database.

WHY DO WE CHUNK?
An average chapter in a web novel can be ~1,888 words (~10,000 characters).
Feeding an entire chapter into the database as a single item creates a
problem called "Relevance Dilution" — if a chapter covers a sword fight,
a political debate, AND a quiet character moment, a single embedding vector
for all three topics will be muddled and won't strongly match any specific
user question.

THE SLIDING WINDOW STRATEGY:
We use a "sliding window" approach:
  - Window size: 600 characters (~100-150 words, about 1-2 paragraphs)
  - Step size:   600 - 150 = 450 characters
  - Overlap:     150 characters

The overlap is the key insight. Novels use pronouns across paragraph
boundaries: if Chunk A says "Yutia grabbed her sword" and Chunk B says
"She swung it at the demon", the 150-char overlap ensures Chunk B still
contains "Yutia" so the embedding captures the relationship.

  [=== Chunk 1 (600 chars) ===]
              [=== Chunk 2 (600 chars) ===]
                          [=== Chunk 3 (600 chars) ===]
              |  overlap  |
=============================================================================
"""

import os
import glob
from typing import List, Dict
import src.core.config as config
from src.core.utils import get_logger, time_it

logger = get_logger("DocumentProcessor")


class DocumentProcessor:
    """
    Reads markdown files, splits them into overlapping chunks, and attaches
    metadata (source filename, chunk index) to each chunk.

    Metadata is how the LLM knows WHERE a particular fact came from, enabling
    it to cite sources like "According to Chapter 12..."
    """

    def __init__(
        self,
        chunk_size: int = config.CHUNK_SIZE,
        chunk_overlap: int = config.CHUNK_OVERLAP
    ):
        """
        Args:
            chunk_size:    Number of characters per chunk (default: 600).
            chunk_overlap: Number of overlapping characters between chunks (default: 150).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _split_text(self, text: str) -> List[str]:
        """
        Implements the sliding window algorithm.

        Visual example with chunk_size=10, overlap=3:
          Text: "ABCDEFGHIJKLMNOP" (16 chars)
          Step: 10 - 3 = 7

          i=0:  "ABCDEFGHIJ"     (chars 0-9)
          i=7:  "HIJKLMNOP"      (chars 7-15, "HIJ" overlaps with previous)

        Args:
            text: The full chapter text.

        Returns:
            A list of text chunks, each up to chunk_size characters.
        """
        # If the text is shorter than a single chunk, just return it as-is.
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        # The step size determines how far we move the window each iteration.
        # By subtracting the overlap, we ensure each new chunk starts
        # `overlap` characters before the end of the previous chunk.
        step_size = self.chunk_size - self.chunk_overlap

        for i in range(0, len(text), step_size):
            chunk = text[i:i + self.chunk_size]
            chunks.append(chunk)

            # If our current chunk reached the end of the text, stop looping.
            if i + self.chunk_size >= len(text):
                break

        return chunks

    @time_it
    def process_directory(self, data_dir: str) -> List[Dict]:
        """
        Reads all markdown files in the given directory, splits them into
        chunks, and attaches metadata (source filename + chunk index).

        Args:
            data_dir: Path to the directory containing .md files.

        Returns:
            A list of dicts, each with keys: 'id', 'text', 'metadata'.
        """
        logger.info(f"Starting to process markdown files in: {data_dir}")

        # Find all .md files in the directory and subdirectories
        search_pattern = os.path.join(data_dir, "**", "*.md")
        filepaths = glob.glob(search_pattern, recursive=True)

        if not filepaths:
            logger.warning(
                f"No markdown files found in {data_dir}. "
                "Did you forget to add your chapters?"
            )
            return []

        processed_documents = []
        total_files = len(filepaths)
        total_chunks = 0

        for filepath in filepaths:
            filename = os.path.basename(filepath)

            try:
                # Open the file and read the raw text
                with open(filepath, 'r', encoding='utf-8') as file:
                    content = file.read()

                # Apply the sliding window chunking algorithm
                chunks = self._split_text(content)
                total_chunks += len(chunks)

                # For every chunk, we store the text AND where it came from.
                # This metadata is the foundation of RAG citation.
                for i, chunk in enumerate(chunks):
                    doc = {
                        "id": f"{filename}-chunk-{i}",
                        "text": chunk,
                        "metadata": {
                            "source": filename,
                            "chunk_index": i
                        }
                    }
                    processed_documents.append(doc)

            except Exception as e:
                logger.error(f"Failed to process file {filename}: {str(e)}")

        logger.info(
            f"Successfully processed {total_files} files "
            f"into {total_chunks} total chunks."
        )

        return processed_documents
