"""
=============================================================================
Tests for Document Processor — Chunking & File Processing
=============================================================================
These tests verify that the sliding window chunking algorithm works correctly
and that the directory processor handles various edge cases.

TEST DESIGN PATTERN:
Each test follows the Arrange-Act-Assert (AAA) pattern:
  - Arrange: Set up the test data and objects
  - Act:     Call the function being tested
  - Assert:  Verify the results are correct
=============================================================================
"""

import os
import pytest


class TestSplitText:
    """Tests for the _split_text() sliding window algorithm."""

    def _create_processor(self, chunk_size=600, chunk_overlap=150):
        """Helper to create a DocumentProcessor with custom settings."""
        from src.core.document_processor import DocumentProcessor
        return DocumentProcessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def test_text_shorter_than_chunk(self):
        """
        Text shorter than chunk_size should return as a single chunk
        without any splitting.
        """
        processor = self._create_processor(chunk_size=100)
        text = "Short text"
        chunks = processor._split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_text_equal_to_chunk_size(self):
        """
        Text exactly equal to chunk_size should also return as one chunk.
        The boundary condition: len(text) == chunk_size.
        """
        processor = self._create_processor(chunk_size=10)
        text = "0123456789"  # Exactly 10 chars
        chunks = processor._split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_overlap_creates_shared_content(self):
        """
        Verify that consecutive chunks share overlapping content.
        This is the core feature — overlapping ensures pronoun references
        aren't lost between chunks.
        """
        processor = self._create_processor(chunk_size=10, chunk_overlap=3)
        text = "ABCDEFGHIJKLMNOP"  # 16 chars
        chunks = processor._split_text(text)

        # With chunk_size=10, overlap=3, step=7:
        # Chunk 0: chars 0-9   = "ABCDEFGHIJ"
        # Chunk 1: chars 7-15  = "HIJKLMNOP"
        assert len(chunks) == 2
        assert chunks[0] == "ABCDEFGHIJ"
        assert chunks[1] == "HIJKLMNOP"

        # Verify overlap: last 3 chars of chunk 0 == first 3 chars of chunk 1
        assert chunks[0][-3:] == "HIJ"
        assert chunks[1][:3] == "HIJ"

    def test_multiple_chunks(self):
        """
        Longer text should produce multiple overlapping chunks.
        """
        processor = self._create_processor(chunk_size=10, chunk_overlap=2)
        text = "A" * 30  # 30 chars
        chunks = processor._split_text(text)

        # step = 10 - 2 = 8
        # Chunks: 0-9, 8-17, 16-25, 24-29
        assert len(chunks) == 4

    def test_empty_text(self):
        """
        Empty text should return a single empty string chunk.
        """
        processor = self._create_processor()
        chunks = processor._split_text("")
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_each_chunk_respects_max_size(self):
        """
        No chunk should ever exceed chunk_size in length.
        """
        processor = self._create_processor(chunk_size=50, chunk_overlap=10)
        text = "X" * 500
        chunks = processor._split_text(text)

        for chunk in chunks:
            assert len(chunk) <= 50


class TestProcessDirectory:
    """Tests for the process_directory() file processing pipeline."""

    def test_process_finds_markdown_files(self, sample_markdown_dir):
        """
        Verify that process_directory discovers all .md files and
        produces chunks with proper metadata.
        """
        from src.core.document_processor import DocumentProcessor
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        docs = processor.process_directory(sample_markdown_dir)

        # Should find chunks from both chapter files
        assert len(docs) > 0

        # Each document should have the required keys
        for doc in docs:
            assert "id" in doc
            assert "text" in doc
            assert "metadata" in doc
            assert "source" in doc["metadata"]
            assert "chunk_index" in doc["metadata"]

    def test_metadata_contains_source_filename(self, sample_markdown_dir):
        """
        Each chunk's metadata should reference its source file.
        This is how the LLM knows which chapter a fact came from.
        """
        from src.core.document_processor import DocumentProcessor
        processor = DocumentProcessor(chunk_size=200, chunk_overlap=50)
        docs = processor.process_directory(sample_markdown_dir)

        sources = set(doc["metadata"]["source"] for doc in docs)
        assert "chapter-1.md" in sources
        assert "chapter-2.md" in sources

    def test_empty_directory_returns_empty(self, temp_dir):
        """
        A directory with no .md files should return an empty list.
        """
        from src.core.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        docs = processor.process_directory(temp_dir)
        assert docs == []

    def test_ids_are_unique(self, sample_markdown_dir):
        """
        Every chunk must have a unique ID — ChromaDB would overwrite
        duplicates otherwise.
        """
        from src.core.document_processor import DocumentProcessor
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=25)
        docs = processor.process_directory(sample_markdown_dir)

        ids = [doc["id"] for doc in docs]
        assert len(ids) == len(set(ids)), "Duplicate IDs found!"
