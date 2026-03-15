"""
=============================================================================
Vector Database Module — ChromaDB Storage & Similarity Search
=============================================================================
This is the heart of the retrieval system.

HOW IT WORKS:
When we chop a novel into small overlapping chunks, we can't just save them
in a regular SQL database — SQL only searches for exact word matches.
A vector database stores the text ALONGSIDE its mathematical representation
(embedding). When the user asks a question, we:
  1. Convert the question into a vector (using the SAME model as stored docs)
  2. Ask ChromaDB to find the N nearest neighbors in vector space
  3. Return those text chunks as "context" for the LLM

COLLECTION-PER-NOVEL DESIGN:
Each novel gets its own ChromaDB "Collection" (like a database table).
This means characters from "Harry Potter" never mix with "Lord of the Rings".
The collection name is derived from the novel's folder name.

PERSISTENCE:
ChromaDB is configured as a PersistentClient, meaning it saves all data to
disk in the db/ directory. You don't have to re-process 300 chapters every
time you restart the app!
=============================================================================
"""

import chromadb
from typing import List, Dict, Any
import src.core.config as config
from src.core.embeddings import Embedder
from src.core.utils import get_logger, time_it

logger = get_logger("VectorDB")


class VectorDB:
    """
    Wraps ChromaDB to provide simple store/query operations for novel chunks.

    Each instance connects to a specific novel's collection, ensuring
    complete isolation between different novels' data.
    """

    def __init__(self, collection_name: str):
        """
        Initialize a connection to a specific novel's vector collection.

        Args:
            collection_name: The novel's identifier (usually the folder name).
                            Sanitized to be alphanumeric + underscores.
        """
        # PersistentClient saves to disk so data survives restarts
        self.client = chromadb.PersistentClient(path=config.DB_DIR)

        # Initialize the embedder for converting queries to vectors
        self.embedder = Embedder()

        # ChromaDB requires collection names to be alphanumeric with
        # underscores/hyphens. We sanitize the novel name just in case.
        safe_name = "".join(
            [c if c.isalnum() else "_" for c in collection_name]
        ).strip("_")
        self.collection_name = safe_name

        # get_or_create_collection either loads an existing collection
        # or creates a new empty one — perfect for first-time setup.
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name
        )

    def has_documents(self) -> bool:
        """
        Check if this novel's collection has any chunks stored.

        Used by the app to decide whether to auto-trigger ingestion
        for a novel that hasn't been processed yet.

        Returns:
            True if the collection contains at least one chunk.
        """
        return self.collection.count() > 0

    @time_it
    def store_documents(
        self,
        documents: List[Dict[str, Any]],
        progress_callback=None
    ) -> None:
        """
        Insert document chunks into ChromaDB with their embeddings.

        We process in batches of 100 to avoid overwhelming memory. For each
        batch, we:
          1. Extract the text, metadata, and IDs
          2. Generate embeddings via the Embedder
          3. Upsert into ChromaDB (insert-or-update)

        The progress_callback is used by the WebSocket endpoint to send
        real-time progress updates to the front-end.

        Args:
            documents: List of dicts with 'id', 'text', 'metadata' keys.
            progress_callback: Optional callable(current, total, message)
                             for progress reporting.
        """
        if not documents:
            logger.warning("No documents provided to store in DB.")
            return

        batch_size = 100
        total_docs = len(documents)
        total_batches = (total_docs + batch_size - 1) // batch_size
        logger.info(
            f"Storing {total_docs} chunks into the database "
            f"in batches of {batch_size}..."
        )

        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            current_batch = i // batch_size + 1

            # Split the dicts into the separate lists ChromaDB expects
            ids = [doc['id'] for doc in batch]
            texts = [doc['text'] for doc in batch]
            metadatas = [doc['metadata'] for doc in batch]

            # Generate embeddings for this batch
            embeddings = self.embedder.embed_texts(texts)

            # upsert = insert + update: if a chunk ID already exists,
            # it gets updated rather than duplicated
            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )

            logger.info(f"Inserted batch {current_batch}/{total_batches}")

            # Report progress to the WebSocket (if connected)
            if progress_callback:
                progress_callback(
                    current=i + len(batch),
                    total=total_docs,
                    message=f"Embedded batch {current_batch}/{total_batches}"
                )

        logger.info(
            f"Finished storing. Database now contains "
            f"{self.collection.count()} chunks."
        )

    @time_it
    def query(self, question: str, n_results: int = 5) -> Dict:
        """
        Find the most relevant text chunks for a user's question.

        THE RETRIEVAL PROCESS:
          1. Convert the question to a vector (same model used for storage!)
          2. ChromaDB calculates the mathematical "distance" between the
             question vector and every stored chunk vector
          3. The N closest chunks are returned, sorted by relevance

        The "distance" score tells you how close the match is:
          - Lower distance = more relevant (0 would be identical)
          - Higher distance = less relevant

        Args:
            question: The user's natural language question.
            n_results: Number of relevant chunks to retrieve (default: 5).

        Returns:
            Dict with 'documents' (text), 'metadatas' (sources), and
            'distances' (relevance scores).
        """
        logger.info(f"Searching database for query: '{question}'")

        # Step 1: Convert the question to the same vector space
        query_embedding = self.embedder.embed_query(question)

        # Step 2: Nearest-neighbor search in ChromaDB
        # query_embeddings expects a list-of-lists (for batch queries)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # ChromaDB wraps results in an extra list layer (for batch support).
        # Since we only asked one question, we unwrap with [0].
        return {
            "documents": results['documents'][0],
            "metadatas": results['metadatas'][0],
            "distances": results['distances'][0]
        }
