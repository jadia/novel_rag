"""
=============================================================================
Embeddings Module — Text-to-Vector Conversion
=============================================================================
This module translates plain English text into arrays of numbers (vectors).

WHY DO WE NEED EMBEDDINGS?
A vector database like ChromaDB doesn't understand words — it only
understands numbers. By converting text into a list of floats (a "vector"),
we can use mathematical distance calculations to find which stored text
chunks are most "similar" to a user's question.

THE SWITCH PATTERN:
This module supports two embedding backends:
  1. LOCAL (free):  HuggingFace's all-MiniLM-L6-v2, runs on your CPU
  2. API (paid):    Google's Gemini Embedding API, higher accuracy

The choice is controlled by the SettingsManager. The critical rule is:
You MUST use the SAME model for embedding documents AND querying.
If you store chapters with MiniLM but query with Gemini, the math won't
match and you'll get garbage results.
=============================================================================
"""

from typing import List
import google.generativeai as genai
import src.core.config as config
from src.core.utils import get_logger, time_it

logger = get_logger("Embeddings")


class Embedder:
    """
    Converts text into numerical vectors using either a local model or
    the Gemini API, controlled by the settings configuration.

    DESIGN DECISION: We initialize the heavy model (SentenceTransformer)
    only if the user explicitly chooses local mode. This saves ~200MB of
    RAM when running in API mode.
    """

    def __init__(self):
        """
        Initialize the appropriate embedding backend based on settings.

        Reads from the SettingsManager singleton to determine which
        backend to use, allowing runtime switching via the web UI.
        """
        # Read settings from the singleton (respects settings.json overrides)
        self.use_local = config.settings.get(
            "use_local_embeddings",
            config.EMBEDDING_CONFIG["USE_LOCAL"]
        )

        if self.use_local:
            # ----- LOCAL EMBEDDINGS (HuggingFace) -----
            logger.info("Initializing LOCAL Sentence-Transformers model...")
            try:
                # We import lazily so sentence-transformers isn't needed
                # if the user only uses the Gemini API backend.
                from sentence_transformers import SentenceTransformer

                model_name = config.settings.get(
                    "local_model_name",
                    config.EMBEDDING_CONFIG["LOCAL_MODEL_NAME"]
                )
                self.local_model = SentenceTransformer(model_name)
                logger.info(f"Successfully loaded local model: {model_name}")
            except ImportError:
                logger.error(
                    "Failed to import sentence_transformers. "
                    "Did you install requirements.txt?"
                )
                raise
        else:
            # ----- API EMBEDDINGS (Gemini) -----
            logger.info("Initializing GEMINI API Embeddings...")
            api_key = config.settings.get(
                "gemini_api_key",
                config.GEMINI_API_KEY
            )
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY is missing, but config is set to use "
                    "API embeddings! Add it to .env or the Settings page."
                )

            # Configure the Google SDK with our API key
            genai.configure(api_key=api_key)
            self.api_model_name = config.settings.get(
                "api_model_name",
                config.EMBEDDING_CONFIG["API_MODEL_NAME"]
            )

    @time_it
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Convert a list of text chunks into their vector representations.

        This is the most time-intensive step during ingestion — each chunk
        must be mathematically transformed. For 300 chapters with ~15 chunks
        each, that's ~4,500 embedding operations.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of vectors (each vector is a list of floats).
        """
        if not texts:
            return []

        logger.info(f"Generating embeddings for {len(texts)} chunks...")

        if self.use_local:
            # SentenceTransformers .encode() returns a numpy array.
            # ChromaDB requires native Python lists, so we call .tolist().
            embeddings = self.local_model.encode(texts).tolist()
            return embeddings
        else:
            try:
                # The Gemini API can process multiple texts at once (batch mode).
                # task_type="retrieval_document" tells Gemini we're storing
                # these for later similarity search (vs. classification, etc.)
                result = genai.embed_content(
                    model=self.api_model_name,
                    content=texts,
                    task_type="retrieval_document",
                )
                return result['embedding']
            except Exception as e:
                logger.error(f"Gemini API Embedding failed: {str(e)}")
                raise

    @time_it
    def embed_query(self, query: str) -> List[float]:
        """
        Convert a single user question into its vector representation.

        CRITICAL: We must use the SAME embedding model that was used to
        store the novel chunks. If we embedded chapters with MiniLM but
        query with Gemini, the vector spaces won't align and similarity
        search will return irrelevant results.

        Args:
            query: The user's question as a string.

        Returns:
            A single vector (list of floats).
        """
        if self.use_local:
            # .encode() always returns a 2D array, so [0] gets the 1st vector
            return self.local_model.encode([query]).tolist()[0]
        else:
            result = genai.embed_content(
                model=self.api_model_name,
                content=query,
                # "retrieval_query" tells Gemini this is an active question,
                # not a document being stored — subtly different optimization.
                task_type="retrieval_query",
            )
            return result['embedding']
