from typing import List, Optional
import google.generativeai as genai
import src.config as config
from src.utils import get_logger, time_it

logger = get_logger("Embeddings")

class Embedder:
    """
    This class translates plain English text into an array of numbers (a Vector/Embedding).
    A vector database like ChromaDB doesn't understand words; it only understands numbers.
    By turning text into math, ChromaDB can calculate the 'distance' between a user's question
    and a novel chapter to see how relevant they are to each other.
    
    This class supports switching between Free Local Embeddings and Paid API Embeddings based
    on the config.py file.
    """
    
    def __init__(self):
        self.use_local = config.EMBEDDING_CONFIG.get("USE_LOCAL", True)
        
        # We only load the heavy HuggingFace model if the user explicitly wants to use it.
        # This saves memory if they are using the Gemini API.
        if self.use_local:
            logger.info("Initializing LOCAL Sentence-Transformers model...")
            try:
                # We import it here so it's not a hard requirement if running exclusively API mode
                from sentence_transformers import SentenceTransformer
                model_name = config.EMBEDDING_CONFIG.get("LOCAL_MODEL_NAME", "all-MiniLM-L6-v2")
                self.local_model = SentenceTransformer(model_name)
                logger.info(f"Successfully loaded local model: {model_name}")
            except ImportError:
                logger.error("Failed to import sentence_transformers. Did you install requirements.txt?")
                raise
        else:
            logger.info("Initializing GEMINI API Embeddings...")
            api_key = config.GEMINI_API_KEY
            if not api_key:
                raise ValueError("GEMINI_API_KEY is missing from .env, but config is set to use API embeddings!")
            
            # Configure the Google SDK
            genai.configure(api_key=api_key)
            self.api_model_name = config.EMBEDDING_CONFIG.get("API_MODEL_NAME", "models/gemini-embedding-001")
            
    @time_it
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Takes a list of string chunks and returns a list of numerical vectors.
        This is the longest part of adding new chapters to the database.
        """
        if not texts:
            return []
            
        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        
        if self.use_local:
            # SentenceTransformers is optimized for speed and works offline.
            # .encode() translates the text list into a mathematical array.
            # We call .tolist() because ChromaDB requires native Python lists, not Numpy arrays.
            embeddings = self.local_model.encode(texts).tolist()
            return embeddings
            
        else:
            # We are using Google's servers to calculate the math for us.
            try:
                # The Gemini API takes a list of strings and returns the vectors
                result = genai.embed_content(
                    model=self.api_model_name,
                    content=texts,
                    task_type="retrieval_document", # Tells Gemini we are storing this for a DB search
                )
                
                # Gemini returns a dict. We extract just the numerical lists.
                return result['embedding']
            
            except Exception as e:
                logger.error(f"Gemini API Embedding failed: {str(e)}")
                raise

    @time_it
    def embed_query(self, query: str) -> List[float]:
        """
        Similar to embed_texts, but explicitly for the user's question.
        We must translate the user's question into math using the EXACT SAME model 
        that we used to translate the novel chapters, or the math won't line up.
        """
        if self.use_local:
            # Even for a single string, we return it as a flat list
            return self.local_model.encode([query]).tolist()[0]
        else:
            result = genai.embed_content(
                model=self.api_model_name,
                content=query,
                task_type="retrieval_query", # Tells Gemini this is an active user question
            )
            return result['embedding']
