import os
from dotenv import load_dotenv

# Load the environment variables from the .env file
load_dotenv(".env.paid", override=True)

# --- GEMINI API ---
# We retrieve the key from our local .env file securely.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- CHUNKING STRATEGY ---
# Because our novel text needs to be broken apart:
# 600 characters is roughly 100-150 words (a few paragraphs).
# 150 overlap ensures that if a conversation spans across two chunks, the context isn't lost.
CHUNK_SIZE = 600
CHUNK_OVERLAP = 150

# --- EMBEDDINGS (THE SWITCH) ---
# This dictionary controls whether we use free local embeddings or paid API embeddings.
# Set "USE_LOCAL" to True to process text locally (free, but slightly slower).
# Set "USE_LOCAL" to False to use Gemini embeddings (costs quota, but highly accurate).
EMBEDDING_CONFIG = {
    "USE_LOCAL": False,  # Change this to False to test the Gemini API Embeddings
    
    # If using local embeddings, this specifies the open-source HuggingFace model.
    # all-MiniLM-L6-v2 is an excellent balance of speed and accuracy for basic retrieval.
    "LOCAL_MODEL_NAME": "all-MiniLM-L6-v2", 
    
    # If using API embeddings, this specifies which Gemini model to hit.
    # "API_MODEL_NAME": "models/embedding-001"
    "API_MODEL_NAME": "models/gemini-embedding-001"
}

# --- DIRECTORIES ---
# Absolute paths keep things clean so the script works no matter where you run it.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where you will place all your chapter markdown files
# We expect subfolders here, e.g. data/became-the-patron-of-villains/, data/another-novel/
DATA_DIR = os.path.join(BASE_DIR, "data") 

# Where ChromaDB will save its local database files.
DB_DIR = os.path.join(BASE_DIR, "db")

def get_available_novels() -> list:
    """Scans the data directory and returns a list of folder names (novels)."""
    if not os.path.exists(DATA_DIR):
        return []
    
    # List all items in DATA_DIR that are directories
    novels = [d for d in os.listdir(DATA_DIR) 
              if os.path.isdir(os.path.join(DATA_DIR, d))]
    
    return sorted(novels)
