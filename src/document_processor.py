import os
import glob
from typing import List, Dict
import src.config as config
from src.utils import get_logger, time_it

logger = get_logger("DocumentProcessor")

class DocumentProcessor:
    """
    This class is responsible for reading your raw markdown novel chapters
    and preparing them for the Vector Database.
    
    The main job is "Chunking" - we can't feed an entire 1888-word chapter
    into the database as a single item because it dilutes the relevance of specific facts.
    We chop it into smaller pieces.
    """
    
    def __init__(self, chunk_size: int = config.CHUNK_SIZE, chunk_overlap: int = config.CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    def _split_text(self, text: str) -> List[str]:
        """
        Takes a long string of text and splits it into smaller chunks of length `chunk_size`.
        Crucially, each chunk overlaps the previous one by `chunk_overlap` characters.
        This overlap is vital! If a character's name is cut off right at the end of Chunk A,
        the overlap ensures they are fully present at the beginning of Chunk B.
        """
        # If the text is shorter than a single chunk, just return it as is.
        if len(text) <= self.chunk_size:
            return [text]
            
        chunks = []
        # We start at the beginning (0) and move forward by (size - overlap) each time
        # This creates the sliding window effect.
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
        Reads all markdown files in the given directory, splits them into chunks,
        and attaches metadata (like the source filename) to each chunk.
        
        Metadata is how the LLM knows *where* a particular fact came from!
        """
        logger.info(f"Starting to process markdown files in: {data_dir}")
        
        # Find all .md files in the data directory and any subdirectories
        search_pattern = os.path.join(data_dir, "**", "*.md")
        filepaths = glob.glob(search_pattern, recursive=True)
        
        if not filepaths:
            logger.warning(f"No markdown files found in {data_dir}. Did you forget to add your chapters?")
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
                    
                # Chop the text into our overlapping chunks
                chunks = self._split_text(content)
                total_chunks += len(chunks)
                
                # For every chunk created, we store the text AND where it came from
                for i, chunk in enumerate(chunks):
                    doc = {
                        "id": f"{filename}-chunk-{i}",
                        "text": chunk,
                        # Metadata is extremely important for RAG. It allows us to filter later,
                        # or provide citations to the user ("This happened in Chapter 12")
                        "metadata": {
                            "source": filename,
                            "chunk_index": i
                        }
                    }
                    processed_documents.append(doc)
                    
            except Exception as e:
                logger.error(f"Failed to process file {filename}: {str(e)}")
                
        logger.info(f"Successfully processed {total_files} files into {total_chunks} total chunks.")
        
        return processed_documents
