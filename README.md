# 📚 Novel RAG Tutorial Project

Welcome to your first Retrieval-Augmented Generation (RAG) system! This project is designed specifically for learning the core mechanics of how an LLM can answer questions about your private markdown files without relying on magic, overly-complex frameworks.

## 🚀 Quick Start Guide

### 1. Installation
Ensure you have Python 3.9+ installed. Follow these commands in your terminal:

```bash
# Optimal but recommended: Create a virtual environment
python -m venv venv
source venv/bin/activate  # Or `venv\Scripts\activate` on Windows

# Install the dependencies
pip install -r requirements.txt
```

### 2. Configuration & API Keys
1. Copy the template environment file:
   `cp .env.example .env`
2. Open `.env` and paste your Google Gemini API key:
   `GEMINI_API_KEY="your-api-key-here"`

### 3. Adding Novels
This system supports **multiple distinct novels**. Each novel gets its own isolated database so characters from "Harry Potter" don't get mixed up with characters from "Lord of the Rings".

To add your novel:
1. Create a folder inside the `data/` directory. The folder name will be your novel's name.
2. Put your `.md` markdown chapters inside that folder.

*Example directory structure:*
```text
novel_rag/
└── data/
    ├── became-the-patron-of-villains/
    │   ├── 001-chapter-1.md
    │   └── 002-chapter-2.md
    └── another-epic-fantasy/
        ├── prologue.md
        └── chapter-1.md
```

### 4. Running the App
You only need one command to run everything! Make sure to use the python binary inside your `.venv` to avoid version conflicts (the app requires Python 3).

```bash
# From the project root (novel_rag/):
./.venv/bin/python src/app.py
```

**What happens next?**
1. The app detects the folders in `data/` and asks you to select a novel.
2. **First Time Setup:** If you select a novel that has never been processed, the system will automatically detect this and run the "Ingestion Pipeline". It will chunk the markdown texts, embed them, and save them to the database. *(This takes a few minutes).*
3. **Chatting:** Once ingestion finishes (or if it was already processed previously), you enter the chat loop. Ask away!

---

## 🔁 Use Cases & Updating Data

### How to add entirely new novels
Simply create a new folder in `data/` and add your `.md` files. When you run `app.py`, it will appear in the selection menu. The script will handle creating a new Database collection for it automatically.

### How to add new chapters to an existing novel
If you drop new `.md` files into a novel's folder that has already been processed in the past, the database won't automatically know about them. 

To force the system to re-read the files and update the database:
1. Run `python src/app.py` and select the novel.
2. When the chat prompt appears, type: `!reingest`
3. The system will process everything and update your chunks.

---

## 🧠 Educational Details: How This Works

This application is split into highly modular parts so you can study exactly how the data flows.

### 1. Chunking & Strategy (`src/document_processor.py`)
An average chapter in this novel contains roughly **1,888 words (or ~10,000 characters)**. You might wonder: *Why not just insert an entire chapter into the database as a single chunk?*

**Why not whole chapters?**
* **Relevance Dilution:** If a chapter covers a sword fight, a political debate, and a quiet character moment, a single mathematical 'vector' representing all three will be muddled. It won't strongly match a specific user question about just the sword fight.
* **Token Limits & Cost:** Passing entire 10,000-character chunks to the LLM for every question burns through API limits and slows down generation times.
* **Retrieval Precision:** RAG is designed to fetch the *exact* paragraphs relevant to your question.

**Why 600 characters with 150-character overlap?**
* **600 characters (~100-150 words)** is the 'Goldilocks zone'. It's about the size of a single descriptive paragraph or a quick back-and-forth dialogue. This ensures the resulting vector is highly specific to one event or interaction.
* **150-character overlap** is a safeguard context window. Novels often replace character names with pronouns across paragraphs. If Chunk A says "Yutia grabbed her sword," and Chunk B says "She swung it at the demon," an overlap ensures Chunk B still contains the word "Yutia," so the database and the LLM know who "She" is related to.

### 2. Embeddings (`src/embeddings.py` & `src/config.py`)
An Embedding is the process of translating plain text into lists of numbers (a vector) so a computer can quickly calculate how similar two sentences are. 
* **The Switch:** Open `src/config.py`. By default, `USE_LOCAL` is set to `True`. Because you have a capable processor, we use HuggingFace (`all-MiniLM-L6-v2`) to generate the math for free locally. 
* Change it to `False` to use the Gemini Embedding API instead (uses your quota, but perfectly matches Gemini's understanding).

### 3. Vector Database (`src/vector_db.py`)
We use **ChromaDB**. It stores the markdown text *alongside* the numerical vectors we generated. Based on your prompt selection, we create "Collections" (like tables) dynamically for each novel.

### 4. Generation Pipeline (`src/app.py`)
We don't just ask Gemini your question blindly. We use "Prompt Engineering" to artificially restrict Gemini. We give it the 7 closest markdown chunks we found, and explicitly tell it: *"Answer the user's question using ONLY these specific chunks, and cite your sources."*
