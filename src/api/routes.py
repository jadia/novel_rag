"""
=============================================================================
API Routes — REST & WebSocket Endpoints
=============================================================================
This module defines all the HTTP and WebSocket endpoints that the front-end
communicates with.

ARCHITECTURE:
  REST endpoints  → Standard request/response for CRUD operations
  WebSocket endpoints → Persistent bidirectional connections for streaming

WHY WEBSOCKETS FOR CHAT?
When the LLM generates a response, it can take 5-30 seconds. Instead of
making the user stare at a spinner, we stream tokens as they're generated.
A WebSocket connection stays open, allowing us to push each token to the
browser the instant it's available.

WHY WEBSOCKETS FOR INGESTION?
Embedding 300 chapters can take several minutes. A WebSocket lets us send
progress updates ("Batch 5 of 45 complete") in real-time, so the user can
watch a progress bar fill up instead of wondering if the app is frozen.

ENDPOINT OVERVIEW:
  GET  /api/novels           → List available novels
  GET  /api/novels/{n}/status → Check if a novel has embeddings
  GET  /api/sessions/{novel} → List chat sessions for a novel
  GET  /api/sessions/{id}/messages → Get messages for a session
  POST /api/sessions         → Create a new session
  DELETE /api/sessions/{id}  → Delete a session
  GET  /api/settings         → Get current settings (keys masked)
  PUT  /api/settings         → Update runtime settings
  WS   /ws/chat              → Streaming chat with RAG
  WS   /ws/ingest/{novel}    → Streaming ingestion progress
=============================================================================
"""

import os
import json
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
from typing import Optional
import google.generativeai as genai

import src.core.config as config
from src.core.document_processor import DocumentProcessor
from src.core.vector_db import VectorDB
from src.core.utils import get_logger
from src.api.chat_store import ChatStore

logger = get_logger("Routes")

# ---------------------------------------------------------------------------
# Create an APIRouter instead of defining routes directly on the FastAPI app.
# This is the "Router" pattern — it lets us organize routes in separate files
# and mount them onto the main app in main.py.
# ---------------------------------------------------------------------------
router = APIRouter()

# ---------------------------------------------------------------------------
# Singletons: We create ONE chat store instance shared across all requests.
# In a multi-user app, you'd use dependency injection instead.
# ---------------------------------------------------------------------------
chat_store = ChatStore()


# ===========================================================================
# PYDANTIC MODELS — Request/Response Validation
# ===========================================================================
# Pydantic models define the shape of JSON data the API accepts/returns.
# FastAPI automatically validates incoming requests against these models
# and generates API documentation (Swagger UI at /docs).
# ===========================================================================

class SessionCreate(BaseModel):
    """Request body for creating a new chat session."""
    novel_name: str


class SettingsUpdate(BaseModel):
    """Request body for updating runtime settings."""
    gemini_api_key: Optional[str] = None
    use_local_embeddings: Optional[bool] = None
    local_model_name: Optional[str] = None
    api_model_name: Optional[str] = None
    generation_model: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


# ===========================================================================
# REST ENDPOINTS — Standard Request/Response
# ===========================================================================

@router.get("/api/novels")
async def list_novels():
    """
    List all available novels (directories in the data/ folder).

    Returns a list of objects with the novel name and whether it has
    been ingested (has embeddings in the database).
    """
    novels = config.get_available_novels()
    result = []
    for novel in novels:
        # Check if this novel has been processed by looking at ChromaDB.
        # We create a temporary VectorDB instance just for the check.
        try:
            db = VectorDB(collection_name=novel)
            has_embeddings = db.has_documents()
        except Exception:
            has_embeddings = False

        result.append({
            "name": novel,
            "has_embeddings": has_embeddings,
        })

    return {"novels": result}


@router.get("/api/novels/{novel_name}/status")
async def novel_status(novel_name: str):
    """
    Check if a specific novel has been ingested.

    Returns the number of chunks stored in the database.
    """
    novels = config.get_available_novels()
    if novel_name not in novels:
        raise HTTPException(status_code=404, detail=f"Novel '{novel_name}' not found")

    try:
        db = VectorDB(collection_name=novel_name)
        count = db.collection.count()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "novel_name": novel_name,
        "has_embeddings": count > 0,
        "chunk_count": count,
    }


# ---------------------------------------------------------------------------
# SESSION ENDPOINTS — CRUD for Chat History
# ---------------------------------------------------------------------------

@router.post("/api/sessions")
async def create_session(data: SessionCreate):
    """Create a new chat session for a specific novel."""
    session_id = chat_store.create_session(data.novel_name)
    return {"session_id": session_id}


@router.get("/api/sessions/{novel_name}")
async def list_sessions(novel_name: str):
    """List all chat sessions for a specific novel, newest first."""
    sessions = chat_store.list_sessions(novel_name)
    return {"sessions": sessions}


@router.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """
    Get all messages for a specific session.

    Returns the session metadata plus the ordered list of messages.
    """
    session = chat_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = chat_store.get_session_messages(session_id)
    return {
        "session": session,
        "messages": messages,
    }


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a chat session and all its messages."""
    deleted = chat_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True}


# ---------------------------------------------------------------------------
# SETTINGS ENDPOINTS — Runtime Configuration
# ---------------------------------------------------------------------------

@router.get("/api/settings")
async def get_settings():
    """
    Get current settings with API keys masked for security.

    The front-end displays these values in the Settings modal.
    Keys are masked (e.g., "sk12***ef90") so they can't be stolen
    from the browser's network tab or a screenshot.
    """
    return {"settings": config.settings.get_all()}


@router.put("/api/settings")
async def update_settings(data: SettingsUpdate):
    """
    Update runtime settings and persist to settings.json.

    Only non-None fields are updated, so the front-end can send
    partial updates (e.g., only change the generation model).
    """
    # Convert Pydantic model to dict, excluding None values
    # This ensures we only update fields the user actually changed
    updates = {k: v for k, v in data.model_dump().items() if v is not None}

    if not updates:
        raise HTTPException(status_code=400, detail="No settings to update")

    config.settings.update(updates)
    return {"settings": config.settings.get_all(), "saved": True}


# ===========================================================================
# WEBSOCKET ENDPOINTS — Real-Time Streaming
# ===========================================================================

@router.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket):
    """
    WebSocket endpoint for streaming RAG chat responses.

    PROTOCOL:
    Client sends a JSON message:
      { "session_id": "...", "novel_name": "...", "question": "..." }

    Server responds with a stream of JSON messages:
      { "type": "token",  "content": "partial text..." }  ← repeated
      { "type": "done",   "content": "full answer text" }  ← final message
      { "type": "error",  "content": "error description" }  ← if something fails

    CONNECTION LIFECYCLE:
      1. Client opens WebSocket connection
      2. Client sends question(s) — one at a time
      3. Server streams back tokens for each question
      4. Connection stays open for follow-up questions
      5. Client closes when switching novels or navigating away
    """
    await ws.accept()
    logger.info("WebSocket chat connection opened")

    try:
        # Connection stays open for multiple question/answer cycles
        while True:
            # Wait for the next question from the client
            raw = await ws.receive_text()
            data = json.loads(raw)

            session_id = data.get("session_id")
            novel_name = data.get("novel_name")
            question = data.get("question", "").strip()

            if not question or not novel_name:
                await ws.send_json({
                    "type": "error",
                    "content": "Missing 'novel_name' or 'question' field"
                })
                continue

            try:
                # ---- STEP 1: Store the user's message ----
                if session_id:
                    chat_store.add_message(session_id, "user", question)

                # ---- STEP 2: Retrieve relevant chunks from ChromaDB ----
                db = VectorDB(collection_name=novel_name)
                retrieval_results = db.query(question, n_results=7)

                # Format chunks into the prompt context
                context = _format_context(retrieval_results)

                # ---- STEP 3: Stream the LLM response ----
                # Configure Gemini with the current API key from settings
                api_key = config.settings.get(
                    "gemini_api_key", config.GEMINI_API_KEY
                )
                genai.configure(api_key=api_key)

                model_name = config.settings.get(
                    "generation_model", config.GENERATION_MODEL
                )
                model = genai.GenerativeModel(model_name)

                # Build the RAG prompt (same template as the CLI)
                prompt = _build_rag_prompt(question, context)

                # generate_content with stream=True returns an iterator
                # that yields response chunks as they're generated
                response = model.generate_content(prompt, stream=True)

                full_answer = ""
                for chunk in response:
                    # Each chunk may contain a piece of the answer
                    if chunk.text:
                        full_answer += chunk.text
                        # Send each token to the client immediately
                        await ws.send_json({
                            "type": "token",
                            "content": chunk.text
                        })
                        # Yield control to the event loop so the WebSocket
                        # buffer can flush (important for responsiveness)
                        await asyncio.sleep(0)

                # ---- STEP 4: Store the complete answer ----
                if session_id:
                    chat_store.add_message(session_id, "assistant", full_answer)

                # Send final "done" message so the client knows streaming is over
                await ws.send_json({
                    "type": "done",
                    "content": full_answer,
                    "session_id": session_id,
                })

            except Exception as e:
                logger.error(f"Chat error: {str(e)}")
                await ws.send_json({
                    "type": "error",
                    "content": f"Error generating answer: {str(e)}"
                })

    except WebSocketDisconnect:
        logger.info("WebSocket chat connection closed")
    except Exception as e:
        logger.error(f"WebSocket unexpected error: {str(e)}")


@router.websocket("/ws/ingest/{novel_name}")
async def websocket_ingest(ws: WebSocket, novel_name: str):
    """
    WebSocket endpoint for streaming ingestion progress.

    PROTOCOL:
    Client opens connection → server immediately starts ingestion.
    Server sends progress messages:
      { "type": "progress", "current": 150, "total": 4500, "message": "..." }
      { "type": "done",     "message": "Ingestion complete!", "chunk_count": 4500 }
      { "type": "error",    "content": "error description" }

    WHY NOT A REST ENDPOINT?
    Ingestion can take 2-10 minutes depending on the number of chapters.
    With a REST endpoint, the HTTP connection would timeout. With WebSockets,
    we maintain an open connection and send real-time progress updates that
    the front-end renders as a progress bar.
    """
    await ws.accept()
    logger.info(f"WebSocket ingest connection opened for '{novel_name}'")

    try:
        # Validate the novel exists
        novels = config.get_available_novels()
        if novel_name not in novels:
            await ws.send_json({
                "type": "error",
                "content": f"Novel '{novel_name}' not found"
            })
            await ws.close()
            return

        novel_dir = os.path.join(config.DATA_DIR, novel_name)

        # ---- STEP 1: Chunk the markdown files ----
        await ws.send_json({
            "type": "progress",
            "current": 0,
            "total": 100,
            "message": "Reading and chunking markdown files..."
        })

        processor = DocumentProcessor(
            chunk_size=config.settings.get("chunk_size", config.CHUNK_SIZE),
            chunk_overlap=config.settings.get("chunk_overlap", config.CHUNK_OVERLAP),
        )

        # Run the blocking process_directory in a thread pool to not
        # block the async event loop (CPU-bound work)
        documents = await asyncio.to_thread(
            processor.process_directory, novel_dir
        )

        if not documents:
            await ws.send_json({
                "type": "error",
                "content": f"No markdown documents found in {novel_dir}"
            })
            await ws.close()
            return

        await ws.send_json({
            "type": "progress",
            "current": 0,
            "total": len(documents),
            "message": f"Found {len(documents)} chunks. Starting embedding..."
        })

        # ---- STEP 2: Embed and store with progress ----
        # We define a callback that the VectorDB calls after each batch.
        # The callback sends a WebSocket message with the current progress.
        loop = asyncio.get_event_loop()

        def progress_callback(current, total, message):
            """
            Called by VectorDB.store_documents() after each batch.

            NOTE: This runs in a thread pool (blocking context), so we
            use asyncio.run_coroutine_threadsafe() to safely send the
            WebSocket message from a non-async context.
            """
            future = asyncio.run_coroutine_threadsafe(
                ws.send_json({
                    "type": "progress",
                    "current": current,
                    "total": total,
                    "message": message
                }),
                loop
            )
            # Wait for the send to complete (with timeout)
            future.result(timeout=10)

        db = VectorDB(collection_name=novel_name)

        # Run the embedding + storage in a thread pool (CPU/IO-bound)
        await asyncio.to_thread(
            db.store_documents, documents, progress_callback
        )

        # ---- STEP 3: Report completion ----
        chunk_count = db.collection.count()
        await ws.send_json({
            "type": "done",
            "message": f"Ingestion complete! {chunk_count} chunks stored.",
            "chunk_count": chunk_count
        })

    except WebSocketDisconnect:
        logger.info("WebSocket ingest connection closed by client")
    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        try:
            await ws.send_json({
                "type": "error",
                "content": f"Ingestion failed: {str(e)}"
            })
        except Exception:
            pass  # Client already disconnected
    finally:
        try:
            await ws.close()
        except Exception:
            pass


# ===========================================================================
# HELPER FUNCTIONS — Shared logic used by the endpoints above
# ===========================================================================

def _format_context(retrieved_results: dict) -> str:
    """
    Format ChromaDB retrieval results into a readable context string.

    This is the same formatting used by the CLI — we extract the text
    and metadata from the retrieval results and format them as labeled
    excerpts that the LLM can reference.
    """
    docs = retrieved_results['documents']
    metas = retrieved_results['metadatas']

    formatted = ""
    for idx, (doc, meta) in enumerate(zip(docs, metas)):
        source = meta.get('source', 'Unknown')
        formatted += f"--- Excerpt {idx + 1} (Source: {source}) ---\n"
        formatted += f"{doc}\n\n"

    return formatted


def _build_rag_prompt(question: str, context: str) -> str:
    """
    Build the full RAG prompt with the system instructions, context, and question.

    This is the "Prompt Engineering" step — we carefully instruct the LLM to:
      1. Only use the provided excerpts (no hallucination)
      2. Cite sources for every claim
      3. Admit when information is missing
      4. Focus on character relationships and actions
    """
    return f"""You are an expert lore-keeper and character historian for a specific novel. 
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
