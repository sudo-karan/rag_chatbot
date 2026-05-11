from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.ingestion import ingest_pdfs
from app.session import create_session, delete_session, get_session, list_sessions
from app.chat import process_message, get_disclaimer
from app.llm import is_ollama_available


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not is_ollama_available():
        print("WARNING: Ollama is not running or model not found.")
        print("Run: ollama serve && ollama pull llama3.1:8b")
    print("Starting up - ingesting PDFs...")
    ingest_pdfs()
    print("Ready.")
    yield


app = FastAPI(
    title="Government Portal RAG Chatbot API",
    description="Fully offline RAG chatbot for the Government Open Data Portal",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    disclaimer_accepted: bool


class SessionResponse(BaseModel):
    session_id: str
    disclaimer_text: str


@app.get("/health")
def health():
    from app.vector_store import collection_count
    return {
        "status": "ok",
        "ollama_available": is_ollama_available(),
        "vector_store_chunks": collection_count(),
    }


@app.post("/session/create", response_model=SessionResponse)
def create_new_session():
    session = create_session()
    return SessionResponse(session_id=session.session_id, disclaimer_text=get_disclaimer())


@app.post("/chat/{session_id}", response_model=ChatResponse)
def chat(session_id: str, request: ChatRequest):
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Create one via POST /session/create")
    response = process_message(session_id, request.message)
    return ChatResponse(
        session_id=session_id,
        response=response,
        disclaimer_accepted=session.disclaimer_accepted,
    )


@app.delete("/session/{session_id}")
def end_session(session_id: str):
    if not delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"message": "Session ended", "session_id": session_id}


@app.get("/sessions")
def get_all_sessions():
    return {"active_sessions": list_sessions()}
