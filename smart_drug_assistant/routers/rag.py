"""
RAG (Retrieval-Augmented Generation) router for medical information queries.
Provides endpoints for querying medical leaflets using LLM with retrieval.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from smart_drug_assistant.embeddings_setup import get_vector_store
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading
import asyncio
import os

router = APIRouter(prefix="/rag", tags=["rag"])

# Lazy initialization
_vector_store = None
_retriever = None
_llm = None
_qa_chain = None
_lock = threading.Lock()

executor = ThreadPoolExecutor(max_workers=2)


def _initialize_rag():
    """Initialize RAG components lazily"""
    global _vector_store, _retriever, _llm, _qa_chain
    
    if _qa_chain is not None:
        return
    
    with _lock:
        if _qa_chain is not None:
            return
        
        print("Initializing RAG components...")
        # Get Ollama base URL from environment variable
        ollama_base_url = os.getenv("OLLAMA_HOST", "http://ollama:11434")
        
        _vector_store = get_vector_store()
        _retriever = _vector_store.as_retriever(search_kwargs={"k": 1})
        _llm = Ollama(
            model="neural-chat",
            base_url=ollama_base_url,
            temperature=0.05,
            num_ctx=512,
            num_predict=150
        )
        
        # Initialize QA chain with prompt
        prompt_template = """Answer ONLY from the leaflet context. Be concise.

Context:
{context}

Question: {question}

Answer:
"""
        
        QA_PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        _qa_chain = RetrievalQA.from_chain_type(
            llm=_llm,
            retriever=_retriever,
            chain_type="stuff",
            return_source_documents=False,
            chain_type_kwargs={"prompt": QA_PROMPT},
            input_key="query",
            output_key="result"
        )
        print("RAG components initialized!")


def get_qa_chain():
    """Get the QA chain, initializing if needed"""
    if _qa_chain is None:
        _initialize_rag()
    return _qa_chain

# Simple in-memory cache for query results
query_cache = {}


def get_cache_key(query: str, k: int) -> str:
    """Generate cache key from query and k parameter"""
    return hashlib.md5(f"{query}:{k}".encode()).hexdigest()


# ======================
# Request/Response Models
# ======================
class QueryRequest(BaseModel):
    """Request model for RAG queries"""
    query: str
    k: int = 2


class QueryResponse(BaseModel):
    """Response model for RAG queries"""
    answer: str


# ======================
# API Endpoints
# ======================
@router.post("/query", response_model=QueryResponse)
async def query_rag(req: QueryRequest):
    """
    Query the medical leaflet RAG system with caching and async processing.
    
    Args:
        req: QueryRequest containing the query string and optional k (number of documents)
        
    Returns:
        QueryResponse with the answer from the RAG system
        
    Raises:
        HTTPException: If an error occurs during query processing
    """
    try:
        # Check cache first
        cache_key = get_cache_key(req.query, req.k)
        if cache_key in query_cache:
            return QueryResponse(answer=query_cache[cache_key])
        
        # Get QA chain
        qa_chain = get_qa_chain()
        
        # Run in thread pool to avoid blocking
        def run_query():
            return qa_chain({"query": req.query})
        
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, run_query)
        
        answer_text = result["result"]

        # Store in cache
        query_cache[cache_key] = answer_text

        return QueryResponse(answer=answer_text)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))