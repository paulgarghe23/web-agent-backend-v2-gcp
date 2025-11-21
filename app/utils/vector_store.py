import os
import logging
import time

import google.auth
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.documents import Document

from app.utils.document_loader import load_documents

logger = logging.getLogger(__name__)

_, project_id = google.auth.default()
LOCATION = os.getenv("LOCATION", "europe-southwest1")

_vector_store = None


def get_embeddings() -> VertexAIEmbeddings:
    logger.info("EMBEDDINGS_INITIALIZED", extra={
        "event": "embeddings_initialized",
        "model_name": "text-embedding-004",
        "project_id": project_id,
        "location": LOCATION,
        "embedding_dimension": 768,  # text-embedding-004 dimension
    })
    return VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=project_id,
        location=LOCATION,
    )


def get_vector_store():
    """Get or create vector store with documents."""
    global _vector_store
    if _vector_store is None:
        store_creation_start = time.time()
        
        logger.info("VECTOR_STORE_CREATION_START", extra={
            "event": "vector_store_creation_started",
        })
        
        # Load documents
        docs_load_start = time.time()
        docs = load_documents()
        docs_load_time = time.time() - docs_load_start
        
        logger.info("DOCUMENTS_LOADED", extra={
            "event": "documents_loaded",
            "documents_count": len(docs),
            "total_chars": sum(len(doc.page_content) for doc in docs),
            "load_time_seconds": round(docs_load_time, 3),
            "sources": [doc.metadata.get("source", "unknown") for doc in docs],
        })
        
        # Log each document
        for i, doc in enumerate(docs):
            logger.info("DOCUMENT_DETAIL", extra={
                "event": "document_detail",
                "document_index": i + 1,
                "document_length": len(doc.page_content),
                "document_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata,
            })
        
        # Ensure all documents have string page_content before splitting
        normalized_docs = []
        for doc in docs:
            content = doc.page_content
            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)
            elif not isinstance(content, str):
                content = str(content)
            normalized_docs.append(Document(page_content=content, metadata=doc.metadata))
        
        # Split documents
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_start = time.time()
        chunks = splitter.split_documents(normalized_docs)
        split_time = time.time() - split_start
        
        logger.info("DOCUMENTS_SPLIT", extra={
            "event": "documents_split",
            "original_docs": len(normalized_docs),
            "chunks_created": len(chunks),
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "split_time_seconds": round(split_time, 3),
            "avg_chunk_length": round(sum(len(chunk.page_content) for chunk in chunks) / len(chunks), 2) if chunks else 0,
        })
        
        # Get embeddings
        embeddings = get_embeddings()
        
        # Create vector store
        vector_store_start = time.time()
        _vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store_time = time.time() - vector_store_start
        
        total_time = time.time() - store_creation_start
        
        # Get store info
        try:
            # FAISS stores vectors, we can get some info
            index = _vector_store.index if hasattr(_vector_store, 'index') else None
            vector_count = index.ntotal if index and hasattr(index, 'ntotal') else len(chunks)
        except:
            vector_count = len(chunks)
        
        logger.info("VECTOR_STORE_CREATED", extra={
            "event": "vector_store_created",
            "vector_count": vector_count,
            "chunks_count": len(chunks),
            "embedding_model": "text-embedding-004",
            "embedding_dimension": 768,
            "vector_store_time_seconds": round(vector_store_time, 3),
            "total_creation_time_seconds": round(total_time, 3),
        })
    else:
        logger.debug("VECTOR_STORE_REUSED", extra={
            "event": "vector_store_reused",
        })
    
    return _vector_store

