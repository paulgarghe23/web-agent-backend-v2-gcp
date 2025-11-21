import os

import google.auth
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_core.documents import Document

from app.utils.document_loader import load_documents

_, project_id = google.auth.default()
LOCATION = os.getenv("LOCATION", "europe-southwest1")

_vector_store = None


def get_embeddings() -> VertexAIEmbeddings:
    return VertexAIEmbeddings(
        model_name="text-embedding-004",
        project=project_id,
        location=LOCATION,
    )


def get_vector_store():
    """Get or create vector store with documents."""
    global _vector_store
    if _vector_store is None:
        docs = load_documents()
        
        # Ensure all documents have string page_content before splitting
        normalized_docs = []
        for doc in docs:
            content = doc.page_content
            if isinstance(content, list):
                content = "\n".join(str(item) for item in content)
            elif not isinstance(content, str):
                content = str(content)
            normalized_docs.append(Document(page_content=content, metadata=doc.metadata))
        
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(normalized_docs)
        embeddings = get_embeddings()
        _vector_store = FAISS.from_documents(chunks, embeddings)
    return _vector_store

