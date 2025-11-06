import os
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

import google.cloud.storage as storage


def _load_from_local(data_dir: str = "data") -> list[Document]:
    """Load documents from local data directory."""
    docs = []
    p = Path(data_dir)
    
    # PDF
    pdf = p / "Paul_G_CV.pdf"
    if pdf.exists():
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())
    
    # Markdown
    md = p / "context.md"
    if md.exists():
        loader = TextLoader(str(md), encoding="utf-8")
        docs.extend(loader.load())
    
    # XLSX - load entire "Work Stories" sheet as text
    xlsx = p / "work_stories.xlsx"
    if xlsx.exists():
        df = pd.read_excel(xlsx, sheet_name="Work Stories")
        text = df.to_string(index=False)
        docs.append(Document(page_content=text, metadata={"source": "work_stories.xlsx"}))
    
    return docs


def _load_from_gcs(bucket_name: str, prefix: str = "data/") -> list[Document]:
    """Load documents from GCS bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    docs = []
    
    # Download PDF
    blob = bucket.blob(f"{prefix}Paul_G_CV.pdf")
    if blob.exists():
        local_path = f"/tmp/Paul_G_CV.pdf"
        blob.download_to_filename(local_path)
        loader = PyPDFLoader(local_path)
        docs.extend(loader.load())
    
    # Download MD
    blob = bucket.blob(f"{prefix}context.md")
    if blob.exists():
        local_path = f"/tmp/context.md"
        blob.download_to_filename(local_path)
        loader = TextLoader(local_path, encoding="utf-8")
        docs.extend(loader.load())
    
    # Download XLSX
    blob = bucket.blob(f"{prefix}work_stories.xlsx")
    if blob.exists():
        local_path = f"/tmp/work_stories.xlsx"
        blob.download_to_filename(local_path)
        df = pd.read_excel(local_path, sheet_name="Work Stories")
        text = df.to_string(index=False)
        docs.append(Document(page_content=text, metadata={"source": "work_stories.xlsx"}))
    
    return docs


def load_documents() -> list[Document]:
    """Load documents from local (dev) or GCS (prod).
    
    Returns:
        List of Document objects from LangChain
    """
    gcs_bucket = os.getenv("GCS_BUCKET_NAME")
    
    if gcs_bucket:
        return _load_from_gcs(gcs_bucket)
    return _load_from_local()

