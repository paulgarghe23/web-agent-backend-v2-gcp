import os
import logging
from pathlib import Path

import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document

import google.cloud.storage as storage

logger = logging.getLogger(__name__)


def _load_from_local(data_dir: str = "data") -> list[Document]:
    """Load documents from local data directory."""
    logger.info("LOADING_DOCUMENTS_FROM_LOCAL", extra={
        "event": "loading_documents_local",
        "data_dir": data_dir,
    })
    
    docs = []
    p = Path(data_dir)
    
    # PDF
    pdf = p / "Paul_G_CV.pdf"
    if pdf.exists():
        logger.info("LOADING_PDF", extra={
            "event": "loading_pdf",
            "file_path": str(pdf),
            "file_size_bytes": pdf.stat().st_size if pdf.exists() else 0,
        })
        loader = PyPDFLoader(str(pdf))
        pdf_docs = loader.load()
        
        # Clean PDF text: remove excessive newlines (PyPDF sometimes extracts with \n after each word)
        for doc in pdf_docs:
            # Replace "\n \n" with space, then collapse multiple spaces
            doc.page_content = doc.page_content.replace('\n \n', ' ').replace('\n', ' ')
            # Collapse multiple spaces into one
            doc.page_content = ' '.join(doc.page_content.split())
        
        docs.extend(pdf_docs)
        logger.info("PDF_LOADED", extra={
            "event": "pdf_loaded",
            "file_path": str(pdf),
            "pages": len(pdf_docs),
            "total_chars": sum(len(doc.page_content) for doc in pdf_docs),
        })
    else:
        logger.warning("PDF_NOT_FOUND", extra={"file_path": str(pdf)})
    
    # Markdown
    md = p / "context.md"
    if md.exists():
        logger.info("LOADING_MARKDOWN", extra={
            "event": "loading_markdown",
            "file_path": str(md),
            "file_size_bytes": md.stat().st_size if md.exists() else 0,
        })
        loader = TextLoader(str(md), encoding="utf-8")
        md_docs = loader.load()
        docs.extend(md_docs)
        logger.info("MARKDOWN_LOADED", extra={
            "event": "markdown_loaded",
            "file_path": str(md),
            "total_chars": sum(len(doc.page_content) for doc in md_docs),
        })
    else:
        logger.warning("MARKDOWN_NOT_FOUND", extra={"file_path": str(md)})
    
    # XLSX - load entire "Work Stories" sheet as text
    xlsx = p / "work_stories.xlsx"
    if xlsx.exists():
        logger.info("LOADING_XLSX", extra={
            "event": "loading_xlsx",
            "file_path": str(xlsx),
            "file_size_bytes": xlsx.stat().st_size if xlsx.exists() else 0,
        })
        df = pd.read_excel(xlsx, sheet_name="Work Stories")
        # Extract text compactly: iterate rows and columns, skip empty cells
        text_lines = []
        for idx, row in df.iterrows():
            row_parts = []
            for col in df.columns:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    row_parts.append(str(value).strip())
            if row_parts:
                text_lines.append(" ".join(row_parts))
        text = "\n".join(text_lines)
        xlsx_doc = Document(page_content=text, metadata={"source": "work_stories.xlsx"})
        docs.append(xlsx_doc)
        logger.info("XLSX_LOADED", extra={
            "event": "xlsx_loaded",
            "file_path": str(xlsx),
            "rows": len(df),
            "columns": len(df.columns),
            "total_chars": len(text),
        })
    else:
        logger.warning("XLSX_NOT_FOUND", extra={"file_path": str(xlsx)})
    
    logger.info("LOCAL_DOCUMENTS_LOADED", extra={
        "event": "local_documents_loaded",
        "total_documents": len(docs),
        "total_chars": sum(len(doc.page_content) for doc in docs),
    })
    
    return docs


def _load_from_gcs(bucket_name: str, prefix: str = "data/") -> list[Document]:
    """Load documents from GCS bucket."""
    logger.info("LOADING_DOCUMENTS_FROM_GCS", extra={
        "event": "loading_documents_gcs",
        "bucket_name": bucket_name,
        "prefix": prefix,
    })
    
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    docs = []
    
    # Download PDF
    blob = bucket.blob(f"{prefix}Paul_G_CV.pdf")
    if blob.exists():
        logger.info("DOWNLOADING_PDF_FROM_GCS", extra={
            "event": "downloading_pdf_gcs",
            "blob_path": f"{prefix}Paul_G_CV.pdf",
            "blob_size_bytes": blob.size,
        })
        local_path = f"/tmp/Paul_G_CV.pdf"
        blob.download_to_filename(local_path)
        loader = PyPDFLoader(local_path)
        pdf_docs = loader.load()
        docs.extend(pdf_docs)
        logger.info("PDF_DOWNLOADED_FROM_GCS", extra={
            "event": "pdf_downloaded_gcs",
            "pages": len(pdf_docs),
            "total_chars": sum(len(doc.page_content) for doc in pdf_docs),
        })
    else:
        logger.warning("PDF_NOT_FOUND_IN_GCS", extra={"blob_path": f"{prefix}Paul_G_CV.pdf"})
    
    # Download MD
    blob = bucket.blob(f"{prefix}context.md")
    if blob.exists():
        logger.info("DOWNLOADING_MARKDOWN_FROM_GCS", extra={
            "event": "downloading_markdown_gcs",
            "blob_path": f"{prefix}context.md",
            "blob_size_bytes": blob.size,
        })
        local_path = f"/tmp/context.md"
        blob.download_to_filename(local_path)
        loader = TextLoader(local_path, encoding="utf-8")
        md_docs = loader.load()
        docs.extend(md_docs)
        logger.info("MARKDOWN_DOWNLOADED_FROM_GCS", extra={
            "event": "markdown_downloaded_gcs",
            "total_chars": sum(len(doc.page_content) for doc in md_docs),
        })
    else:
        logger.warning("MARKDOWN_NOT_FOUND_IN_GCS", extra={"blob_path": f"{prefix}context.md"})
    
    # Download XLSX
    blob = bucket.blob(f"{prefix}work_stories.xlsx")
    if blob.exists():
        logger.info("DOWNLOADING_XLSX_FROM_GCS", extra={
            "event": "downloading_xlsx_gcs",
            "blob_path": f"{prefix}work_stories.xlsx",
            "blob_size_bytes": blob.size,
        })
        local_path = f"/tmp/work_stories.xlsx"
        blob.download_to_filename(local_path)
        df = pd.read_excel(local_path, sheet_name="Work Stories")
        # Extract text compactly: iterate rows and columns, skip empty cells
        text_lines = []
        for idx, row in df.iterrows():
            row_parts = []
            for col in df.columns:
                value = row[col]
                if pd.notna(value) and str(value).strip():
                    row_parts.append(str(value).strip())
            if row_parts:
                text_lines.append(" ".join(row_parts))
        text = "\n".join(text_lines)
        xlsx_doc = Document(page_content=text, metadata={"source": "work_stories.xlsx"})
        docs.append(xlsx_doc)
        logger.info("XLSX_DOWNLOADED_FROM_GCS", extra={
            "event": "xlsx_downloaded_gcs",
            "rows": len(df),
            "columns": len(df.columns),
            "total_chars": len(text),
        })
    else:
        logger.warning("XLSX_NOT_FOUND_IN_GCS", extra={"blob_path": f"{prefix}work_stories.xlsx"})
    
    logger.info("GCS_DOCUMENTS_LOADED", extra={
        "event": "gcs_documents_loaded",
        "total_documents": len(docs),
        "total_chars": sum(len(doc.page_content) for doc in docs),
    })
    
    return docs


def load_documents() -> list[Document]:
    """Load documents from local (dev) or GCS (prod).
    
    Returns:
        List of Document objects from LangChain
    """
    import time
    load_start = time.time()
    
    gcs_bucket = os.getenv("GCS_BUCKET_NAME")
    
    logger.info("LOAD_DOCUMENTS_START", extra={
        "event": "load_documents_start",
        "source": "gcs" if gcs_bucket else "local",
        "gcs_bucket": gcs_bucket if gcs_bucket else "None",
    })
    
    if gcs_bucket:
        docs = _load_from_gcs(gcs_bucket)
    else:
        docs = _load_from_local()
    
    load_time = time.time() - load_start
    
    logger.info("LOAD_DOCUMENTS_COMPLETED", extra={
        "event": "load_documents_completed",
        "source": "gcs" if gcs_bucket else "local",
        "total_documents": len(docs),
        "total_chars": sum(len(doc.page_content) for doc in docs),
        "load_time_seconds": round(load_time, 3),
    })
    
    return docs

