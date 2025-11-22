import re
import logging
import time
from app.utils.vector_store import get_vector_store

logger = logging.getLogger(__name__)


def _clean_markdown_headers(text: str) -> str:
    """Remove markdown headers (##, ###, etc.) from text."""
    # Remove markdown headers (## Header, ### Subheader, etc.)
    text = re.sub(r'^#{1,6}\s+.+$', '', text, flags=re.MULTILINE)
    # Remove multiple consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def search(query: str, k: int = 3, max_distance: float = 1.2) -> str:
    """Search documents and return relevant context.
    
    Args:
        query: Search query
        k: Number of results to return
        max_distance: Maximum distance threshold (lower = more similar)
        
    Returns:
        Combined context from top-k results above threshold
    """
    search_start = time.time()
    
    logger.info("VECTOR_SEARCH_START", extra={
        "event": "vector_search_started",
        "query": query,
        "query_length": len(query),
        "k": k,
        "max_distance": max_distance,
    })
    
    store = get_vector_store()
    vector_search_start = time.time()
    results_with_scores = store.similarity_search_with_score(query, k=k)
    vector_search_time = time.time() - vector_search_start
    
    logger.info("VECTOR_SEARCH_RESULTS", extra={
        "event": "vector_search_results",
        "query": query,
        "total_results": len(results_with_scores),
        "scores": [round(score, 4) for _, score in results_with_scores],
        "vector_search_time_seconds": round(vector_search_time, 3),
    })
    
    # Log each result with details
    for i, (doc, score) in enumerate(results_with_scores):
        source = doc.metadata.get("source", "unknown")
        is_cv = "CV" in str(source).upper() or "cv" in str(source).lower()
        logger.info("VECTOR_SEARCH_RESULT_DETAIL", extra={
            "event": "vector_search_result_detail",
            "query": query,
            "result_index": i + 1,
            "score": round(score, 4),
            "score_passed_threshold": score <= max_distance,
            "document_length": len(doc.page_content),
            "document_source": source,
            "is_from_cv": is_cv,
            "document_content_full": doc.page_content,  # FULL CONTENT to see if CV is included
            "document_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            "metadata": doc.metadata,
        })
    
    # Filter by distance threshold (FAISS returns distance, lower = more similar)
    filtered = [
        doc for doc, score in results_with_scores 
        if score <= max_distance
    ]
    
    logger.info("VECTOR_SEARCH_FILTERED", extra={
        "event": "vector_search_filtered",
        "query": query,
        "results_before_filter": len(results_with_scores),
        "results_after_filter": len(filtered),
        "threshold": max_distance,
    })
    
    # If no results pass threshold, return at least the best one
    if not filtered and results_with_scores:
        best_score = results_with_scores[0][1]
        logger.warning("NO_RESULTS_PASS_THRESHOLD", extra={
            "event": "no_results_pass_threshold",
            "query": query,
            "best_score": round(best_score, 4),
            "threshold": max_distance,
            "using_best_result": True,
        })
        filtered = [results_with_scores[0][0]]  # Best result (lowest distance)
    
    if not filtered:
        logger.warning("NO_RESULTS_FOUND", extra={
            "event": "no_results_found",
            "query": query,
        })
        return ""
    
    # Clean markdown headers from chunks before returning
    cleaned_chunks = [_clean_markdown_headers(doc.page_content) for doc in filtered]
    
    # Return clean context (formatting happens in answer_question)
    context = "\n\n".join(cleaned_chunks)
    
    total_search_time = time.time() - search_start
    
    logger.info("VECTOR_SEARCH_COMPLETED", extra={
        "event": "vector_search_completed",
        "query": query,
        "chunks_used": len(cleaned_chunks),
        "context_length": len(context),
        "context_preview": context[:500] + "..." if len(context) > 500 else context,
        "context_full": context,  # Full context for debugging
        "total_search_time_seconds": round(total_search_time, 3),
    })
    
    return context

