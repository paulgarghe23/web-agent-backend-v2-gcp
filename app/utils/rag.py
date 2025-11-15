import re
from app.utils.vector_store import get_vector_store


def _clean_markdown_headers(text: str) -> str:
    """Remove markdown headers (##, ###, etc.) from text."""
    # Remove markdown headers (## Header, ### Subheader, etc.)
    text = re.sub(r'^#{1,6}\s+.+$', '', text, flags=re.MULTILINE)
    # Remove multiple consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def search(query: str, k: int = 3, max_distance: float = 0.7) -> str:
    """Search documents and return relevant context.
    
    Args:
        query: Search query
        k: Number of results to return
        max_distance: Maximum distance threshold (lower = more similar)
        
    Returns:
        Combined context from top-k results above threshold
    """
    store = get_vector_store()
    results_with_scores = store.similarity_search_with_score(query, k=k)
    
    # Filter by distance threshold (FAISS returns distance, lower = more similar)
    filtered = [
        doc for doc, score in results_with_scores 
        if score <= max_distance
    ]
    
    # If no results pass threshold, return at least the best one
    if not filtered and results_with_scores:
        filtered = [results_with_scores[0][0]]  # Best result (lowest distance)
    
    if not filtered:
        return ""
    
    # Clean markdown headers from chunks before returning
    cleaned_chunks = [_clean_markdown_headers(doc.page_content) for doc in filtered]
    
    # Return clean context (formatting happens in answer_question)
    context = "\n\n".join(cleaned_chunks)
    return context

