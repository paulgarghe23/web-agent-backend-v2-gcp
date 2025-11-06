import os

import google.auth
from langchain_google_vertexai import VertexAIEmbeddings

# Get default project and location
_, project_id = google.auth.default()
LOCATION = os.getenv("LOCATION", "europe-southwest1")


def get_embeddings() -> VertexAIEmbeddings:
    """Initialize Vertex AI Embeddings for vector operations.
    
    Returns:
        VertexAIEmbeddings instance configured with project and location
    """
    return VertexAIEmbeddings(
        model_name="textembedding-gecko@003",
        project=project_id,
        location=LOCATION,
    )

