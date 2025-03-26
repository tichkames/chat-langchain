from langchain_core.embeddings import Embeddings
from langchain_google_vertexai.embeddings import VertexAIEmbeddings

from backend.constants import EMBEDDINGS_MODEL_NAME

def get_embeddings_model() -> Embeddings:
    return VertexAIEmbeddings(model_name=EMBEDDINGS_MODEL_NAME)
