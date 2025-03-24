from langchain_core.embeddings import Embeddings
from langchain_google_vertexai.embeddings import VertexAIEmbeddings


def get_embeddings_model() -> Embeddings:
    return VertexAIEmbeddings(model_name=settings.EMBEDDINGS_MODEL_NAME)
