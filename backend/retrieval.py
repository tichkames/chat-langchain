from contextlib import contextmanager
from typing import Iterator

from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from backend.configuration import BaseConfiguration
from backend.constants import QDRANT_COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY
from backend.embeddings import get_embeddings_model


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)

        case "vertexai":
            from langchain_google_vertexai.embeddings import VertexAIEmbeddings

            return VertexAIEmbeddings(model=model)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


@contextmanager
def make_qdrant_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Iterator[BaseRetriever]:
    with QdrantClient(url=QDRANT_URL) as vs_client:
        store = QdrantVectorStore.from_existing_collection(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            collection_name=QDRANT_COLLECTION_NAME,
            # prefer_grpc=True,
            embedding=get_embeddings_model(),
        )
        search_kwargs = {**configuration.search_kwargs, "return_uuids": True}
        yield store.as_retriever(search_kwargs=search_kwargs)


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Iterator[BaseRetriever]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    match configuration.retriever_provider:
        case "qdrant":
            with make_qdrant_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(BaseConfiguration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )
