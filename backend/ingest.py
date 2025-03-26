"""Load html from files, clean up, split, ingest into Qdrant."""
import logging
import os
import re
from typing import Optional

from bs4 import BeautifulSoup, SoupStrainer
from langchain_community.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.indexes import index
from langchain_community.indexes._document_manager import MongoDocumentManager
from langchain.utils.html import PREFIXES_TO_IGNORE_REGEX, SUFFIXES_TO_IGNORE_REGEX
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from backend.constants import QDRANT_COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY, ATLAS_URI, DBNAME
from backend.embeddings import get_embeddings_model
from backend.parser import langchain_docs_extractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def metadata_extractor(
    meta: dict, soup: BeautifulSoup, title_suffix: Optional[str] = None
) -> dict:
    title_element = soup.find("title")
    description_element = soup.find("meta", attrs={"name": "description"})
    html_element = soup.find("html")
    title = title_element.get_text() if title_element else ""
    if title_suffix is not None:
        title += title_suffix

    return {
        "source": meta["loc"],
        "title": title,
        "description": description_element.get("content", "")
        if description_element
        else "",
        "language": html_element.get("lang", "") if html_element else "",
        **meta,
    }


def load_langchain_docs():
    return SitemapLoader(
        "https://python.langchain.com/sitemap.xml",
        filter_urls=["https://python.langchain.com/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def load_langgraph_docs():
    return SitemapLoader(
        "https://langchain-ai.github.io/langgraph/sitemap.xml",
        parsing_function=simple_extractor,
        default_parser="lxml",
        bs_kwargs={"parse_only": SoupStrainer(name=("article", "title"))},
        meta_function=lambda meta, soup: metadata_extractor(
            meta, soup, title_suffix=" | 🦜🕸️LangGraph"
        ),
    ).load()


def load_fruitsandroots_docs():
    return RecursiveUrlLoader(
        url="https://www.fruitsandroots.co.za/Contact-Fruits-and-Roots/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
    ).load()


def simple_extractor(html: str | BeautifulSoup) -> str:
    if isinstance(html, str):
        soup = BeautifulSoup(html, "lxml")
    elif isinstance(html, BeautifulSoup):
        soup = html
    else:
        raise ValueError(
            "Input should be either BeautifulSoup object or an HTML string"
        )
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_api_docs():
    return RecursiveUrlLoader(
        url="https://api.python.langchain.com/en/latest/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://api.python.langchain.com/en/latest/_sources",
            "https://api.python.langchain.com/en/latest/_modules",
        ),
    ).load()


def ingest_docs():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    embedding = get_embeddings_model()

    vs_client = QdrantClient(url=QDRANT_URL)
    vectorstore = QdrantVectorStore.from_existing_collection(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        collection_name=QDRANT_COLLECTION_NAME,
        # prefer_grpc=True,
        embedding=embedding,
    )

    record_manager = MongoDocumentManager(
        namespace=f"qdrant/{QDRANT_COLLECTION_NAME}",
        mongodb_url=ATLAS_URI,
        db_name=DBNAME,
        collection_name="qdrant_rm",
    )

    record_manager.create_schema()

    # docs_from_documentation = load_langchain_docs()
    # logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")
    # docs_from_api = load_api_docs()
    # logger.info(f"Loaded {len(docs_from_api)} docs from API")
    docs_from_fruitsandroots = load_fruitsandroots_docs()
    logger.info(f"Loaded {len(docs_from_fruitsandroots)} docs from Fruits & Roots")
    # docs_from_langgraph = load_langgraph_docs()
    # logger.info(f"Loaded {len(docs_from_langgraph)} docs from LangGraph")

    docs_transformed = text_splitter.split_documents(
        # docs_from_documentation
        # + docs_from_api
        docs_from_fruitsandroots
        # + docs_from_langgraph
    )
    docs_transformed = [
        doc for doc in docs_transformed if len(doc.page_content) > 10
    ]

    # We try to return 'source' and 'title' metadata when querying vector store and
    # Qdrant will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    indexing_stats = index(
        docs_transformed,
        record_manager,
        vectorstore,
        cleanup="full",
        source_id_key="source", #TODO: change to doc_id
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )

    logger.info(f"Indexing stats: {indexing_stats}")
    num_stats = vs_client.get_collection(collection_name=QDRANT_COLLECTION_NAME)

    logger.info(
        f"VS Stats: {list(num_stats)}",
    )


if __name__ == "__main__":
    ingest_docs()
