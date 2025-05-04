import os
import time
import logging
from typing import Optional
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models

# Import centralized logging configuration
# Assuming logging_config.py is at the project root and logger is configured there
# We might need to adjust this import based on project structure or pass logger
try:
    from logging_config import logger
except ImportError:
    # Fallback if logging_config is not directly importable here
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO) # Basic fallback config

# --- Configuration (Import or pass necessary parts) ---
# For now, let's assume we can import Config or pass necessary values
# A more robust solution might involve a shared config object or dependency injection
try:
    from main import Config # Assuming Config is importable from main
except ImportError:
    # Define a minimal Config if main.Config is not available for standalone testing
    class Config:
        MAX_RETRIES: int = 3
        CHUNK_SIZE: int = 300
        CHUNK_OVERLAP: int = 0
        VECTORSTORE_PATH: str = "./qdrant_vectorstore"
        COLLECTION_NAME: str = "default_collection"
        EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2" # A common HuggingFace embedding model


@retry(stop=stop_after_attempt(Config.MAX_RETRIES), wait=wait_exponential(multiplier=1, min=4, max=10))
def process_pdf_and_create_vectorstore(pdf_url: str, collection_name: str, vectorstore_path: str, embedding_model_name: str) -> Optional[Qdrant]:
    """
    Loads and parses a PDF, creates embeddings, and initializes a persistent Qdrant vectorstore.

    Args:
        pdf_url (str): URL of the PDF to be processed.
        collection_name (str): Name of the collection for vector storage.
        vectorstore_path (str): Path to persist the vectorstore.
        embedding_model_name (str): Name of the embedding model to use.

    Returns:
        Optional[Qdrant): The initialized Qdrant vectorstore if successful, None otherwise.

    # 5. Memory Optimization (PDF Loading):
    # Strategies for optimizing memory when loading large PDFs:
    # - Process the PDF page by page or in chunks using the loader's lazy loading capabilities if available.
    # - Use generators to yield document chunks instead of loading everything into memory at once.
    # - Consider alternative PDF loading libraries that support streaming or more efficient parsing.
    # The current PyMuPDFLoader loads the entire document at once, which can be memory intensive for large files.
    """
    try:
        # Check if vectorstore exists
        if os.path.exists(vectorstore_path):
            logger.info("Loading vectorstore from disk: path=%s, collection=%s", vectorstore_path, collection_name)
            embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
            client = QdrantClient(path=vectorstore_path)
            # Check if collection exists in the loaded client
            try:
                client.get_collection(collection_name=collection_name)
                vectorstore = Qdrant(client=client, collection_name=collection_name, embeddings=embedding_model)
                logger.info("Successfully loaded vectorstore from disk: path=%s, collection=%s", vectorstore_path, collection_name)
                return vectorstore
            except Exception:
                 logger.warning("Collection not found in existing vectorstore path, reprocessing PDF: path=%s, collection=%s", vectorstore_path, collection_name)

        logger.info("Processing PDF and creating new vectorstore: pdf_url=%s", pdf_url)


        start_time_load = time.perf_counter()
        # Current implementation loads the whole PDF. See comments above for optimization ideas.
        docs = PyMuPDFLoader(pdf_url).load()
        end_time_load = time.perf_counter()
        logger.info("Successfully loaded PDF: pdf_url=%s, duration_ms=%s", pdf_url, (end_time_load - start_time_load)*1000)

        start_time_split = time.perf_counter()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
        splits = text_splitter.split_documents(docs)
        end_time_split = time.perf_counter()
        logger.info("Successfully split documents: num_splits=%s, duration_ms=%s", len(splits), (end_time_split - start_time_split)*1000)

        start_time_embed = time.perf_counter()
        embedding_model = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vectorstore = Qdrant.from_documents(
            splits,
            embedding_model,
            path=vectorstore_path, # Use path for persistence
            collection_name=collection_name
        )
        end_time_embed = time.perf_counter()
        logger.info("Successfully created and persisted vector store: path=%s, collection_name=%s, duration_ms=%s", vectorstore_path, collection_name, (end_time_embed - start_time_embed)*1000)

        return vectorstore
    except FileNotFoundError:
        logger.error("File not found: pdf_url=%s", pdf_url)
        return None
    except Exception as e:
        logger.error("Unexpected error occurred while processing PDF or setting up vector store: pdf_url=%s, error=%s", pdf_url, str(e))
        return None