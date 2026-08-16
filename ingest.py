from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

import config


def load_pdfs(data_dir: Path):
    """
    Load all PDF files from the data directory and normalize
    citation metadata before chunking.
    """

    pages = []

    pdf_files = sorted(data_dir.glob("*.pdf"))

    for pdf_path in pdf_files:
        loader = PyPDFLoader(str(pdf_path))
        pdf_pages = loader.load()

        document_name = pdf_path.name

        for page in pdf_pages:
            # PyPDFLoader page numbers are zero-indexed.
            page_number = page.metadata.get("page", 0) + 1

            page.metadata["document_name"] = document_name
            page.metadata["page_number"] = page_number

        pages.extend(pdf_pages)

    return pages


def chunk_documents(pages):
    """
    Split documents into section-aware chunks while preserving
    citation metadata.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE * 4,
        chunk_overlap=config.CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(pages)

    # Give every chunk a stable ID.
    for i, chunk in enumerate(chunks):
        document_name = chunk.metadata.get("document_name", "unknown")
        page_number = chunk.metadata.get("page_number", 0)

        chunk.metadata["chunk_id"] = (
            f"{document_name}::page-{page_number}::chunk-{i}"
        )

    return chunks


def get_embedding_function():
    """
    Return the local FastEmbed embedding model.

    A small batch size is used to reduce RAM usage during
    embedding generation on machines with limited memory.
    """

    return FastEmbedEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        batch_size=4,
    )


def build_index(chunks):
    """
    Build and persist a local Chroma vector index.
    """

    embedding_function = get_embedding_function()

    persist_directory = str(
        Path(__file__).resolve().parent / "chroma_db"
    )

    vectordb = Chroma(
        collection_name="clinical_guidelines",
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )

    # Avoid accidentally inserting duplicate chunks
    # if the notebook cell is executed more than once.
    ids = [
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ]

    vectordb.add_documents(
        documents=chunks,
        ids=ids,
    )

    return vectordb