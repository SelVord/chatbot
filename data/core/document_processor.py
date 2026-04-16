"""
core/document_processor.py — Load documents and split them into chunks.

Supported input types:
  - PDF  (.pdf)
  - Word (.docx)
  - Text (.txt)
  - Raw text string (pasted directly by user)
"""

from pathlib import Path
from typing import BinaryIO

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def _splitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )


def load_pdf(file_path: str | Path) -> tuple[list[Document], int]:
    loader = PyPDFLoader(str(file_path))
    pages = loader.load()
    return pages, len(pages)


def load_txt(file_path: str | Path) -> tuple[list[Document], int]:
    loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    return docs, 1


def load_docx(file_path: str | Path) -> tuple[list[Document], int]:
    try:
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(str(file_path))
        docs = loader.load()
        return docs, 1
    except Exception as e:
        raise RuntimeError(f"Cannot load .docx file: {e}. Make sure docx2txt is installed.")


def load_raw_text(text: str, source_name: str = "pasted_text") -> tuple[list[Document], int]:
    doc = Document(
        page_content=text,
        metadata={"source_file": source_name, "page": 0},
    )
    return [doc], 1


def split_documents(docs: list[Document], source_name: str) -> list[Document]:
    chunks = _splitter().split_documents(docs)
    for chunk in chunks:
        chunk.metadata["source_file"] = chunk.metadata.get("source_file", source_name)
        chunk.metadata.setdefault("page", 0)
    return chunks


def load_and_split(file_path: str | Path) -> tuple[list[Document], int]:
    path = Path(file_path)
    ext = path.suffix.lower()
    source_name = path.name

    if ext == ".pdf":
        docs, pages = load_pdf(path)
    elif ext == ".txt":
        docs, pages = load_txt(path)
    elif ext == ".docx":
        docs, pages = load_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    chunks = split_documents(docs, source_name)
    return chunks, pages


def load_and_split_text(text: str, source_name: str = "pasted_text") -> tuple[list[Document], int]:
    docs, pages = load_raw_text(text, source_name)
    chunks = split_documents(docs, source_name)
    return chunks, pages


def save_uploaded_file(uploaded_file: BinaryIO, filename: str) -> Path:
    dest = config.UPLOADS_DIR / filename
    dest.write_bytes(uploaded_file.read())
    return dest
