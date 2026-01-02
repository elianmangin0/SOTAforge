"""ChromaDB-backed storage utilities for SOTAforge (minimal API)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, List
from uuid import uuid4

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import EmbeddingFunction

from sotaforge.utils.constants import CHROMA_PATH
from sotaforge.utils.dataclasses import Document, NotParsedDocument
from sotaforge.utils.logger import get_logger

logger = get_logger(__name__)


class NullEmbeddingFunction(EmbeddingFunction[list[str]]):
    """Null embedding function that satisfies Chroma without real embeddings.

    This is a placeholder to allow Chroma to work without an actual
    embedding model. Use this for development/testing only.
    """

    @staticmethod
    def name() -> str:
        """Return a stable name for Chroma (suppresses deprecation warning)."""
        return "null-embedding"

    def get_config(self) -> dict[str, Any]:
        """Return minimal config for compatibility with Chroma expectations."""
        return {}

    def __call__(self, input: List[str]) -> List[List[float]]:  # type: ignore[override]
        """Return dummy embeddings for texts.

        Args:
            input: List of text strings to embed.

        Returns:
            List of zero vectors (no actual embedding).

        """
        return [[0.0] * 64 for _ in input]


class ChromaStore:
    """Lightweight wrapper around Chroma persistent client."""

    def __init__(self, path: str | Path | None = None) -> None:
        """Initialize ChromaStore with a persistent storage path.

        Args:
            path: Path to store Chroma data. Defaults to SOTAFORGE_CHROMA_PATH
                environment variable or CHROMA_PATH constant.

        """
        self.path = Path(path or os.getenv("SOTAFORGE_CHROMA_PATH") or CHROMA_PATH)
        self.path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.path))
        self.embedder = NullEmbeddingFunction()
        logger.debug(f"Initialized ChromaStore at {self.path}")

    def get_collection(self, name: str) -> Collection:
        """Return a collection, creating it if missing."""
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedder,  # type: ignore[arg-type]
        )

    def upsert_documents(
        self, collection: str, documents: Iterable[Document | NotParsedDocument]
    ) -> List[str]:
        """Upsert pipeline Document or NotParsedDocument objects directly.

        This simplifies callers: pass Document or NotParsedDocument instances
        and let the store derive ids, document text, metadata and payload.
        """
        items = list(documents)
        if not items:
            return []

        col = self.get_collection(collection)
        ids = [str(uuid4()) for _ in items]
        docs = []
        for d in items:
            # For Document, use text; for NotParsedDocument, use
            # summary/snippet/abstract
            if isinstance(d, Document):
                doc_text = d.text
            else:  # NotParsedDocument
                doc_text = ""
            docs.append(doc_text)

        metadatas = [
            {
                k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                for k, v in d.to_dict().items()
                if k != "text"
            }
            for d in items
        ]
        col.upsert(ids=ids, documents=docs, metadatas=metadatas)  # type: ignore[arg-type]
        logger.debug(f"Upserted {len(ids)} documents into '{collection}'")
        return ids

    def fetch_documents(
        self, collection: str, limit: int | None = None
    ) -> List[Document | NotParsedDocument]:
        """Fetch all documents from a collection.

        Args:
            collection: The collection name to fetch from
            limit: Optional limit on number of documents to retrieve

        Returns:
            List of Document or NotParsedDocument instances

        """
        col = self.get_collection(collection)
        results = col.get(limit=limit)

        if not results or not results["ids"]:
            logger.debug(f"No documents found in collection '{collection}'")
            return []

        documents: list[Document | NotParsedDocument] = []
        for i in range(len(results["ids"])):
            logger.debug(f"Reconstructing document {i} from fetched results")
            logger.debug(f"Results : {results}")
            metadatas = results.get("metadatas")
            documents_list = results.get("documents")

            metadata = metadatas[i] if metadatas else {}
            doc_text = documents_list[i] if documents_list else ""

            # Reconstruct document dict with parsed JSON fields
            doc_dict = {}
            for key, value in metadata.items():
                if isinstance(value, str) and value.startswith(("[", "{")):
                    try:
                        doc_dict[key] = json.loads(value)
                    except json.JSONDecodeError:
                        # Be forgiving if stored metadata isn't valid JSON
                        logger.warning(
                            "Failed to decode metadata key '%s'; keeping raw string",
                            key,
                        )
                        doc_dict[key] = value
                else:
                    doc_dict[key] = value

            # Add text field if stored
            if doc_text:
                doc_dict["text"] = doc_text

            # Return Document if has text, otherwise NotParsedDocument
            if doc_dict.get("text") and isinstance(doc_dict.get("text"), str):
                logger.debug(
                    f"Document {doc_dict.get('title', 'unknown')} has text : "
                    f"{doc_dict.get('text')[:100]}..."  # type: ignore[index]
                )
                documents.append(Document.from_dict(doc_dict))
            else:
                documents.append(NotParsedDocument.from_dict(doc_dict))

        logger.debug(f"Fetched {len(documents)} documents from '{collection}'")
        return documents
