"""Unit tests for db.py ChromaDB utilities."""

import tempfile
from pathlib import Path
from typing import Any, Dict, Generator

import pytest

from sotaforge.utils.db import ChromaStore, NullEmbeddingFunction
from sotaforge.utils.models import NotParsedDocument, ParsedDocument


class TestNullEmbeddingFunction:
    """Tests for NullEmbeddingFunction."""

    def test_null_embedding_function_name(self) -> None:
        """Test the name method returns expected value."""
        embedder = NullEmbeddingFunction()
        assert embedder.name() == "null-embedding"

    def test_null_embedding_function_config(self) -> None:
        """Test get_config returns empty dict."""
        embedder = NullEmbeddingFunction()
        config = embedder.get_config()
        assert isinstance(config, dict)
        assert len(config) == 0

    def test_null_embedding_function_call(self) -> None:
        """Test calling the embedding function returns dummy vectors."""
        embedder = NullEmbeddingFunction()
        texts = ["text1", "text2", "text3"]

        embeddings = embedder(texts)

        assert len(embeddings) == 3
        assert all(len(emb) == 64 for emb in embeddings)
        assert all(all(val == 0.0 for val in emb) for emb in embeddings)


class TestChromaStore:
    """Tests for ChromaStore class."""

    @pytest.fixture
    def temp_chroma_path(self) -> Generator[Path, None, None]:
        """Create a temporary directory for ChromaDB."""
        tmpdir = tempfile.mkdtemp()
        yield Path(tmpdir)
        # Cleanup is handled by chroma_store fixture

    @pytest.fixture
    def chroma_store(
        self, temp_chroma_path: Path
    ) -> Generator[ChromaStore, None, None]:
        """Create a ChromaStore instance with temp path."""
        store = ChromaStore(path=temp_chroma_path)
        yield store
        # Close client connections before cleanup
        try:
            if hasattr(store.client, "_producer"):
                store.client._producer.stop()
            if hasattr(store.client, "clear_system_cache"):
                store.client.clear_system_cache()
        except Exception:
            pass
        # Give Windows time to release file handles
        import time

        time.sleep(0.1)
        # Cleanup temp directory
        import shutil

        try:
            shutil.rmtree(temp_chroma_path, ignore_errors=True)
        except Exception:
            pass

    def test_chroma_store_initialization(self, temp_chroma_path: Path) -> None:
        """Test ChromaStore initialization creates path and client."""
        store = ChromaStore(path=temp_chroma_path)

        assert store.path == temp_chroma_path
        assert temp_chroma_path.exists()
        assert store.client is not None
        assert isinstance(store.embedder, NullEmbeddingFunction)

    def test_chroma_store_get_collection(self, chroma_store: ChromaStore) -> None:
        """Test getting or creating a collection."""
        collection = chroma_store.get_collection("test_collection")

        assert collection is not None
        assert collection.name == "test_collection"

    def test_chroma_store_get_same_collection_twice(
        self, chroma_store: ChromaStore
    ) -> None:
        """Test getting the same collection twice returns same collection."""
        col1 = chroma_store.get_collection("test")
        col2 = chroma_store.get_collection("test")

        assert col1.name == col2.name

    def test_upsert_parsed_documents(
        self,
        chroma_store: ChromaStore,
        sample_parsed_document: Dict[str, Any],
    ) -> None:
        """Test upserting ParsedDocument objects."""
        doc = ParsedDocument.from_dict(sample_parsed_document)
        docs = [doc]

        ids = chroma_store.upsert_documents("test_collection", docs)

        assert len(ids) == 1
        assert all(isinstance(doc_id, str) for doc_id in ids)

        # Verify document was stored
        collection = chroma_store.get_collection("test_collection")
        result = collection.get(ids=ids)
        assert len(result["ids"]) == 1

    def test_upsert_not_parsed_documents(
        self,
        chroma_store: ChromaStore,
        sample_not_parsed_document: Dict[str, Any],
    ) -> None:
        """Test upserting NotParsedDocument objects."""
        doc = NotParsedDocument.from_dict(sample_not_parsed_document)
        docs = [doc]

        ids = chroma_store.upsert_documents("test_collection", docs)

        assert len(ids) == 1

        # Verify document was stored
        collection = chroma_store.get_collection("test_collection")
        result = collection.get(ids=ids)
        assert len(result["ids"]) == 1

    def test_upsert_empty_documents_list(self, chroma_store: ChromaStore) -> None:
        """Test upserting empty list returns empty list."""
        ids = chroma_store.upsert_documents("test_collection", [])
        assert ids == []

    def test_upsert_multiple_documents(
        self,
        chroma_store: ChromaStore,
        sample_parsed_document: Dict[str, Any],
    ) -> None:
        """Test upserting multiple documents at once."""
        doc1 = ParsedDocument.from_dict(sample_parsed_document)
        doc2 = ParsedDocument(
            title="Second Document",
            url="https://example.com/2",
            text="Second document text",
        )

        ids = chroma_store.upsert_documents("test_collection", [doc1, doc2])

        assert len(ids) == 2
        assert len(set(ids)) == 2  # All IDs are unique

    def test_fetch_documents(
        self,
        chroma_store: ChromaStore,
        sample_parsed_document: Dict[str, Any],
    ) -> None:
        """Test fetching all documents from a collection."""
        doc1 = ParsedDocument.from_dict(sample_parsed_document)
        doc2 = ParsedDocument(title="Second", url="https://test.com", text="Text")

        chroma_store.upsert_documents("test_collection", [doc1, doc2])
        docs = chroma_store.fetch_documents("test_collection")

        assert len(docs) == 2

    def test_fetch_documents_empty_collection(self, chroma_store: ChromaStore) -> None:
        """Test fetching documents from empty collection."""
        docs = chroma_store.fetch_documents("empty_collection")
        assert len(docs) == 0

    def test_fetch_documents_with_limit(
        self,
        chroma_store: ChromaStore,
        sample_parsed_document: Dict[str, Any],
    ) -> None:
        """Test fetching documents with a limit."""
        doc1 = ParsedDocument.from_dict(sample_parsed_document)
        doc2 = ParsedDocument(title="Second", url="https://test.com", text="Text")
        doc3 = ParsedDocument(title="Third", url="https://test3.com", text="More")

        chroma_store.upsert_documents("test_collection", [doc1, doc2, doc3])
        docs = chroma_store.fetch_documents("test_collection", limit=2)

        assert len(docs) == 2

    def test_fetch_documents_returns_parsed_documents(
        self,
        chroma_store: ChromaStore,
        sample_parsed_document: Dict[str, Any],
    ) -> None:
        """Test that fetch_documents returns ParsedDocument instances."""
        doc = ParsedDocument.from_dict(sample_parsed_document)
        chroma_store.upsert_documents("test_collection", [doc])

        docs = chroma_store.fetch_documents("test_collection")

        assert len(docs) > 0
        assert all(isinstance(d, ParsedDocument) for d in docs)

    def test_metadata_serialization(self, chroma_store: ChromaStore) -> None:
        """Test that document metadata is properly serialized."""
        doc = ParsedDocument(
            title="Test",
            url="https://test.com",
            text="Content",
            authors=["Author 1", "Author 2"],
            year=2024,
            metadata={"custom_field": "value"},
        )

        ids = chroma_store.upsert_documents("test_collection", [doc])

        # Retrieve and verify metadata
        collection = chroma_store.get_collection("test_collection")
        result = collection.get(ids=ids)

        metadatas = result["metadatas"]
        assert metadatas is not None
        assert len(metadatas) == 1
        metadata = metadatas[0]
        assert "title" in metadata
        assert metadata["title"] == "Test"
