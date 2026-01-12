"""Unit tests for models.py data classes."""

from typing import Any, Dict

from sotaforge.utils.models import (
    Document,
    DocumentScore,
    NotParsedDocument,
    ParsedDocument,
    SourceType,
    ThemesAndInsights,
)


class TestSourceType:
    """Tests for SourceType enum."""

    def test_source_type_values(self) -> None:
        """Test SourceType enum values."""
        assert SourceType.WEB.value == "web"
        assert SourceType.PAPER.value == "paper"
        assert SourceType.UNKNOWN.value == "unknown"


class TestDocument:
    """Tests for Document base class."""

    def test_document_creation_minimal(self) -> None:
        """Test creating a document with minimal fields."""
        doc = Document(title="Test Document")

        assert doc.title == "Test Document"
        assert doc.url == ""
        assert doc.source_type == SourceType.UNKNOWN
        assert doc.snippet == ""
        assert doc.abstract == ""
        assert doc.authors == []
        assert doc.year == 0
        assert doc.venue == ""
        assert doc.metadata == {}

    def test_document_creation_full(self) -> None:
        """Test creating a document with all fields."""
        doc = Document(
            title="Research Paper",
            url="https://example.com/paper",
            source_type=SourceType.PAPER,
            snippet="A snippet",
            abstract="An abstract",
            authors=["Author 1", "Author 2"],
            year=2024,
            venue="Conference",
            metadata={"key": "value"},
        )

        assert doc.title == "Research Paper"
        assert doc.url == "https://example.com/paper"
        assert doc.source_type == SourceType.PAPER
        assert doc.authors == ["Author 1", "Author 2"]
        assert doc.year == 2024
        assert doc.metadata == {"key": "value"}

    def test_document_to_dict(self) -> None:
        """Test converting document to dictionary."""
        doc = Document(
            title="Test",
            url="https://test.com",
            source_type=SourceType.WEB,
            authors=["Author"],
        )

        doc_dict = doc.to_dict()

        assert doc_dict["title"] == "Test"
        assert doc_dict["url"] == "https://test.com"
        assert doc_dict["source_type"] == "web"
        assert doc_dict["authors"] == ["Author"]

    def test_document_from_dict(self) -> None:
        """Test creating document from dictionary."""
        data = {
            "title": "Test Document",
            "url": "https://test.com",
            "source_type": "paper",
            "authors": ["Author 1"],
            "year": 2023,
        }

        doc = Document.from_dict(data)

        assert doc.title == "Test Document"
        assert doc.url == "https://test.com"
        assert doc.source_type == SourceType.PAPER
        assert doc.authors == ["Author 1"]
        assert doc.year == 2023

    def test_document_from_dict_missing_fields(self) -> None:
        """Test creating document from dict with missing fields uses defaults."""
        data = {"title": "Minimal"}

        doc = Document.from_dict(data)

        assert doc.title == "Minimal"
        assert doc.url == ""
        assert doc.source_type == SourceType.UNKNOWN
        assert doc.authors == []
        assert doc.year == 0

    def test_document_parse_source_type_invalid(self) -> None:
        """Test parsing invalid source type defaults to UNKNOWN."""
        data = {"title": "Test", "source_type": "invalid"}

        doc = Document.from_dict(data)

        assert doc.source_type == SourceType.UNKNOWN


class TestNotParsedDocument:
    """Tests for NotParsedDocument class."""

    def test_not_parsed_document_creation(self) -> None:
        """Test creating a NotParsedDocument."""
        doc = NotParsedDocument(
            title="Unparsed",
            url="https://example.com",
            source_type=SourceType.WEB,
            snippet="A snippet of text",
        )

        assert doc.title == "Unparsed"
        assert doc.snippet == "A snippet of text"
        assert isinstance(doc, Document)


class TestParsedDocument:
    """Tests for ParsedDocument class."""

    def test_parsed_document_creation(self) -> None:
        """Test creating a ParsedDocument with all fields."""
        doc = ParsedDocument(
            title="Parsed Paper",
            url="https://example.com",
            source_type=SourceType.PAPER,
            text="Full text content",
            themes=["AI", "ML"],
            insights=["Insight 1", "Insight 2"],
        )

        assert doc.title == "Parsed Paper"
        assert doc.text == "Full text content"
        assert doc.themes == ["AI", "ML"]
        assert doc.insights == ["Insight 1", "Insight 2"]

    def test_parsed_document_from_dict(
        self, sample_parsed_document: Dict[str, Any]
    ) -> None:
        """Test creating ParsedDocument from dictionary."""
        doc = ParsedDocument.from_dict(sample_parsed_document)

        assert doc.title == "Sample Research Paper"
        assert doc.text == "Full text of the research paper goes here."
        assert doc.themes == ["AI", "Machine Learning"]
        assert doc.insights == ["Novel approach to training", "Better performance"]

    def test_parsed_document_to_dict_with_text_limit(self) -> None:
        """Test converting to dict with text length limit."""
        long_text = "A" * 1000
        doc = ParsedDocument(title="Test", text=long_text)

        doc_dict = doc.to_dict_with_text_limit(char_limit=100)

        assert len(doc_dict["text"]) < len(long_text)
        assert doc_dict["text"].endswith("...[truncated]")

    def test_parsed_document_to_dict_with_text_limit_short_text(self) -> None:
        """Test that short text is not truncated."""
        short_text = "Short text"
        doc = ParsedDocument(title="Test", text=short_text)

        doc_dict = doc.to_dict_with_text_limit(char_limit=100)

        assert doc_dict["text"] == short_text
        assert "truncated" not in doc_dict["text"]

    def test_parsed_document_from_not_parsed(
        self, sample_not_parsed_document: Dict[str, Any]
    ) -> None:
        """Test creating ParsedDocument from NotParsedDocument."""
        not_parsed = NotParsedDocument.from_dict(sample_not_parsed_document)

        parsed = ParsedDocument.from_not_parsed(
            not_parsed,
            text="Extracted text",
            themes=["Theme 1"],
            insights=["Insight 1"],
        )

        assert parsed.title == not_parsed.title
        assert parsed.url == not_parsed.url
        assert parsed.source_type == not_parsed.source_type
        assert parsed.abstract == not_parsed.abstract
        assert parsed.text == "Extracted text"
        assert parsed.themes == ["Theme 1"]
        assert parsed.insights == ["Insight 1"]

    def test_parsed_document_from_not_parsed_defaults(self) -> None:
        """Test from_not_parsed with default values."""
        not_parsed = NotParsedDocument(title="Test", url="https://test.com")

        parsed = ParsedDocument.from_not_parsed(not_parsed)

        assert parsed.text == ""
        assert parsed.themes == []
        assert parsed.insights == []


class TestThemesAndInsights:
    """Tests for ThemesAndInsights dataclass."""

    def test_themes_and_insights_creation(self) -> None:
        """Test creating ThemesAndInsights object."""
        tai = ThemesAndInsights(
            themes=["AI", "ML", "Deep Learning"],
            insights=["Finding 1", "Finding 2"],
        )

        assert tai.themes == ["AI", "ML", "Deep Learning"]
        assert tai.insights == ["Finding 1", "Finding 2"]

    def test_themes_and_insights_empty(self) -> None:
        """Test creating empty ThemesAndInsights."""
        tai = ThemesAndInsights(themes=[], insights=[])

        assert tai.themes == []
        assert tai.insights == []


class TestDocumentScore:
    """Tests for DocumentScore dataclass."""

    def test_document_score_creation(self) -> None:
        """Test creating a DocumentScore."""
        score = DocumentScore(
            title="Test Document",
            scores={"relevance": 4, "quality": 5, "novelty": 3},
            mean_score=4.0,
            keep=True,
        )

        assert score.title == "Test Document"
        assert score.scores["relevance"] == 4
        assert score.mean_score == 4.0
        assert score.keep is True

    def test_document_score_to_dict(self) -> None:
        """Test converting DocumentScore to dictionary."""
        score = DocumentScore(
            title="Test",
            scores={"relevance": 3},
            mean_score=3.0,
            keep=True,
        )

        score_dict = score.to_dict()

        assert score_dict["title"] == "Test"
        assert score_dict["scores"] == {"relevance": 3}
        assert score_dict["mean_score"] == 3.0
        assert score_dict["keep"] is True

    def test_document_score_keep_threshold(self) -> None:
        """Test DocumentScore keep flag based on mean score."""
        high_score = DocumentScore(
            title="Good", scores={"q": 4}, mean_score=4.0, keep=True
        )
        low_score = DocumentScore(
            title="Bad", scores={"q": 1}, mean_score=1.0, keep=False
        )

        assert high_score.keep is True
        assert low_score.keep is False
