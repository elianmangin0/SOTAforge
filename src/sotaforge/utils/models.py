"""Data classes for SOTAforge."""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Self, Union


class SourceType(str, Enum):
    """Source type enumeration for documents."""

    WEB = "web"
    PAPER = "paper"
    UNKNOWN = "unknown"


@dataclass
class Document:
    """Base document class with common fields across pipeline stages."""

    title: str
    url: str = ""
    source_type: SourceType = SourceType.UNKNOWN

    # Search stage fields (available before parsing)
    snippet: str = ""  # for web results
    abstract: str = ""  # for papers
    authors: List[str] = field(default_factory=list)  # for papers
    year: int = 0  # for papers
    venue: str = ""  # for papers

    # Generic metadata that can hold any enrichment
    metadata: Dict[str, Union[str, int, List[str]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def _parse_source_type(cls, source_type_str: str) -> SourceType:
        """Parse source type string to enum, defaulting to UNKNOWN."""
        return (
            SourceType(source_type_str)
            if source_type_str in {st.value for st in SourceType}
            else SourceType.UNKNOWN
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Create from a mapping, filling missing fields with defaults."""
        source_type_str = data.get("source_type", "unknown")
        source_type = cls._parse_source_type(source_type_str)
        return cls(
            title=data.get("title", ""),
            url=data.get("url", ""),
            source_type=source_type,
            snippet=data.get("snippet", ""),
            abstract=data.get("abstract", ""),
            authors=list(data.get("authors", []) or []),
            year=int(data.get("year", 0) or 0),
            venue=data.get("venue", ""),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class NotParsedDocument(Document):
    """Document before parsing - contains metadata but no extracted text."""

    pass


@dataclass
class ParsedDocument(Document):
    """Fully parsed document that evolved through the pipeline."""

    # Parse stage fields
    text: str = ""  # full extracted content

    # Analysis stage fields
    themes: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)

    def to_dict_with_text_limit(self, char_limit: int) -> Dict[str, Any]:
        """Convert to dict, limiting text field to char_limit characters."""
        doc_dict = self.to_dict()
        if len(self.text) > char_limit:
            doc_dict["text"] = self.text[:char_limit] + "...[truncated]"
        return doc_dict

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ParsedDocument":
        """Create a ParsedDocument from a mapping, filling missing with defaults."""
        source_type_str = data.get("source_type", "unknown")
        source_type = cls._parse_source_type(source_type_str)
        return cls(
            title=data.get("title", ""),
            url=data.get("url", ""),
            source_type=source_type,
            snippet=data.get("snippet", ""),
            abstract=data.get("abstract", ""),
            authors=list(data.get("authors", []) or []),
            year=int(data.get("year", 0) or 0),
            venue=data.get("venue", ""),
            text=data.get("text", ""),
            themes=list(data.get("themes", []) or []),
            insights=list(data.get("insights", []) or []),
            metadata=dict(data.get("metadata", {}) or {}),
        )

    @classmethod
    def from_not_parsed(
        cls, not_parsed: NotParsedDocument, **updates: Any
    ) -> "ParsedDocument":
        """Create a ParsedDocument from a NotParsedDocument, adding fields."""
        return cls(
            title=not_parsed.title,
            url=not_parsed.url,
            source_type=not_parsed.source_type,
            snippet=not_parsed.snippet,
            abstract=not_parsed.abstract,
            authors=not_parsed.authors,
            year=not_parsed.year,
            venue=not_parsed.venue,
            text=updates.get("text", ""),
            themes=updates.get("themes", []),
            insights=updates.get("insights", []),
            metadata={**not_parsed.metadata, **updates.get("metadata", {})},
        )


@dataclass
class ThemesAndInsights:
    """Extracted themes and insights (for Pydantic AI output format)."""

    themes: List[str]
    insights: List[str]


@dataclass
class DocumentScore:
    """Scores for a document on multiple criteria."""

    title: str
    scores: Dict[str, int]  # criteria -> score (1-5)
    mean_score: float
    keep: bool  # True if mean > 2

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return asdict(self)
