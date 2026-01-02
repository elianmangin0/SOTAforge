"""Data classes for SOTAforge."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Union


@dataclass
class NotParsedDocument:
    """Document before parsing - contains metadata but no extracted text."""

    title: str
    url: str = ""
    source_type: str = "unknown"  # "web" or "paper"

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
    def from_dict(cls, data: Mapping[str, Any]) -> "NotParsedDocument":
        """Create from a mapping, filling missing fields with defaults."""
        return cls(
            title=data.get("title", ""),
            url=data.get("url", ""),
            source_type=data.get("source_type", "unknown"),
            snippet=data.get("snippet", ""),
            abstract=data.get("abstract", ""),
            authors=list(data.get("authors", []) or []),
            year=int(data.get("year", 0) or 0),
            venue=data.get("venue", ""),
            metadata=dict(data.get("metadata", {}) or {}),
        )


@dataclass
class Document:
    """Unified document that evolves through the pipeline (fully parsed)."""

    title: str
    url: str = ""
    source_type: str = "unknown"  # "web" or "paper"

    # Search stage fields
    snippet: str = ""  # for web results
    abstract: str = ""  # for papers
    authors: List[str] = field(default_factory=list)  # for papers
    year: int = 0  # for papers
    venue: str = ""  # for papers

    # Parse stage fields
    text: str = ""  # full extracted content

    # Analysis stage fields
    themes: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)

    # Generic metadata that can hold any enrichment
    metadata: Dict[str, Union[str, int, List[str]]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert this Document into a JSON-serializable dict."""
        return asdict(self)

    def to_dict_with_text_limit(self, char_limit: int) -> Dict[str, Any]:
        """Convert to dict, limiting text field to char_limit characters."""
        doc_dict = self.to_dict()
        if len(self.text) > char_limit:
            doc_dict["text"] = self.text[:char_limit] + "...[truncated]"
        return doc_dict

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Document":
        """Create a Document from a mapping, filling missing fields with defaults."""
        return cls(
            title=data.get("title", ""),
            url=data.get("url", ""),
            source_type=data.get("source_type", "unknown"),
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
    ) -> "Document":
        """Create a Document from a NotParsedDocument, adding parse-stage fields."""
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
