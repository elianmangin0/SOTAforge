"""Custom exceptions for the Watcher project."""


class SOTAError(Exception):
    """Base exception for all SOTA-related errors."""

    pass


class SearchError(SOTAError):
    """Raised when a search operation fails."""

    pass


class ScrapingError(SOTAError):
    """Raised when web scraping fails."""

    pass


class ExtractionError(SOTAError):
    """Raised when text extraction from HTML fails."""

    pass


class FilterError(SOTAError):
    """Raised when filtering candidates fails."""

    pass


class NetworkError(SOTAError):
    """Raised when network-related operations fail."""

    pass


class ValidationError(SOTAError):
    """Raised when data validation fails."""

    pass


class ConfigurationError(SOTAError):
    """Raised when configuration is invalid or missing."""

    pass


class DatabaseError(SOTAError):
    """Raised when database operations fail."""

    pass
