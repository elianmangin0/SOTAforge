"""Parsing server for extracting text and metadata from web pages and papers."""

import tempfile
from pathlib import Path

import requests
import trafilatura
from docling.document_converter import DocumentConverter

from sotaforge.utils.dataclasses import Document, NotParsedDocument
from sotaforge.utils.logger import get_logger

logger = get_logger(__name__)


async def parse_web_result(result: NotParsedDocument) -> Document:
    """Parse a web result and extract full text content.

    Args:
        result: NotParsedDocument with title, url, snippet

    Returns:
        Document with parsed text and metadata

    """
    logger.info(f"Parsing web result: {result.url}")

    text = result.snippet

    # Try to fetch and extract main content using trafilatura
    try:
        response = requests.get(
            result.url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (SOTAforge)"},
        )
        response.raise_for_status()

        extracted = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=False,
        )

        if extracted:
            text = extracted.strip()
            logger.info(
                f"Trafilatura extracted {len(text)} characters from {result.url}"
            )
        else:
            logger.info(
                f"Trafilatura found no extractable text at {result.url}, using snippet"
            )
    except Exception as e:
        logger.warning(f"Failed to fetch/parse content from {result.url}: {e}")
        logger.info("Using snippet as fallback")

    # Create Document from NotParsedDocument with parsed text
    return Document.from_not_parsed(
        result,
        text=text,
        summary=result.snippet,
    )


async def parse_paper_result(result: NotParsedDocument) -> Document:
    """Parse a paper result and extract full text content.

    Args:
        result: NotParsedDocument with paper metadata

    Returns:
        Document with parsed text and metadata

    """
    logger.info(f"Parsing paper result: {result.title}")

    # Start with abstract as the text content
    text = result.abstract

    # Try to fetch full paper content from URL
    try:
        # Check if it's an arXiv paper
        if "arxiv.org" in result.url:
            # Extract arXiv ID and construct PDF URL
            arxiv_id = result.url.split("/")[-1]
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            logger.info(f"Fetching arXiv paper: {pdf_url}")

            # Download PDF to temporary file
            response = requests.get(
                pdf_url,
                timeout=30,
                headers={"User-Agent": "Mozilla/5.0 (SOTAforge)"},
            )
            response.raise_for_status()

            # Save to temporary file and parse with Docling
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = Path(tmp.name)

            try:
                logger.info(f"Parsing PDF with Docling: {tmp_path}")
                converter = DocumentConverter()
                result_doc = converter.convert(tmp_path)

                # Extract markdown text from Docling result
                extracted_text = result_doc.document.export_to_markdown()

                if extracted_text and len(extracted_text.strip()) > len(text):
                    text = extracted_text.strip()
                    logger.info(f"Docling extracted {len(text)} characters from PDF")
                else:
                    logger.warning(
                        "Docling extraction yielded less content than abstract"
                    )
            finally:
                # Clean up temporary file
                tmp_path.unlink(missing_ok=True)
        else:
            # Try to fetch from the URL
            response = requests.get(
                result.url,
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0 (SOTAforge)"},
            )
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" in content_type:
                # Parse PDF with Docling
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(response.content)
                    tmp_path = Path(tmp.name)

                try:
                    logger.info(f"Parsing PDF with Docling: {tmp_path}")
                    converter = DocumentConverter()
                    result_doc = converter.convert(tmp_path)
                    extracted_text = result_doc.document.export_to_markdown()

                    if extracted_text and len(extracted_text.strip()) > len(text):
                        text = extracted_text.strip()
                        logger.info(
                            f"Docling extracted {len(text)} characters from PDF"
                        )
                finally:
                    tmp_path.unlink(missing_ok=True)
            else:
                logger.warning(
                    f"Non-PDF content type: {content_type} - using abstract only"
                )

    except Exception as e:
        logger.warning(f"Failed to fetch/parse full paper from {result.url}: {e}")
        logger.info("Using abstract as fallback")

    # Create Document from NotParsedDocument with parsed text
    return Document.from_not_parsed(
        result,
        text=text,
        summary=result.abstract,
    )
