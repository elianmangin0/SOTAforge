"""Parsing server for extracting text and metadata from web pages and papers."""

import asyncio
import base64
import tempfile
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import requests
import trafilatura
from openai import AsyncOpenAI

from sotaforge.utils.constants import (
    MAX_CONCURRENT_PDF_PAGES,
    MODEL,
    PDF_PARSING_MAX_TOKENS,
    REQUEST_TIMEOUT_PDF,
    REQUEST_TIMEOUT_WEB,
)
from sotaforge.utils.logger import get_logger
from sotaforge.utils.models import NotParsedDocument, ParsedDocument
from sotaforge.utils.prompts import PDF_PARSING_PROMPT

logger = get_logger(__name__)
llm = AsyncOpenAI()

# Semaphore to limit concurrent PDF page parsing
_pdf_page_semaphore = asyncio.Semaphore(MAX_CONCURRENT_PDF_PAGES)


async def parse_single_page_with_vlm(page_image_b64: str, **kwargs: Any) -> str:
    """Parse a single PDF page using GPT-5 nano vision model.

    Args:
        page_image_b64: Base64-encoded PNG image of the page
        kwargs: Additional context (e.g. pdf_path, page_num)

    Returns:
        Extracted text from the page

    """
    logger.debug(
        f"Calling VLM for page {kwargs.get('page_num')} "
        f"from PDF {kwargs.get('pdf_path')}"
    )

    content: list[dict[str, Any]] = [
        {"type": "text", "text": PDF_PARSING_PROMPT},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{page_image_b64}",
                "detail": "high",
            },
        },
    ]

    response = await llm.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": content}],  # type: ignore[misc,list-item]
        max_completion_tokens=PDF_PARSING_MAX_TOKENS,
    )

    extracted_text = response.choices[0].message.content or ""
    logger.debug(
        f"VLM extracted {len(extracted_text)} characters"
        f" from page {kwargs.get('page_num')}"
        f" of PDF {kwargs.get('pdf_path')}"
    )

    return extracted_text.strip()


async def parse_pdf_with_vlm(pdf_path: Path) -> str:
    """Parse a PDF using GPT-5 nano vision model with parallel page processing.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Extracted text from the PDF

    """
    import asyncio

    logger.debug(f"Converting PDF to images: {pdf_path}")

    # Open PDF with PyMuPDF
    pdf_document = fitz.open(pdf_path)

    # Limit to first 10 pages for token efficiency
    page_count = min(len(pdf_document), 50)
    logger.debug(f"Processing {page_count} pages from PDF")

    # Convert pages to base64-encoded images
    page_images = []

    for page_num in range(page_count):
        page = pdf_document[page_num]

        # Render page to image (pixmap)
        pix = page.get_pixmap(dpi=150)

        # Convert to PNG bytes
        img_bytes = pix.pil_tobytes(format="PNG")

        # Encode to base64
        base64_image = base64.b64encode(img_bytes).decode("utf-8")
        page_images.append(base64_image)

        logger.debug(f"Processed page {page_num + 1}/{page_count}")

    pdf_document.close()

    # Parse pages in controlled batches using semaphore

    async def parse_with_semaphore(page_img: str, page_num: int) -> str:
        async with _pdf_page_semaphore:
            return await parse_single_page_with_vlm(
                page_img, page_num=page_num, pdf_path=pdf_path
            )

    tasks = [
        parse_with_semaphore(page_img, page_num + 1)
        for page_num, page_img in enumerate(page_images)
    ]

    page_texts = await asyncio.gather(*tasks)

    # Combine all page texts
    extracted_text = "\n\n".join(page_texts)
    logger.debug(
        f"VLM extracted total {len(extracted_text)} characters from {page_count} pages"
    )

    return extracted_text.strip()


async def parse_web_result(result: NotParsedDocument) -> ParsedDocument:
    """Parse a web result and extract full text content.

    Args:
        result: NotParsedDocument with title, url, snippet

    Returns:
        ParsedDocument with parsed text and metadata

    """
    logger.debug(f"Parsing web result: {result.url}")

    text = result.snippet

    # Try to fetch and extract main content using trafilatura
    try:
        response = requests.get(
            result.url,
            timeout=REQUEST_TIMEOUT_WEB,
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
            logger.debug(
                f"Trafilatura extracted {len(text)} characters from {result.url}"
            )
        else:
            logger.debug(
                f"Trafilatura found no extractable text at {result.url}, using snippet"
            )
    except Exception as e:
        logger.warning(f"Failed to fetch/parse content from {result.url}: {e}")
        logger.debug("Using snippet as fallback")

    # Create ParsedDocument from NotParsedDocument with parsed text
    return ParsedDocument.from_not_parsed(
        result,
        text=text,
        summary=result.snippet,
    )


async def parse_paper_result(result: NotParsedDocument) -> ParsedDocument:
    """Parse a paper result and extract text from PDF.

    Args:
        result: NotParsedDocument with paper metadata

    Returns:
        ParsedDocument with parsed text and metadata

    """
    logger.debug(f"Parsing paper result: {result.title}")

    # Start with abstract as the text content
    text = result.abstract

    # Try to fetch full paper content from URL
    try:
        # Check if it's an arXiv paper
        if "arxiv.org" in result.url:
            # Extract arXiv ID and construct PDF URL
            arxiv_id = result.url.split("/")[-1]
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            logger.debug(f"Fetching arXiv paper: {pdf_url}")

            # Download PDF to temporary file
            response = requests.get(
                pdf_url,
                timeout=REQUEST_TIMEOUT_PDF,
                headers={"User-Agent": "Mozilla/5.0 (SOTAforge)"},
            )
            response.raise_for_status()

            # Save to temporary file and parse with VLM
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(response.content)
                tmp_path = Path(tmp.name)

            try:
                extracted_text = await parse_pdf_with_vlm(tmp_path)

                if extracted_text and len(extracted_text.strip()) > len(text):
                    text = extracted_text.strip()
                    logger.debug(f"VLM extracted {len(text)} characters from PDF")
                else:
                    logger.warning("VLM extraction yielded less content than abstract")
            finally:
                # Clean up temporary file
                tmp_path.unlink(missing_ok=True)
        else:
            # Try to fetch from the URL
            response = requests.get(
                result.url,
                timeout=REQUEST_TIMEOUT_WEB,
                headers={"User-Agent": "Mozilla/5.0 (SOTAforge)"},
            )
            response.raise_for_status()

            # Check content type
            content_type = response.headers.get("content-type", "").lower()
            if "pdf" in content_type:
                # Parse PDF with VLM
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(response.content)
                    tmp_path = Path(tmp.name)

                try:
                    extracted_text = await parse_pdf_with_vlm(tmp_path)

                    if extracted_text and len(extracted_text.strip()) > len(text):
                        text = extracted_text.strip()
                        logger.debug(f"VLM extracted {len(text)} characters from PDF")
                finally:
                    tmp_path.unlink(missing_ok=True)
            else:
                logger.warning(
                    f"Non-PDF content type: {content_type} - using abstract only"
                )

    except Exception as e:
        logger.warning(f"Failed to fetch/parse full paper from {result.url}: {e}")
        logger.debug("Using abstract as fallback")

    # Create ParsedDocument from NotParsedDocument with parsed text
    return ParsedDocument.from_not_parsed(
        result,
        text=text,
        summary=result.abstract,
    )
