"""Tests for parsing module."""

import base64
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import requests

from sotaforge.utils.models import NotParsedDocument, ParsedDocument
from sotaforge.utils.parsing import (
    parse_paper_result,
    parse_pdf_with_vlm,
    parse_single_page_with_vlm,
    parse_web_result,
)


class TestParseSinglePageWithVLM:
    """Tests for parse_single_page_with_vlm function."""

    @pytest.mark.asyncio
    async def test_parse_single_page_success(self) -> None:
        """Test successful page parsing with VLM."""
        # Create a simple base64 image
        test_image = base64.b64encode(b"fake_image_data").decode("utf-8")

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Extracted text from page"

        with patch("sotaforge.utils.parsing.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm
            mock_llm.chat.completions.create = AsyncMock(return_value=mock_response)

            result = await parse_single_page_with_vlm(
                test_image, page_num=1, pdf_path="test.pdf"
            )

            assert result == "Extracted text from page"
            mock_llm.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_parse_single_page_empty_response(self) -> None:
        """Test handling of empty VLM response."""
        test_image = base64.b64encode(b"fake_image_data").decode("utf-8")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None

        with patch("sotaforge.utils.parsing.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm
            mock_llm.chat.completions.create = AsyncMock(return_value=mock_response)

            result = await parse_single_page_with_vlm(test_image)

            assert result == ""

    @pytest.mark.asyncio
    async def test_parse_single_page_strips_whitespace(self) -> None:
        """Test that extracted text is stripped of whitespace."""
        test_image = base64.b64encode(b"fake_image_data").decode("utf-8")

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "  Text with spaces  \n\n"

        with patch("sotaforge.utils.parsing.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_get_llm.return_value = mock_llm
            mock_llm.chat.completions.create = AsyncMock(return_value=mock_response)

            result = await parse_single_page_with_vlm(test_image)

            assert result == "Text with spaces"


class TestParsePDFWithVLM:
    """Tests for parse_pdf_with_vlm function."""

    @pytest.fixture
    def mock_pdf_file(self) -> Path:
        """Create a temporary mock PDF file."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            # Write minimal valid PDF
            tmp.write(b"%PDF-1.4\n%%EOF")
            return Path(tmp.name)

    @pytest.mark.asyncio
    async def test_parse_pdf_with_mocked_fitz(self, mock_pdf_file: Path) -> None:
        """Test PDF parsing with mocked fitz library."""
        # Mock fitz.open
        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.pil_tobytes.return_value = b"fake_png_data"
        mock_page.get_pixmap.return_value = mock_pixmap

        mock_document = MagicMock()
        mock_document.__len__.return_value = 2  # 2 pages
        mock_document.__getitem__.return_value = mock_page
        mock_document.close = MagicMock()

        with (
            patch("sotaforge.utils.parsing.fitz.open", return_value=mock_document),
            patch(
                "sotaforge.utils.parsing.parse_single_page_with_vlm",
                new=AsyncMock(return_value="Page text"),
            ),
        ):
            result = await parse_pdf_with_vlm(mock_pdf_file)

            assert "Page text" in result
            mock_document.close.assert_called_once()

        # Cleanup
        mock_pdf_file.unlink(missing_ok=True)


class TestParseWebResult:
    """Tests for parse_web_result function."""

    @pytest.mark.asyncio
    async def test_parse_web_result_success(self) -> None:
        """Test successful web result parsing."""
        doc = NotParsedDocument(
            title="Test Article",
            url="https://example.com/article",
            snippet="Short snippet",
        )

        # Mock successful HTTP request
        mock_response = Mock()
        mock_response.text = "<html><body><p>Full article content</p></body></html>"
        mock_response.raise_for_status = Mock()

        with (
            patch("sotaforge.utils.parsing.requests.get", return_value=mock_response),
            patch(
                "sotaforge.utils.parsing.trafilatura.extract",
                return_value="Full article content here",
            ),
        ):
            result = await parse_web_result(doc)

            assert isinstance(result, ParsedDocument)
            assert result.title == "Test Article"
            assert result.url == "https://example.com/article"
            assert result.text == "Full article content here"
            assert result.snippet == "Short snippet"

    @pytest.mark.asyncio
    async def test_parse_web_result_trafilatura_fails(self) -> None:
        """Test web parsing fallback when trafilatura returns None."""
        doc = NotParsedDocument(
            title="Test Article",
            url="https://example.com/article",
            snippet="Short snippet",
        )

        mock_response = Mock()
        mock_response.text = "<html><body>Content</body></html>"
        mock_response.raise_for_status = Mock()

        with (
            patch("sotaforge.utils.parsing.requests.get", return_value=mock_response),
            patch("sotaforge.utils.parsing.trafilatura.extract", return_value=None),
        ):
            result = await parse_web_result(doc)

            assert result.text == "Short snippet"  # Fallback to snippet

    @pytest.mark.asyncio
    async def test_parse_web_result_network_error(self) -> None:
        """Test web parsing handles network errors gracefully."""
        doc = NotParsedDocument(
            title="Test Article",
            url="https://example.com/article",
            snippet="Short snippet",
        )

        with patch(
            "sotaforge.utils.parsing.requests.get",
            side_effect=requests.RequestException("Network error"),
        ):
            result = await parse_web_result(doc)

            # Should fall back to snippet
            assert result.text == "Short snippet"

    @pytest.mark.asyncio
    async def test_parse_web_result_timeout(self) -> None:
        """Test web parsing handles timeout errors."""
        doc = NotParsedDocument(
            title="Test Article",
            url="https://example.com/article",
            snippet="Short snippet",
        )

        with patch(
            "sotaforge.utils.parsing.requests.get",
            side_effect=requests.Timeout("Request timeout"),
        ):
            result = await parse_web_result(doc)

            assert result.text == "Short snippet"


class TestParsePaperResult:
    """Tests for parse_paper_result function."""

    @pytest.mark.asyncio
    async def test_parse_paper_result_abstract_only(self) -> None:
        """Test paper parsing with abstract only (no PDF fetch)."""
        doc = NotParsedDocument(
            title="Research Paper",
            url="https://example.com/paper",
            snippet="Paper snippet",
            abstract="This is the abstract of the paper.",
        )

        with patch(
            "sotaforge.utils.parsing.requests.get",
            side_effect=requests.RequestException("Cannot fetch"),
        ):
            result = await parse_paper_result(doc)

            assert isinstance(result, ParsedDocument)
            assert result.title == "Research Paper"
            assert result.text == "This is the abstract of the paper."
            assert result.snippet == "Paper snippet"

    @pytest.mark.asyncio
    async def test_parse_paper_arxiv_url(self) -> None:
        """Test parsing arXiv paper URL."""
        doc = NotParsedDocument(
            title="arXiv Paper",
            url="https://arxiv.org/abs/2401.12345",
            snippet="Paper snippet",
            abstract="Short abstract",
        )

        # Mock PDF response
        mock_pdf_response = Mock()
        mock_pdf_response.content = b"%PDF-1.4\n%%EOF"
        mock_pdf_response.raise_for_status = Mock()

        # Mock fitz
        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.pil_tobytes.return_value = b"fake_png_data"
        mock_page.get_pixmap.return_value = mock_pixmap

        mock_document = MagicMock()
        mock_document.__len__.return_value = 1
        mock_document.__getitem__.return_value = mock_page
        mock_document.close = MagicMock()

        with (
            patch(
                "sotaforge.utils.parsing.requests.get", return_value=mock_pdf_response
            ),
            patch("sotaforge.utils.parsing.fitz.open", return_value=mock_document),
            patch(
                "sotaforge.utils.parsing.parse_single_page_with_vlm",
                new=AsyncMock(return_value="Full paper text from VLM"),
            ),
        ):
            result = await parse_paper_result(doc)

            assert result.text == "Full paper text from VLM"

    @pytest.mark.asyncio
    async def test_parse_paper_pdf_content_type(self) -> None:
        """Test parsing non-arXiv paper with PDF content type."""
        doc = NotParsedDocument(
            title="Research Paper",
            url="https://example.com/paper.pdf",
            snippet="Paper snippet",
            abstract="Short abstract",
        )

        mock_response = Mock()
        mock_response.content = b"%PDF-1.4\n%%EOF"
        mock_response.headers = {"content-type": "application/pdf"}
        mock_response.raise_for_status = Mock()

        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.pil_tobytes.return_value = b"fake_png_data"
        mock_page.get_pixmap.return_value = mock_pixmap

        mock_document = MagicMock()
        mock_document.__len__.return_value = 1
        mock_document.__getitem__.return_value = mock_page
        mock_document.close = MagicMock()

        with (
            patch("sotaforge.utils.parsing.requests.get", return_value=mock_response),
            patch("sotaforge.utils.parsing.fitz.open", return_value=mock_document),
            patch(
                "sotaforge.utils.parsing.parse_single_page_with_vlm",
                new=AsyncMock(return_value="Extracted PDF text"),
            ),
        ):
            result = await parse_paper_result(doc)

            assert result.text == "Extracted PDF text"

    @pytest.mark.asyncio
    async def test_parse_paper_non_pdf_content(self) -> None:
        """Test parsing paper with non-PDF content type."""
        doc = NotParsedDocument(
            title="Research Paper",
            url="https://example.com/paper.html",
            snippet="Paper snippet",
            abstract="This is the abstract",
        )

        mock_response = Mock()
        mock_response.headers = {"content-type": "text/html"}
        mock_response.raise_for_status = Mock()

        with patch("sotaforge.utils.parsing.requests.get", return_value=mock_response):
            result = await parse_paper_result(doc)

            # Should fall back to abstract
            assert result.text == "This is the abstract"

    @pytest.mark.asyncio
    async def test_parse_paper_vlm_extraction_shorter_than_abstract(self) -> None:
        """Test that abstract is used if VLM extraction is shorter."""
        doc = NotParsedDocument(
            title="arXiv Paper",
            url="https://arxiv.org/abs/2401.12345",
            snippet="Snippet",
            abstract="This is a longer abstract with more content than VLM extract",
        )

        mock_pdf_response = Mock()
        mock_pdf_response.content = b"%PDF-1.4\n%%EOF"
        mock_pdf_response.raise_for_status = Mock()

        mock_page = MagicMock()
        mock_pixmap = MagicMock()
        mock_pixmap.pil_tobytes.return_value = b"fake_png_data"
        mock_page.get_pixmap.return_value = mock_pixmap

        mock_document = MagicMock()
        mock_document.__len__.return_value = 1
        mock_document.__getitem__.return_value = mock_page
        mock_document.close = MagicMock()

        with (
            patch(
                "sotaforge.utils.parsing.requests.get", return_value=mock_pdf_response
            ),
            patch("sotaforge.utils.parsing.fitz.open", return_value=mock_document),
            patch(
                "sotaforge.utils.parsing.parse_single_page_with_vlm",
                new=AsyncMock(return_value="Short"),
            ),
        ):
            result = await parse_paper_result(doc)

            # Should keep abstract since it's longer
            assert (
                result.text
                == "This is a longer abstract with more content than VLM extract"
            )
