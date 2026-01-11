"""Filter agent to select relevant documents based on query."""

from typing import Any, Dict, List

from fastmcp import FastMCP
from pydantic import Field, create_model
from pydantic_ai import Agent

from sotaforge.utils.constants import PYDANTIC_AI_MODEL
from sotaforge.utils.db import ChromaStore
from sotaforge.utils.logger import get_logger
from sotaforge.utils.models import DocumentScore

logger = get_logger(__name__)
server = FastMCP("filter")
db_store = ChromaStore()


@server.tool(
    name="filter_results",
    description=(
        "Filters documents from a collection based on provided criteria. "
        "Scores each document on the 5 given criteria (1-5 scale) and "
        "keeps only documents with mean > 2."
    ),
)
async def filter_results(
    query: str, collection: str, criteria: List[str]
) -> Dict[str, Any]:
    """Filter documents from a collection using criteria-based scoring.

    Args:
        collection: The collection name to retrieve and filter from
        query: The research query/topic for context
        criteria: List of 5 criteria strings to score documents on

    Returns:
        Dict with query, criteria, scored documents, and filtered results

    """
    # Step 1: Retrieve documents from collection
    logger.info(f"Retrieving documents from collection: {collection}")
    documents = db_store.fetch_documents(collection)

    if not documents:
        logger.warning(f"No documents found in collection '{collection}'")
        return {
            "query": query,
            "collection": collection,
            "criteria": [],
            "scored_documents": [],
            "filtered_results": [],
            "count": 0,
        }

    logger.info(f"Retrieved {len(documents)} documents from '{collection}'")

    # Step 2: Build dynamic Pydantic model from criteria
    if not criteria or len(criteria) != 5:
        logger.warning(
            "Expected exactly 5 criteria; received %s",
            len(criteria) if criteria else 0,
        )
    logger.info("Using criteria: %s", criteria)

    # Create a dynamic Pydantic model with one field per criterion
    # Each field is an int constrained to 1-5
    field_defs = {
        # Sanitize criterion name to valid Python identifier
        f"criterion_{i + 1}": (int, Field(..., ge=1, le=5, description=criterion))
        for i, criterion in enumerate(criteria)
    }
    DynamicScoreModel = create_model(
        "DynamicScoreModel",
        **field_defs,  # type: ignore[call-overload]
    )

    criteria_str = "\n".join([f"{i + 1}. {c}" for i, c in enumerate(criteria)])
    score_agent = Agent(
        PYDANTIC_AI_MODEL,
        output_type=DynamicScoreModel,
        system_prompt=(
            "You are a strict reviewer. Score the document from 1 (poor) "
            "to 5 (excellent) on each criterion. Return exactly one integer "
            "score per criterion."
        ),
    )

    # Step 3: Score each document on the criteria
    logger.info(f"Scoring {len(documents)} documents on {len(criteria)} criteria...")
    scored_documents = []
    filtered_documents = []

    for doc in documents:
        doc_dict: dict[str, Any]
        if isinstance(doc, dict):
            doc_dict = doc
        else:
            doc_dict = doc.to_dict()
        doc_title = doc_dict.get("title", "Untitled")
        doc_snippet = (
            doc_dict.get("snippet")
            or doc_dict.get("abstract")
            or doc_dict.get("text", "")
        )

        # Build scoring prompt
        scoring_prompt = (
            f"Topic: {query}\n\n"
            f"Criteria:\n{criteria_str}\n\n"
            f"Document:\nTitle: {doc_title}\n"
            f"Excerpt: {doc_snippet[:600]}\n\n"
            f"Score this document on each criterion (1-5)."
        )

        try:
            score_result = await score_agent.run(scoring_prompt)
            # Convert dynamic model output to Dict[str, int]
            scores_dict = {
                criterion: getattr(score_result.output, f"criterion_{i + 1}")
                for i, criterion in enumerate(criteria)
            }
            mean_score = (
                sum(scores_dict.values()) / len(scores_dict) if scores_dict else 0
            )
            keep = mean_score > 2

            scored_doc = DocumentScore(
                title=doc_title,
                scores=scores_dict,
                mean_score=mean_score,
                keep=keep,
            )
            scored_documents.append(scored_doc.to_dict())

            if keep:
                filtered_documents.append(doc_dict)
                logger.debug(f"Kept '{doc_title}' (mean={mean_score:.1f})")
            else:
                logger.debug(f"Filtered '{doc_title}' (mean={mean_score:.1f} <= 2)")
        except Exception as e:
            logger.warning(f"Failed to score '{doc_title}': {e}")

    logger.info(f"Filtered to {len(filtered_documents)}/{len(documents)} documents")

    return {
        "query": query,
        "collection": collection,
        "criteria": criteria,
        "scored_documents": scored_documents,
        "results": filtered_documents,
        "count": len(filtered_documents),
    }


async def launch_filter_server() -> None:
    """Launch the filter server."""
    await server.run_stdio_async()


if __name__ == "__main__":
    import asyncio

    asyncio.run(server.run_stdio_async())
