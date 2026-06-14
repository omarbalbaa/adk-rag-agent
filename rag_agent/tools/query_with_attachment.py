"""
Tool for querying Vertex AI RAG with temporary attachments without saving to corpus.
"""

import logging
import re
from typing import List, Optional

from google.adk.tools.tool_context import ToolContext
from vertexai import rag

from ..config import (
    DEFAULT_DISTANCE_THRESHOLD,
    DEFAULT_TOP_K,
)
from .utils import check_corpus_exists, get_corpus_resource_name

logger = logging.getLogger(__name__)


def query_with_attachment(
    query: str,
    attachment_paths: List[str],
    corpus_name: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Query using temporary attachments without saving them to a corpus.
    Optionally combine with an existing corpus.

    Args:
        query (str): The text query to search for
        attachment_paths (List[str]): List of file URLs or GCS paths to temporarily include.
                                     Supported formats:
                                     - Google Drive: "https://drive.google.com/file/d/{FILE_ID}/view"
                                     - Google Docs/Sheets/Slides: "https://docs.google.com/{type}/d/{FILE_ID}/..."
                                     - Google Cloud Storage: "gs://{BUCKET}/{PATH}"
                                     - Local files: "file://{PATH}" (if accessible)
        corpus_name (Optional[str]): Optional corpus name to query alongside attachments.
                                    If empty/None, only attachments will be queried.
        tool_context (Optional[ToolContext]): The tool context

    Returns:
        dict: The query results and status
    """
    try:
        # Validate inputs
        if not query or not isinstance(query, str):
            return {
                "status": "error",
                "message": "Invalid query: Please provide a non-empty text query",
                "query": query,
            }

        if not attachment_paths or not isinstance(attachment_paths, list):
            return {
                "status": "error",
                "message": "Invalid attachment_paths: Please provide a list of file paths or URLs",
                "query": query,
                "attachment_paths": attachment_paths,
            }

        # Validate and normalize attachment paths
        validated_paths = []
        invalid_paths = []
        conversions = []

        for path in attachment_paths:
            if not path or not isinstance(path, str):
                invalid_paths.append(f"{path} (Not a valid string)")
                continue

            # Check for Google Docs/Sheets/Slides URLs and convert them to Drive format
            docs_match = re.match(
                r"https:\/\/docs\.google\.com\/(?:document|spreadsheets|presentation)\/d\/([a-zA-Z0-9_-]+)(?:\/|$)",
                path,
            )
            if docs_match:
                file_id = docs_match.group(1)
                drive_url = f"https://drive.google.com/file/d/{file_id}/view"
                validated_paths.append(drive_url)
                conversions.append(f"{path} → {drive_url}")
                continue

            # Check for valid Drive URL format
            drive_match = re.match(
                r"https:\/\/drive\.google\.com\/(?:file\/d\/|open\?id=)([a-zA-Z0-9_-]+)(?:\/|$)",
                path,
            )
            if drive_match:
                # Normalize to the standard Drive URL format
                file_id = drive_match.group(1)
                drive_url = f"https://drive.google.com/file/d/{file_id}/view"
                validated_paths.append(drive_url)
                if drive_url != path:
                    conversions.append(f"{path} → {drive_url}")
                continue

            # Check for GCS paths
            if path.startswith("gs://"):
                validated_paths.append(path)
                continue

            # Check for file:// paths
            if path.startswith("file://"):
                validated_paths.append(path)
                continue

            # If we're here, the path wasn't in a recognized format
            invalid_paths.append(f"{path} (Invalid format)")

        if invalid_paths and not validated_paths:
            return {
                "status": "error",
                "message": f"Invalid attachment paths: {', '.join(invalid_paths)}",
                "query": query,
                "attachment_paths": attachment_paths,
            }

        # Log any invalid paths but continue if we have valid ones
        if invalid_paths:
            logger.warning(f"Some invalid paths ignored: {invalid_paths}")

        # Build RAG resources list
        rag_resources = []

        # Add inline data sources for attachments
        print(f"Adding {len(validated_paths)} attachment(s) as inline data source...")
        for path in validated_paths:
            try:
                rag_resources.append(
                    rag.RagResource(
                        rag_inline_data_source=rag.RagInlineDataSource(
                            data=[rag.RagInlineData(uri=path)]
                        )
                    )
                )
            except Exception as e:
                logger.warning(f"Error adding attachment {path}: {str(e)}")
                invalid_paths.append(f"{path} (Failed to create inline source)")

        # Optionally add corpus if specified
        corpus_resource_name = None
        if corpus_name:
            if not check_corpus_exists(corpus_name, tool_context):
                return {
                    "status": "error",
                    "message": f"Corpus '{corpus_name}' does not exist. Provide valid attachment paths or create the corpus first.",
                    "query": query,
                    "attachment_paths": attachment_paths,
                    "corpus_name": corpus_name,
                }
            corpus_resource_name = get_corpus_resource_name(corpus_name)
            rag_resources.append(rag.RagResource(rag_corpus=corpus_resource_name))

        if not rag_resources:
            return {
                "status": "error",
                "message": "No valid attachments or corpus provided for query",
                "query": query,
            }

        # Configure retrieval parameters
        rag_retrieval_config = rag.RagRetrievalConfig(
            top_k=DEFAULT_TOP_K,
            filter=rag.Filter(vector_distance_threshold=DEFAULT_DISTANCE_THRESHOLD),
        )

        # Perform the query
        print("Performing retrieval query with attachments...")
        response = rag.retrieval_query(
            rag_resources=rag_resources,
            text=query,
            rag_retrieval_config=rag_retrieval_config,
        )

        # Process the response into a more usable format
        results = []
        if hasattr(response, "contexts") and response.contexts:
            for ctx_group in response.contexts.contexts:
                result = {
                    "source_uri": (
                        ctx_group.source_uri if hasattr(ctx_group, "source_uri") else ""
                    ),
                    "source_name": (
                        ctx_group.source_display_name
                        if hasattr(ctx_group, "source_display_name")
                        else ""
                    ),
                    "text": ctx_group.text if hasattr(ctx_group, "text") else "",
                    "score": ctx_group.score if hasattr(ctx_group, "score") else 0.0,
                }
                results.append(result)

        # Prepare response
        response_dict = {
            "status": "success",
            "query": query,
            "attachment_count": len(validated_paths),
            "invalid_attachment_count": len(invalid_paths),
            "corpus_queried": bool(corpus_resource_name),
            "corpus_name": corpus_name if corpus_resource_name else None,
            "results": results,
        }

        # Add conversion info if there were any
        if conversions:
            response_dict["path_conversions"] = conversions

        # Add invalid paths info if there were any
        if invalid_paths:
            response_dict["invalid_paths"] = invalid_paths

        return response_dict

    except Exception as e:
        logger.error(f"Error querying with attachments: {str(e)}")
        return {
            "status": "error",
            "message": f"Error during query: {str(e)}",
            "query": query,
            "attachment_paths": attachment_paths,
        }