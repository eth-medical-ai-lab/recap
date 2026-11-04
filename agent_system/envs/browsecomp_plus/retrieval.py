from __future__ import annotations

from functools import lru_cache
from typing import Any

import httpx


def _retrieval_endpoint(mcp_url: str, tool_name: str) -> str:
    """Map the configured MCP URL to the server's low-overhead HTTP route."""
    base_url = mcp_url.rstrip("/")
    if base_url.endswith("/mcp"):
        base_url = base_url[: -len("/mcp")]
    return f"{base_url}/{tool_name}"


@lru_cache(maxsize=16)
def _get_http_client(mcp_url: str, timeout_sec: float) -> httpx.Client:
    del mcp_url
    return httpx.Client(
        timeout=timeout_sec,
        limits=httpx.Limits(max_connections=256, max_keepalive_connections=128),
    )


def _call_retrieval_tool(
    mcp_url: str,
    tool_name: str,
    arguments: dict[str, Any],
    timeout_sec: float,
) -> Any:
    # A persistent HTTP connection avoids the multi-request MCP handshake on
    # every search. The same server still exposes the original MCP tools.
    client = _get_http_client(mcp_url, timeout_sec)
    endpoint = _retrieval_endpoint(mcp_url, tool_name)
    try:
        response = client.post(endpoint, json=arguments)
    except httpx.TransportError:
        # Uvicorn can close an idle keep-alive connection just as a worker
        # reuses it. Retrying once opens a fresh connection in the same pool.
        response = client.post(endpoint, json=arguments)
    response.raise_for_status()
    return response.json()


def search_mcp(mcp_url: str, query: str, timeout_sec: float = 60.0) -> list[dict[str, Any]]:
    """Call the BrowseComp-Plus retrieval service's search route."""
    result = _call_retrieval_tool(mcp_url, "search", {"query": query}, timeout_sec)
    return normalize_search_results(result)


def get_document_mcp(
    mcp_url: str,
    docid: str,
    timeout_sec: float = 60.0,
) -> dict[str, Any] | None:
    """Call the BrowseComp-Plus retrieval service's document route."""
    result = _call_retrieval_tool(mcp_url, "get_document", {"docid": docid}, timeout_sec)
    return normalize_document(result)


def normalize_search_results(result: Any) -> list[dict[str, Any]]:
    """Validate the search HTTP response and normalize its field types."""
    if not isinstance(result, list):
        raise ValueError("The BrowseComp-Plus search tool returned an invalid result.")

    normalized = []
    for item in result:
        if not isinstance(item, dict) or "docid" not in item:
            raise ValueError("Each BrowseComp-Plus search result must contain a docid.")
        snippet = item.get("snippet")
        if snippet is None:
            raise ValueError("Each BrowseComp-Plus search result must contain a snippet.")

        normalized_item: dict[str, Any] = {
            "docid": str(item["docid"]),
            "snippet": str(snippet),
        }
        if item.get("score") is not None:
            normalized_item["score"] = float(item["score"])
        normalized.append(normalized_item)
    return normalized


def normalize_document(result: Any) -> dict[str, Any] | None:
    """Validate the document HTTP response; null means the document was not found."""
    if result is None:
        return None
    if not isinstance(result, dict) or "docid" not in result:
        raise ValueError("The BrowseComp-Plus get_document tool returned an invalid result.")

    text = result.get("text")
    if text is None:
        raise ValueError("The BrowseComp-Plus document result must contain text.")
    return {"docid": str(result["docid"]), "text": str(text)}
