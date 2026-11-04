# Preserve research prompt text and whitespace exactly for reproducibility.
# ruff: noqa: E501

from __future__ import annotations

import json
from typing import Any

# This keeps the original BrowseComp-Plus research instruction while making
# termination explicit for our stateful environment. Tool syntax and argument
# shapes are injected by the model's chat template from the schemas below.
BROWSECOMP_PLUS_PROMPT = (
    "You are a deep research agent. You need to answer the given question by "
    "interacting with a search engine, using the search and get_document tools "
    "provided. Please perform reasoning and use the tools step by step, in an "
    "interleaved manner. You may use the search and get_document tools multiple times."
    "\n\nNOTE:"
    "\n- You should always call one tool at a time. Use a concise, targeted query containing the most discriminative clues for each tool call."
    "\n- Before each tool call, provide your reasoning process to explain what you have learned and why this is the next useful call."
    "\n- When you have found the answer, call the finish tool with your succinct final answer."
    "\n\nQuestion: {query}"
)


BROWSECOMP_PLUS_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Perform a search on a knowledge source. Returns top-5 hits with "
                "docid, score, and snippet. The snippet contains the document's "
                "contents and may be truncated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Retrieve the first 4,096 tokens of a document by its docid.",
            "parameters": {
                "type": "object",
                "properties": {
                    "docid": {
                        "type": "string",
                        "description": "Document ID to retrieve.",
                    }
                },
                "required": ["docid"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Submit the succinct final answer and end the episode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "Succinct final answer.",
                    }
                },
                "required": ["answer"],
            },
        },
    },
]


def parse_tool_calls(action: Any) -> tuple[list[dict[str, Any]], bool]:
    """Validate structured calls parsed by the model-specific tool parser."""
    calls = action if isinstance(action, list) else [action]
    if not calls or not all(isinstance(call, dict) for call in calls):
        return [], False

    parsed = []
    for call in calls:
        name = str(call.get("name", "")).strip()
        arguments = call.get("arguments")
        if name not in {"search", "get_document", "finish"} or not isinstance(arguments, dict):
            return [], False
        required_argument = {
            "search": "query",
            "get_document": "docid",
            "finish": "answer",
        }[name]
        value = arguments.get(required_argument)
        if not isinstance(value, str) or not value.strip():
            return [], False
        parsed.append({"name": name, "arguments": arguments})
    return parsed, True


def render_initial_messages(query: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": BROWSECOMP_PLUS_PROMPT.format(query=query)}]


def render_search_results(query: str, results: list[dict[str, Any]]) -> dict[str, str]:
    return {
        "role": "tool",
        "content": json.dumps(
            {"query": query, "results": results},
            ensure_ascii=False,
        ),
    }


def render_document(docid: str, document: dict[str, Any] | None) -> dict[str, str]:
    if document is None:
        content: dict[str, Any] = {"error": f"Document with docid {docid!r} was not found."}
    else:
        content = document
    return {"role": "tool", "content": json.dumps(content, ensure_ascii=False)}


def render_invalid_action() -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": (
                "Invalid action. Call one or more of the available search, "
                "get_document, or finish tools with valid arguments."
            ),
        }
    ]


def render_answer_submitted() -> list[dict[str, str]]:
    return [{"role": "tool", "content": "Final answer submitted."}]
