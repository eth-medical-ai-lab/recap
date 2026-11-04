# Preserve research prompt text and whitespace exactly for reproducibility.
# ruff: noqa: E501

"""The model interface: instructions, tool schemas, action parsing and feedback."""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

# Keep all instructions in the first user message so compaction needs no system prompt.
TASK_INSTRUCTIONS = """You are an autonomous software engineer working in the repository at /testbed. Your goal is to resolve the issue described by the user with a correct and maintainable patch.

## Recommended workflow

Work step by step so you can learn from command output and test results:

1. Explore the repository and read the relevant code.
2. Reproduce or clearly identify the reported problem.
3. Determine the underlying cause.
4. Edit the source code to fix the issue.
5. Run focused tests or a reproduction to verify the fix.
6. Check relevant edge cases and run broader tests when practical.
7. Submit the completed work with the finish tool.

Adapt this workflow when a step is unnecessary or impractical.

## Tool-use rules

While actively solving the issue, briefly explain what you learned and why
the next action is useful, then call one tool: bash, str_replace_editor,
or finish. Do not make multiple or parallel tool calls in one response.

Use bash to inspect the repository, search code, run programs, and execute tests. Use str_replace_editor for precise file viewing, creation, and editing.

The repository filesystem persists across tool calls. Each bash call starts in /testbed, but shell directory changes and environment variables do not persist to later calls.

Do not use Git commands. Git metadata is intentionally unavailable, and the environment captures your patch automatically. Do not weaken or delete tests merely to make them pass. Keep the patch focused on the reported issue and preserve unrelated behavior.

Avoid commands that produce unnecessarily large output. Prefer targeted file views, searches, and test commands. If a command fails, inspect its output and continue diagnosing the problem.

When the issue is fixed and adequately tested, briefly summarize the verification and call finish with empty arguments. After calling finish, you cannot make further changes."""
# Verified exposes a shallow task checkout; SWE-smith keeps its original prompt.
GIT_TASK_INSTRUCTIONS = TASK_INSTRUCTIONS.replace(
    "Do not use Git commands. Git metadata is intentionally unavailable, and the environment captures your patch "
    "automatically. Do not weaken or delete tests merely to make them pass. Keep the patch focused on the reported "
    "issue and preserve unrelated behavior.\n\n",
    "",
)
QWEN35_MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

# Native tool definitions and strict validation of one model action.
BASH_TOOL_NAME = "bash"
EDITOR_TOOL_NAME = "str_replace_editor"
FINISH_TOOL_NAME = "finish"

BASH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": BASH_TOOL_NAME,
        "description": (
            "Run one shell command in the repository. Each call starts a fresh "
            "bash shell in /testbed, while filesystem changes persist across calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to run in /testbed.",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
}

EDITOR_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": EDITOR_TOOL_NAME,
        "description": (
            "Custom editing tool for viewing, creating, and editing plain-text files. "
            "State persists for this task. File views use cat -n style line numbers; "
            "directory views list sorted non-hidden entries up to two levels. "
            "str_replace requires old_str to match exactly once."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["view", "create", "str_replace", "insert"],
                    "description": "The editor operation to perform.",
                },
                "path": {
                    "type": "string",
                    "description": "Absolute path below /testbed, or a relative path from /testbed (e.g. src/file.py).",
                },
                "file_text": {
                    "type": "string",
                    "description": "Complete file content required by create.",
                },
                "old_str": {
                    "type": "string",
                    "description": "Exact unique text required by str_replace.",
                },
                "new_str": {
                    "type": "string",
                    "description": (
                        "Replacement text for str_replace, or text required by insert. "
                        "Omit it from str_replace to delete old_str."
                    ),
                },
                "insert_line": {
                    "type": "integer",
                    "description": ("Insert new_str after this one-based line. Use 0 to insert before the first line."),
                },
                "view_range": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": (
                        "Optional inclusive [start_line, end_line] for file view. "
                        "Use -1 as end_line to read through the end."
                    ),
                },
            },
            "required": ["command", "path"],
            "additionalProperties": False,
        },
    },
}

FINISH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": FINISH_TOOL_NAME,
        "description": "Finish the task and submit the current repository changes for evaluation.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}

SWE_AGENT_TOOL_SCHEMAS = [BASH_TOOL_SCHEMA, EDITOR_TOOL_SCHEMA, FINISH_TOOL_SCHEMA]

_COMPLETE_CALL_RE = re.compile(r"(?s)<tool_call>\s*<function=([^>\n]+)>.*?</function>\s*</tool_call>")


class ToolCallError(ValueError):
    """A deterministic, model-visible native tool-call error."""


@dataclass(frozen=True)
class ToolAction:
    """One validated native tool call plus prompt-compliance metadata."""

    name: str
    arguments: dict[str, Any]
    tool_call_id: str
    has_rationale: bool


def tool_schemas() -> list[dict[str, Any]]:
    """Return fresh schema dictionaries so an environment cannot mutate globals."""
    return deepcopy(SWE_AGENT_TOOL_SCHEMAS)


def parse_tool_action(action: Any, *, require_qwen_xml: bool = True) -> ToolAction:
    """Validate one call; the training path also requires complete generated XML."""
    if not isinstance(action, dict):
        raise ToolCallError("Make exactly one native tool call.")

    raw_response = action.get("raw_response", "" if not require_qwen_xml else None)
    assistant_content = action.get("assistant_content", "")
    calls = action.get("tool_calls")
    if not isinstance(raw_response, str) or not isinstance(assistant_content, str):
        raise ToolCallError("The native tool-call envelope is invalid.")
    if not isinstance(calls, list) or len(calls) != 1:
        count = len(calls) if isinstance(calls, list) else 0
        raise ToolCallError(f"Make exactly one native tool call; received {count}.")

    # Verl's shared Qwen parser intentionally recovers partial XML for other
    # environments. SWE tasks are stricter at their own boundary.
    if require_qwen_xml:
        matches = list(_COMPLETE_CALL_RE.finditer(raw_response))
        if len(matches) != 1 or raw_response[matches[0].end() :].strip():
            raise ToolCallError("The response must end after one complete native tool call.")
        if raw_response[: matches[0].start()].count("<tool_call>"):
            raise ToolCallError("Make exactly one complete native tool call.")
        call_text = matches[0].group(0)
        if (
            call_text.count("<tool_call>") != 1
            or call_text.count("</tool_call>") != 1
            or call_text.count("<function=") != 1
            or call_text.count("</function>") != 1
            or call_text.count("<parameter=") != call_text.count("</parameter>")
        ):
            raise ToolCallError("The native tool-call XML is incomplete or nested.")

    call = calls[0]
    if not isinstance(call, dict):
        raise ToolCallError("The parsed tool call is invalid.")
    name = call.get("name")
    arguments = call.get("arguments")
    tool_call_id = call.get("tool_call_id")
    if require_qwen_xml and name != matches[0].group(1).strip():
        raise ToolCallError("The parsed tool name does not match the generated call.")
    if name not in {BASH_TOOL_NAME, EDITOR_TOOL_NAME, FINISH_TOOL_NAME}:
        raise ToolCallError(f"Unknown tool {name!r}.")
    if not isinstance(arguments, dict):
        raise ToolCallError("Tool arguments must be an object.")
    if not isinstance(tool_call_id, str) or not tool_call_id:
        raise ToolCallError("The tool call is missing its call ID.")

    if name == BASH_TOOL_NAME:
        _require_exact_keys(arguments, required={"command"})
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ToolCallError("bash.command must be a nonempty string.")
        # exec cannot accept literal NUL bytes; let the model correct its command.
        if "\x00" in command:
            raise ToolCallError("bash.command contains a NUL byte. Use an escaped \\x00 in your script instead.")
    elif name == FINISH_TOOL_NAME:
        _require_exact_keys(arguments, required=set())
    else:
        _validate_editor_arguments(arguments)

    return ToolAction(
        name=name,
        arguments=arguments,
        tool_call_id=tool_call_id,
        has_rationale=bool(assistant_content.strip()),
    )


def _validate_editor_arguments(arguments: dict[str, Any]) -> None:
    command = arguments.get("command")
    path = arguments.get("path")
    if not isinstance(command, str) or command not in {"view", "create", "str_replace", "insert"}:
        raise ToolCallError("str_replace_editor.command is invalid.")
    if not isinstance(path, str) or not path:
        raise ToolCallError("str_replace_editor.path must be a nonempty string.")

    allowed = {
        "view": {"command", "path", "view_range"},
        "create": {"command", "path", "file_text"},
        "str_replace": {"command", "path", "old_str", "new_str"},
        "insert": {"command", "path", "new_str", "insert_line"},
    }[command]
    unexpected = set(arguments) - allowed
    if unexpected:
        raise ToolCallError(f"Unexpected str_replace_editor argument(s): {', '.join(sorted(unexpected))}.")

    required = {
        "view": set(),
        "create": {"file_text"},
        "str_replace": {"old_str"},
        "insert": {"new_str", "insert_line"},
    }[command]
    missing = required - set(arguments)
    if missing:
        raise ToolCallError(f"Missing str_replace_editor argument(s): {', '.join(sorted(missing))}.")


def _require_exact_keys(arguments: dict[str, Any], *, required: set[str]) -> None:
    missing = required - set(arguments)
    unexpected = set(arguments) - required
    if missing:
        raise ToolCallError(f"Missing tool argument(s): {', '.join(sorted(missing))}.")
    if unexpected:
        raise ToolCallError(f"Unexpected tool argument(s): {', '.join(sorted(unexpected))}.")


# Model-visible observations and feedback for blocked commands.
_HIDDEN_GIT_DIR = "/opt/agl_tmp"

# These filters mirror the frozen Agent Lightning scaffold. The sandbox backend
# also denies network access and hides Git, so the filters provide useful model
# feedback while the environment remains the authoritative security boundary.
_GIT_INVOKE_RE = re.compile(
    r"(?:^|[\n;`(]|&&|\|\|?|\$\()\s*(?:\w+=\S+\s+)*(?:[\w./-]*/)?git(?:-[a-z]+)?(?=\s|$|;|&|\|)",
    re.I,
)
_GIT_ACCESS_RE = re.compile(r'--git-dir|--work-tree|(?:^|[\s=:"\'/])\.git(?:/|\b)', re.I)
_NET_FETCH_RE = re.compile(
    r"(?:^|[\n;`(]|&&|\|\|?|\$\()\s*(?:\w+=\S+\s+)*(?:[\w./-]*/)?"
    r"(?:curl|wget|httpie|http|https|aria2c|scp|sftp|rsync|nc|ncat|netcat|telnet)"
    r"(?=\s|$|;|&|\|)",
    re.I,
)
_PKG_INSTALL_RE = re.compile(
    r"(?:^|[\n;`(]|&&|\|\|?|\$\()\s*(?:\w+=\S+\s+)*(?:"
    r"(?:[\w./-]*/)?(?:pip|pip3|conda|mamba|easy_install|uv)\b[^\n;&|]*?\binstall\b"
    r"|(?:[\w./-]*/)?python[0-9.]*\s+-m\s+pip\b[^\n;&|]*?\binstall\b)",
    re.I,
)
_PY_NET_RE = re.compile(
    r"urllib\.request|\burlopen\b|\brequests\.(?:get|post|put|head|Session)\b|"
    r"\bhttpx\.|\bsocket\.(?:socket|create_connection)\b|\burllib3\b",
    re.I,
)
_TEST_TAMPER_RE = re.compile(
    r'(?:>>?|\btee\b(?:\s+-a)?\s+)\s*[\'"]?[^\s\'"|;&<>]*'
    r"(?:conftest\.py|pytest\.ini|tox\.ini|sitecustomize\.py|usercustomize\.py|"
    r'setup\.cfg|pyproject\.toml|\.pth)(?=[\'"\s;&|]|$)',
    re.I,
)

_OVERLONG_WARNING = (
    "The output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can use head, tail or sed to view a smaller "
    "number of lines selectively.\n"
    "If you're using grep or find and it produced too much output, use a more "
    "selective search pattern.\n"
    "If you really need the full output, redirect it to a file and search within it."
)


def render_initial_messages(problem_statement: str, *, allow_git: bool = False) -> list[dict[str, str]]:
    instructions = GIT_TASK_INSTRUCTIONS if allow_git else TASK_INSTRUCTIONS
    return [
        {"role": "user", "content": f"{instructions}\n\n## Issue\n\n{problem_statement}"},
    ]


def render_observation(return_code: int, output: str, character_cap: int) -> str:
    if character_cap <= 0:
        raise ValueError("character_cap must be positive")
    if len(output) <= character_cap:
        return f"<returncode>{return_code}</returncode>\n<output>\n{output}</output>"
    head_size = character_cap // 2
    tail_size = character_cap - head_size
    elided = len(output) - character_cap
    return (
        f"<returncode>{return_code}</returncode>\n"
        f"<warning>\n{_OVERLONG_WARNING}\n</warning>\n"
        f"<output_head>\n{output[:head_size]}\n</output_head>\n"
        f"<elided_chars>\n{elided} characters elided\n</elided_chars>\n"
        f"<output_tail>\n{output[-tail_size:]}\n</output_tail>"
    )


def invalid_tool_message(reason: str) -> str:
    return (
        f"Invalid tool call: {reason}\n"
        "Briefly explain your next action, then make exactly one native call to "
        "bash, str_replace_editor, or finish."
    )


def forbidden_action(action: str, *, allow_git: bool = False, filter_network_text: bool = True) -> str | None:
    if not allow_git and _GIT_INVOKE_RE.search(action):
        return (
            "git is disabled in this environment; do not use it for any purpose. "
            "Inspect and edit the source files under /testbed directly (cat, grep, "
            "sed, python) to fix the bug."
        )
    if not allow_git and (_GIT_ACCESS_RE.search(action) or _HIDDEN_GIT_DIR in action):
        return (
            "Accessing the git metadata directory is not allowed. Work only with "
            "the source files under /testbed; do not read .git."
        )
    if filter_network_text and (_NET_FETCH_RE.search(action) or _PY_NET_RE.search(action)):
        return (
            "Network access is disabled. Do not fetch code from the internet "
            "(curl, wget, urllib, requests, etc.); all dependencies are already "
            "installed. Solve the bug using only the source files already present "
            "under /testbed."
        )
    if _PKG_INSTALL_RE.search(action):
        return (
            "Installing packages is not allowed. Everything needed to run the code "
            "and its tests is already installed. Fix the bug by editing the source "
            "under /testbed; do not install anything."
        )
    if _TEST_TAMPER_RE.search(action):
        return (
            "Modifying test-harness or config files (conftest.py, pytest.ini, "
            "tox.ini, setup.cfg, pyproject.toml, sitecustomize.py, .pth) is not "
            "allowed. Fix the bug in the source under /testbed instead."
        )
    return None
