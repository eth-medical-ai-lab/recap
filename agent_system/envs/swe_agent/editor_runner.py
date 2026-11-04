"""Standalone stdlib editor installed once at /opt/swe_agent_editor.py.

Each invocation handles one operation below /testbed and prints a JSON result.
The episode serializes foreground editor and bash actions.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

_TESTBED = Path("/testbed")
_VISIBLE_ROOT = PurePosixPath("/testbed")
_CLIP_MARKER = "\n<response clipped>\n"


class EditorError(ValueError):
    """A deterministic, model-visible validation or editing error."""


def run_editor(arguments: Mapping[str, Any], max_output_chars: int = 20_000) -> dict[str, Any]:
    """Return editor feedback; unexpected implementation failures still raise."""
    try:
        output = _edit(arguments)
        error = False
    except (EditorError, OSError, UnicodeEncodeError) as exc:
        output = str(exc)
        error = True
    return {"output": _clip(output, max_output_chars), "error": error}


def _edit(arguments: Mapping[str, Any]) -> str:
    if not isinstance(arguments, Mapping):
        raise EditorError("Tool arguments must be an object.")
    command, raw_path = arguments.get("command"), arguments.get("path")
    if not isinstance(command, str) or not command:
        raise EditorError("command must be a nonempty string.")
    if not isinstance(raw_path, str) or not raw_path:
        raise EditorError("path must be a nonempty string.")

    allowed_arguments = {
        "view": {"command", "path", "view_range"},
        "create": {"command", "path", "file_text"},
        "str_replace": {"command", "path", "old_str", "new_str"},
        "insert": {"command", "path", "insert_line", "new_str"},
    }
    if command not in allowed_arguments:
        raise EditorError(f"Unknown command {command!r}. Allowed commands: {', '.join(allowed_arguments)}.")
    unexpected = sorted(set(arguments) - allowed_arguments[command])
    if unexpected:
        raise EditorError(f"Unexpected argument(s) for {command}: {', '.join(unexpected)}.")
    path, visible_path = _resolve_path(raw_path)

    if command == "view":
        if not path.exists():
            raise EditorError(f"Path does not exist: {visible_path}")
        if path.is_dir():
            if "view_range" in arguments:
                raise EditorError("view_range can only be used when viewing a file.")
            return _view_directory(path, visible_path)
        if not path.is_file():
            raise EditorError(f"Path is not a regular file or directory: {visible_path}")
        lines = _read_text(path, visible_path).splitlines()
        start, end = _validate_view_range(arguments.get("view_range"), len(lines))
        return "\n".join(f"{number:6}\t{line}" for number, line in enumerate(lines[start - 1 : end], start=start))

    if command == "create":
        file_text = _required_string(arguments, "file_text", "create")
        if path.exists() or path.is_symlink():
            raise EditorError(f"Cannot create a file because the path already exists: {visible_path}")
        if not path.parent.is_dir():
            raise EditorError(f"Parent directory does not exist: {PurePosixPath(visible_path).parent}")
        path.write_bytes(file_text.encode("utf-8"))
        return f"File created successfully at {visible_path}."

    if not path.exists():
        raise EditorError(f"File does not exist: {visible_path}")
    if not path.is_file():
        raise EditorError(f"Path is not a regular file: {visible_path}")
    text = _read_text(path, visible_path)
    if command == "str_replace":
        old_str = _required_string(arguments, "old_str", "str_replace")
        new_str = arguments.get("new_str", "")
        if not isinstance(new_str, str):
            raise EditorError("new_str must be a string for str_replace.")
        if not old_str:
            raise EditorError("old_str must not be empty for str_replace.")
        occurrences = text.count(old_str)
        if occurrences == 0:
            raise EditorError("old_str was not found in the file.")
        if occurrences != 1:
            raise EditorError(f"old_str must match exactly once, but matched {occurrences} times.")
        updated = text.replace(old_str, new_str, 1)
    else:
        new_str = _required_string(arguments, "new_str", "insert")
        insert_line = arguments.get("insert_line")
        if isinstance(insert_line, bool) or not isinstance(insert_line, int):
            raise EditorError("insert_line must be an integer.")
        lines = text.splitlines(keepends=True)
        if insert_line < 0 or insert_line > len(lines):
            raise EditorError(f"insert_line must be between 0 and {len(lines)}.")

        # Preserve the original newline bytes; add only missing separators at
        # the insertion boundary, matching the existing editor behavior.
        offset = sum(len(line) for line in lines[:insert_line])
        before, after = text[:offset], text[offset:]
        if before and not before.endswith(("\n", "\r")):
            before += "\n"
        if after and new_str and not new_str.endswith(("\n", "\r")):
            new_str += "\n"
        updated = before + new_str + after
    path.write_bytes(updated.encode("utf-8"))
    return f"The file {visible_path} has been edited."


def _resolve_path(raw_path: str) -> tuple[Path, str]:
    """Resolve relative paths from /testbed and reject paths that escape it."""
    if "\x00" in raw_path or "\\" in raw_path:
        raise EditorError("path must be a POSIX path below /testbed or relative to it.")
    # Match bash's starting directory, including the common ./file form.
    if not raw_path.startswith("/"):
        raw_path = str(_VISIBLE_ROOT / raw_path)
    raw_parts = raw_path.split("/")
    if ".." in raw_parts or "." in raw_parts:
        raise EditorError("path must be a canonical absolute path below /testbed.")
    visible = PurePosixPath(raw_path)
    if raw_path != str(visible):
        raise EditorError("path must be a canonical absolute path below /testbed.")
    if visible != _VISIBLE_ROOT and _VISIBLE_ROOT not in visible.parents:
        raise EditorError("path must be /testbed or a path below /testbed.")
    if ".git" in visible.parts:
        raise EditorError("Accessing Git metadata is not allowed.")
    try:
        path = _TESTBED.joinpath(*visible.relative_to(_VISIBLE_ROOT).parts).resolve(strict=False)
    except RuntimeError as exc:
        # Python versions before 3.13 report symlink loops as RuntimeError.
        raise EditorError("Could not resolve the requested path.") from exc
    if path != _TESTBED and _TESTBED not in path.parents:
        raise EditorError("path resolves outside /testbed through a symbolic link.")
    if ".git" in path.relative_to(_TESTBED).parts:
        raise EditorError("Accessing Git metadata is not allowed.")
    return path, str(visible)


def _view_directory(path: Path, visible_path: str) -> str:
    entries = []
    # List two actual levels, excluding hidden entries and never following
    # directory symlinks into other directories or outside the episode root.
    for child in path.iterdir():
        if child.name.startswith("."):
            continue
        child_visible = f"{visible_path}/{child.name}"
        is_directory = not child.is_symlink() and child.is_dir()
        entries.append(child_visible + ("/" if is_directory else ""))
        if is_directory:
            for grandchild in child.iterdir():
                if grandchild.name.startswith("."):
                    continue
                suffix = "/" if not grandchild.is_symlink() and grandchild.is_dir() else ""
                entries.append(f"{child_visible}/{grandchild.name}{suffix}")
    return "\n".join(sorted(entries))


def _read_text(path: Path, visible_path: str) -> str:
    try:
        return path.read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EditorError(f"File is not valid UTF-8 plain text: {visible_path}") from exc


def _required_string(arguments: Mapping[str, Any], name: str, command: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str):
        raise EditorError(f"{name} must be a string for {command}.")
    return value


def _validate_view_range(value: Any, line_count: int) -> tuple[int, int]:
    if value is None:
        return 1, line_count
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise EditorError("view_range must contain exactly two integers.")
    start, end = value
    if start < 1:
        raise EditorError("view_range must start at line 1 or later.")
    if end == -1:
        end = line_count
    if line_count == 0 and start == 1 and end == 0:
        return 1, 0
    if end < start:
        raise EditorError("view_range end must be -1 or greater than or equal to its start.")
    if start > line_count and line_count > 0:
        raise EditorError(f"view_range starts after the file's last line ({line_count}).")
    return start, min(end, line_count)


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    if limit <= len(_CLIP_MARKER):
        return _CLIP_MARKER[:limit]
    payload = limit - len(_CLIP_MARKER)
    head = payload // 2
    tail = payload - head
    return text[:head] + _CLIP_MARKER + text[-tail:]


if __name__ == "__main__":
    arguments = json.loads(base64.b64decode(sys.argv[1]))
    result = run_editor(arguments, int(sys.argv[2]))
    print(json.dumps(result, ensure_ascii=True))
