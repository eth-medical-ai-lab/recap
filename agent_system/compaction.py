# Preserve research prompt text and whitespace exactly for reproducibility.
# ruff: noqa: E501

from __future__ import annotations

from copy import deepcopy
from typing import Any

COMPACTION_INSTRUCTION = """The current task is paused for context compaction. For this turn, write a handoff summary for another agent that will resume the original task.

Include:

- Current progress and key decisions made
- Important context, constraints, or observations
- What remains to be done (clear next steps)
- Any critical data, examples, or references needed to continue
- Be concise, structured, and focused on helping the next agent seamlessly continue the work.

Do not call any tools in this turn. Return only the handoff summary. The entire response will be used as the summary for the next agent.
"""


def get_compaction_instruction(env_name: str) -> str:
    """Render the released handoff prompt with the environment's native tools."""
    if env_name.startswith("swe_agent"):
        tool_names = "`bash`, `str_replace_editor` and `finish`"
    elif env_name.startswith("browsecomp_plus"):
        tool_names = "`search`, `get_document` and `finish`"
    else:
        return COMPACTION_INSTRUCTION
    return COMPACTION_INSTRUCTION.replace(
        "Do not call any tools in this turn.",
        f"DO not call any tools in this turn (including {tool_names}).",
    )


def flatten_retained_turn_tokens(
    recent_turn_token_ids: list[list[int]],
    keep_recent_turns: int,
) -> tuple[list[int], int]:
    """Return an exact whole-turn token suffix and the number of retained turns."""
    retained_turn_count = min(keep_recent_turns, len(recent_turn_token_ids))
    retained_turns = recent_turn_token_ids[-retained_turn_count:] if retained_turn_count else []
    retained_ids = [token_id for turn_ids in retained_turns for token_id in turn_ids]
    return retained_ids, retained_turn_count


def build_compacted_messages(
    initial_messages: list[dict[str, Any]],
    summary: str,
    *,
    for_compaction: bool = False,
) -> list[dict[str, Any]]:
    """Build the fixed prompt from the original task and compacted summary.

    The complete initial prompt, including demonstrations and the actual task,
    stays unchanged. Retained raw turns are appended later from their exact
    token IDs, so they are not re-tokenized or counted against prompt_length.
    During compaction, frame the previous summary as material to update.
    """
    messages = deepcopy(initial_messages)
    resume = (
        "Another agent started to solve this problem and produced a summary of its thinking process. "
        "You also have access to the actions and observations that were obtained by that agent. "
        "Use this to build on the work that has already been done and avoid duplicating work. "
        "Here is the summary produced by the other agent, use the information in this summary to assist with your own analysis:\n\n"
        f"{summary.strip()}\n"
    )
    if for_compaction:
        resume = (
            "The following is a historical handoff summary from an earlier context compaction. "
            "Use it together with the subsequent conversation as source material for an updated handoff summary.\n\n"
            "Previous handoff summary:\n"
            f"{summary.strip()}\n"
        )
    messages.append({"role": "user", "content": resume})
    return messages
