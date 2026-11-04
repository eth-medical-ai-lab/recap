from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from agent_system.env_base import ChatMessage, TaskId
from agent_system.envs.browsecomp_plus.judge import (
    JudgeResult,
    exact_match,
    judge_answer,
)
from agent_system.envs.browsecomp_plus.prompts import (
    BROWSECOMP_PLUS_TOOL_SCHEMAS,
    parse_tool_calls,
    render_answer_submitted,
    render_document,
    render_initial_messages,
    render_invalid_action,
    render_search_results,
)
from agent_system.envs.browsecomp_plus.retrieval import get_document_mcp, search_mcp

SearchFunction = Callable[[str], list[dict[str, Any]]]
DocumentFunction = Callable[[str], dict[str, Any] | None]
JudgeFunction = Callable[[str, str, str], JudgeResult]


@lru_cache(maxsize=16)
def _load_tasks(data_file: str) -> tuple[dict[str, Any], ...]:
    """Load the compact decrypted task file once per environment-worker process."""
    tasks = []
    seen_ids = set()
    with Path(data_file).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            task = json.loads(line)
            task_id = str(task.get("query_id", ""))
            if not task_id or not str(task.get("query", "")).strip() or not str(task.get("answer", "")).strip():
                raise ValueError(f"Invalid BrowseComp-Plus task at {data_file}:{line_number}.")
            if task_id in seen_ids:
                raise ValueError(f"Duplicate BrowseComp-Plus query_id {task_id!r} in {data_file}.")
            seen_ids.add(task_id)
            tasks.append(task)

    if not tasks:
        raise ValueError(f"BrowseComp-Plus task file is empty: {data_file}")
    return tuple(tasks)


def _resolve_data_file(data_path: str, split: str) -> str:
    path = Path(data_path).expanduser()
    if path.is_dir():
        path = path / f"{split}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"BrowseComp-Plus task file not found: {path}")
    return str(path.resolve())


class BrowseCompPlusEnv:
    """Synchronous BC+ episode, exposed asynchronously through its Ray worker.

    Rollout workers keep both large models out of process. Search calls go to the
    MCP retriever, while non-exact final answers go to an OpenAI-compatible judge.
    """

    def __init__(
        self,
        data_path: str,
        split: str = "test",
        mcp_url: str = "http://127.0.0.1:8080/mcp",
        search_timeout_sec: float = 60.0,
        judge_url: str | None = None,
        judge_model: str = "Qwen/Qwen3.8-27B",
        judge_timeout_sec: float = 120.0,
        judge_max_retries: int = 3,
        defer_judge: bool = False,
        search_fn: SearchFunction | None = None,
        get_document_fn: DocumentFunction | None = None,
        judge_fn: JudgeFunction | None = None,
    ) -> None:
        self.data_path = data_path
        self.split = split
        self.mcp_url = mcp_url
        self.search_timeout_sec = search_timeout_sec
        self.judge_url = judge_url
        self.judge_model = judge_model
        self.judge_timeout_sec = judge_timeout_sec
        self.judge_max_retries = judge_max_retries
        self.defer_judge = defer_judge
        self.search_fn = search_fn
        self.get_document_fn = get_document_fn
        self.judge_fn = judge_fn

        self.task: dict[str, Any] | None = None
        self.task_id: str | None = None
        self.search_count = 0
        self.get_document_count = 0
        self.finish_count = 0
        self.retrieved_docids: list[str] = []
        self._retrieved_docid_set: set[str] = set()
        self._finished = False

    @classmethod
    def list_task_specs(cls, split: str = "test", **kwargs: Any) -> list[dict[str, Any]]:
        data_path = kwargs.get("data_path")
        if not data_path:
            raise ValueError("BrowseCompPlusEnv.list_task_specs requires data_path.")
        data_file = _resolve_data_file(str(data_path), split)
        return [
            {
                "env_name": "browsecomp_plus",
                "split": split,
                "task_id": str(task["query_id"]),
                "uid": f"browsecomp_plus:{split}:{task['query_id']}",
            }
            for task in _load_tasks(data_file)
        ]

    def reset(
        self,
        task_id: TaskId | None = None,
        split: str | None = None,
    ) -> tuple[list[ChatMessage], dict[str, Any]]:
        self.close()
        if split is not None:
            self.split = split

        data_file = _resolve_data_file(self.data_path, self.split)
        tasks = _load_tasks(data_file)
        requested_id = str(task_id) if task_id is not None else str(tasks[0]["query_id"])
        task_by_id = {str(task["query_id"]): task for task in tasks}
        if requested_id not in task_by_id:
            raise KeyError(f"Unknown BrowseComp-Plus query_id {requested_id!r} in {data_file}.")

        self.task = task_by_id[requested_id]
        self.task_id = requested_id
        reset_info = {
            "env_name": "browsecomp_plus",
            "split": self.split,
            "task_id": self.task_id,
            "query_id": self.task_id,
            "query": str(self.task["query"]),
            "defer_judge": self.defer_judge,
            "mcp_url": self.mcp_url,
            "tool_schemas": deepcopy(BROWSECOMP_PLUS_TOOL_SCHEMAS),
        }
        return render_initial_messages(str(self.task["query"])), reset_info

    def step(self, action: Any) -> tuple[list[ChatMessage], float, bool, dict[str, Any]]:
        if self.task is None or self.task_id is None:
            raise RuntimeError("Call reset() before step().")
        if self._finished:
            raise RuntimeError("The BrowseComp-Plus episode is already finished.")

        tool_calls, format_ok = parse_tool_calls(action)
        if not format_ok:
            info = self._episode_info(format_ok=False)
            info.update({"action_type": "invalid", "raw_action": action})
            return render_invalid_action(), 0.0, False, info

        observation_messages = []
        action_names = []
        final_answer = None
        for tool_call in tool_calls:
            name = tool_call["name"]
            arguments = tool_call["arguments"]
            action_names.append(name)

            if name == "search":
                query = str(arguments["query"]).strip()
                results = self._search(query)
                self.search_count += 1
                for result in results:
                    self._track_docid(result["docid"])
                observation_messages.append(render_search_results(query, results))
                continue

            if name == "get_document":
                docid = str(arguments["docid"]).strip()
                document = self._get_document(docid)
                self.get_document_count += 1
                if document is not None:
                    self._track_docid(document["docid"])
                observation_messages.append(render_document(docid, document))
                continue

            self.finish_count += 1
            final_answer = str(arguments["answer"]).strip()
            observation_messages.extend(render_answer_submitted())
            break

        info = self._episode_info(format_ok=True)
        info.update(
            {
                "action_type": action_names[0] if len(action_names) == 1 else "multiple",
                "tool_calls": action_names,
            }
        )
        if final_answer is None:
            return observation_messages, 0.0, False, info

        question = str(self.task["query"])
        reference_answer = str(self.task.get("answer", ""))
        reward, reward_info = self._score_answer(
            question=question,
            reference_answer=reference_answer,
            predicted_answer=final_answer,
        )
        self._finished = True
        info.update(
            {
                "completed": True,
                "terminated": True,
                "truncated": False,
                "status": "completed",
                "final_answer": final_answer,
                "reference_answer": reference_answer,
                "reward": reward,
                "result": [{"type": "output_text", "output": final_answer}],
                **reward_info,
            }
        )
        return observation_messages, reward, True, info

    def truncate(self, reason: str) -> tuple[list[ChatMessage], float, bool, dict[str, Any]]:
        """End an unanswered episode without submitting an answer or calling the judge."""
        if self.task is None or self.task_id is None:
            raise RuntimeError("Call reset() before truncate().")
        if self._finished:
            raise RuntimeError("The BrowseComp-Plus episode is already finished.")
        if not isinstance(reason, str) or not reason:
            raise ValueError("truncate reason must be a nonempty string")

        # Finalization is not a model action: preserve counters and omit format_ok.
        self._finished = True
        info = self._episode_info(format_ok=None)
        info.update(
            {
                "completed": True,
                "terminated": False,
                "truncated": True,
                "status": reason,
                "termination_reason": reason,
                "action_type": "rollout_limit",
                "tool_calls": [],
                "reward": 0.0,
                "reward_pending": False,
            }
        )
        return [], 0.0, True, info

    def close(self) -> None:
        self.task = None
        self.task_id = None
        self.search_count = 0
        self.get_document_count = 0
        self.finish_count = 0
        self.retrieved_docids = []
        self._retrieved_docid_set = set()
        self._finished = False

    def _search(self, query: str) -> list[dict[str, Any]]:
        if self.search_fn is not None:
            return self.search_fn(query)
        return search_mcp(self.mcp_url, query, timeout_sec=self.search_timeout_sec)

    def _get_document(self, docid: str) -> dict[str, Any] | None:
        if self.get_document_fn is not None:
            return self.get_document_fn(docid)
        return get_document_mcp(self.mcp_url, docid, timeout_sec=self.search_timeout_sec)

    def _track_docid(self, docid: Any) -> None:
        docid = str(docid)
        if docid not in self._retrieved_docid_set:
            self._retrieved_docid_set.add(docid)
            self.retrieved_docids.append(docid)

    def _score_answer(
        self,
        question: str,
        reference_answer: str,
        predicted_answer: str,
    ) -> tuple[float, dict[str, Any]]:
        # Obvious matches are free, deterministic, and avoid unnecessary judge load.
        if exact_match(reference_answer, predicted_answer):
            return 1.0, {
                "reward_source": "normalized_exact_match",
                "reward_pending": False,
                "judge": None,
            }

        # Evaluation can collect complete rollouts first and judge them in one
        # separate job. The placeholder reward is never used as the final score.
        if self.defer_judge:
            return 0.0, {
                "reward_source": "deferred_llm_judge",
                "reward_pending": True,
                "judge": None,
            }

        if self.judge_fn is not None:
            result = self.judge_fn(question, reference_answer, predicted_answer)
        else:
            if not self.judge_url:
                raise RuntimeError("A non-exact BrowseComp-Plus answer requires trainer.gym_env_kwargs.judge_url.")
            result = judge_answer(
                question=question,
                reference_answer=reference_answer,
                predicted_answer=predicted_answer,
                judge_url=self.judge_url,
                model=self.judge_model,
                timeout_sec=self.judge_timeout_sec,
                max_retries=self.judge_max_retries,
            )

        if result.score not in {0, 1}:
            raise ValueError(f"BrowseComp-Plus judge returned non-binary score {result.score!r}.")
        return float(result.score), {
            "reward_source": "llm_judge",
            "reward_pending": False,
            "judge": {
                "model": self.judge_model,
                "score": result.score,
                "seed": result.seed,
                "attempts": result.attempts,
                "extracted_final_answer": result.extracted_final_answer,
                "reasoning": result.reasoning,
                "raw_response": result.raw_response,
            },
        }

    def _episode_info(self, format_ok: bool | None) -> dict[str, Any]:
        """Return changing counters; fixed task/configuration metadata belongs to reset()."""
        info = {
            "search_count": self.search_count,
            "get_document_count": self.get_document_count,
            "finish_count": self.finish_count,
            "tool_call_counts": {
                "search": self.search_count,
                "get_document": self.get_document_count,
                "finish": self.finish_count,
            },
            "retrieved_docids": list(self.retrieved_docids),
        }
        if format_ok is not None:
            info["format_ok"] = format_ok
        return info
