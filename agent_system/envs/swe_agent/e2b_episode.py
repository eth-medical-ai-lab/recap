"""Direct asynchronous E2B sandbox for one SWE-agent rollout."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shlex
import tempfile
import time
import weakref
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from uuid import uuid4

from agent_system.envs.swe_agent import EvaluationResult, SweAgentServiceError
from agent_system.envs.swe_agent.editor import RUNNER_PATH, AsyncE2BEditor, EditorError
from agent_system.envs.swe_agent.tasks import DATASET_REVISION, HARNESS_REVISION

_TESTBED = "/testbed"
_AGENT_USER = "swe-agent"
_HIDDEN_GIT_DIR = "/opt/agl_tmp"
_EDITOR_SOURCE = Path(__file__).with_name("editor_runner.py").read_bytes()
_TASK_PATH = (
    "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
_STATUS_RE = re.compile(
    r"(?:^|\s)(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)|"
    r"(\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)(?:\s|$)"
)
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_UNITTEST_NODE_RE = re.compile(r"([A-Za-z_]\w*) \(([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\)")
_CREATE_LIMITERS: weakref.WeakKeyDictionary[Any, tuple[int, asyncio.Semaphore]] = weakref.WeakKeyDictionary()
logger = logging.getLogger(__name__)


# Agent sandbox lifecycle: create, act, capture the patch, evaluate, close.
class E2BEpisode:
    """Own one E2B sandbox and expose the async SWE episode contract.

    The sandbox contains only the agent-visible task image. Patch capture seals
    the episode, so no agent command can run after trusted Git inspection starts.
    """

    workdir = _TESTBED

    def __init__(
        self,
        sandbox: Any,
        *,
        task_id: str,
        template_name: str | None = None,
        image_name: str | None = None,
        checkout_ref: str | None = None,
        allow_git: bool = False,
        timeout_seconds: float = 3600,
        sandbox_factory: Callable[..., Any] | None = None,
        run_id: str | None = None,
        create_concurrency: int = 4,
        create_max_retries: int = 4,
        create_retry_base_seconds: float = 1.0,
        create_limiter: asyncio.Semaphore | None = None,
        create_permit_held: bool = False,
        sandbox_started_at: float | None = None,
        max_output_chars: int = 1_000_000,
        editor_output_chars: int = 20_000,
    ) -> None:
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("task_id must be a nonempty string")
        if max_output_chars <= 0 or editor_output_chars <= 0:
            raise ValueError("output limits must be positive")
        episode_id = getattr(sandbox, "sandbox_id", None)
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("E2B sandbox is missing sandbox_id")

        self.sandbox = sandbox
        self.task_id = task_id
        self.template_name = template_name
        self.image_name = image_name
        self.checkout_ref = checkout_ref
        self.allow_git = allow_git
        self.timeout_seconds = timeout_seconds
        self._sandbox_factory = sandbox_factory
        self.run_id = run_id
        self.create_concurrency = create_concurrency
        self.create_max_retries = create_max_retries
        self.create_retry_base_seconds = create_retry_base_seconds
        self._create_limiter = create_limiter
        self._create_permit_held = create_permit_held
        self._sandbox_started_at = sandbox_started_at if sandbox_started_at is not None else time.monotonic()
        self.episode_id = episode_id
        self.create_duration_seconds: float | None = None
        self.create_attempts = 1
        self.create_retries = 0
        self.create_limiter_wait_seconds = 0.0
        self.create_rate_limit_wait_seconds = 0.0
        self.sandbox_lifetime_seconds: float | None = None
        self.teardown_duration_seconds: float | None = None
        self.max_output_chars = max_output_chars
        self._editor = AsyncE2BEditor(sandbox, max_output_chars=editor_output_chars, workdir=self.workdir)
        self._baseline_untracked: frozenset[str] = frozenset()
        self._submodules: dict[str, str] = {}
        self._baseline_submodule_untracked: dict[str, frozenset[str]] = {}
        self._git_hidden = False
        self._captured_patch: str | None = None
        self._evaluation_result: dict[str, Any] | None = None
        self._sealed = False
        self._closed = False

    @classmethod
    async def create(
        cls,
        *,
        template_name: str,
        task_id: str,
        image_name: str | None = None,
        checkout_ref: str | None = None,
        allow_git: bool = False,
        timeout_seconds: float = 3600,
        max_output_chars: int = 1_000_000,
        sandbox_factory: Callable[..., Any] | None = None,
        run_id: str | None = None,
        create_concurrency: int = 4,
        create_max_retries: int = 4,
        create_retry_base_seconds: float = 1.0,
        create_limiter: asyncio.Semaphore | None = None,
    ) -> E2BEpisode:
        """Create a network-isolated sandbox and validate its repository state."""
        if not isinstance(template_name, str) or not template_name:
            raise ValueError("template_name must be a nonempty string")
        if not _positive_number(timeout_seconds):
            raise ValueError("timeout_seconds must be positive")
        if checkout_ref is not None and (not isinstance(checkout_ref, str) or not checkout_ref):
            raise ValueError("checkout_ref must be a nonempty string or None")
        if run_id is not None and (not isinstance(run_id, str) or not run_id):
            raise ValueError("run_id must be a nonempty string or None")
        if not isinstance(create_concurrency, int) or isinstance(create_concurrency, bool) or create_concurrency <= 0:
            raise ValueError("create_concurrency must be a positive integer")
        if not isinstance(create_max_retries, int) or isinstance(create_max_retries, bool) or create_max_retries < 0:
            raise ValueError("create_max_retries must be a non-negative integer")
        if not _positive_number(create_retry_base_seconds):
            raise ValueError("create_retry_base_seconds must be positive")

        if sandbox_factory is None:
            from e2b import AsyncSandbox

            # The SDK reads E2B_API_KEY in this rollout process. We never copy
            # the credential into sandbox metadata, files, or command envs.
            sandbox_factory = AsyncSandbox.create

        metadata = {"swe-agent": "rollout", "task-id": task_id}
        if run_id is not None:
            metadata["run-id"] = run_id
        limiter = create_limiter or _shared_create_limiter(create_concurrency)
        started_at = time.monotonic()
        try:
            sandbox, create_stats, sandbox_started_at = await _create_e2b_sandbox(
                sandbox_factory,
                template_name,
                {
                    "timeout": math.ceil(timeout_seconds),
                    "metadata": metadata,
                    "allow_internet_access": False,
                },
                limiter=limiter,
                max_retries=create_max_retries,
                retry_base_seconds=create_retry_base_seconds,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise _e2b_error(error, "create") from error

        episode = cls(
            sandbox,
            task_id=task_id,
            template_name=template_name,
            image_name=image_name,
            checkout_ref=checkout_ref,
            allow_git=allow_git,
            timeout_seconds=timeout_seconds,
            sandbox_factory=sandbox_factory,
            run_id=run_id,
            create_concurrency=create_concurrency,
            create_max_retries=create_max_retries,
            create_retry_base_seconds=create_retry_base_seconds,
            create_limiter=limiter,
            create_permit_held=True,
            sandbox_started_at=sandbox_started_at,
            max_output_chars=max_output_chars,
        )
        episode.create_attempts = create_stats["attempts"]
        episode.create_retries = create_stats["retries"]
        episode.create_limiter_wait_seconds = create_stats["limiter_wait_seconds"]
        episode.create_rate_limit_wait_seconds = create_stats["rate_limit_wait_seconds"]
        try:
            await episode._initialize(checkout_ref)
        except BaseException:
            try:
                await _finish_cleanup(episode.close())
            except Exception:
                logger.warning("E2B cleanup after initialization failure also failed", exc_info=True)
            raise
        episode.create_duration_seconds = time.monotonic() - started_at
        return episode

    async def __aenter__(self) -> E2BEpisode:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        await self.close()

    async def execute(self, command: str, timeout_seconds: float) -> dict[str, Any]:
        if not _positive_number(timeout_seconds):
            raise ValueError("timeout_seconds must be positive")
        self._ensure_agent_active()
        started_at = time.monotonic()
        if not isinstance(command, str) or not command.strip():
            message = "command must be a nonempty string."
        elif "\x00" in command:
            message = "command contains a NUL byte. Use an escaped \\x00 in your script instead."
        else:
            try:
                command.encode("utf-8")
                return await self._run_command(command, timeout_seconds)
            except UnicodeEncodeError:
                message = "command must contain valid UTF-8 text."
            except SweAgentServiceError as error:
                # Only the model action boundary can recover a rejected command.
                # Internal Git and evaluator requests keep their service errors.
                if error.kind != "e2b_invalid_request":
                    raise
                message = str(error)
        stderr, truncated = _truncate(message, self.max_output_chars)
        return {
            "stdout": "",
            "stderr": stderr,
            "return_code": 1,
            "timed_out": False,
            "duration_seconds": time.monotonic() - started_at,
            "output_truncated": truncated,
        }

    async def edit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self._ensure_agent_active()
        started_at = time.monotonic()
        try:
            output = await self._editor.run(arguments)
            error = False
        except EditorError as editor_error:
            output = str(editor_error)
            error = True
        except UnicodeEncodeError:
            output = "Editor paths and text must contain valid UTF-8 text."
            error = True
        except asyncio.CancelledError:
            raise
        except Exception as operation_error:
            # The helper returns file/action errors as structured feedback.
            # A failed helper process itself remains an environment failure.
            service_error = _e2b_error(operation_error, "edit")
            if service_error.kind != "e2b_invalid_request":
                raise service_error from operation_error
            output = str(service_error)
            error = True
        output, _ = _truncate(output, self._editor.max_output_chars)
        return {
            "output": output,
            "error": error,
            "duration_seconds": time.monotonic() - started_at,
        }

    async def capture_patch(self) -> dict[str, Any]:
        self._ensure_open()
        started_at = time.monotonic()
        if self._captured_patch is None:
            # Sealing before Git inspection prevents a racing agent operation
            # from changing what is submitted after patch capture begins.
            self._sealed = True
            patches = []
            # Parent Git cannot follow submodule .git pointers after isolation.
            # Diff each initialized submodule through its private Git directory,
            # using repository-relative prefixes so one patch applies freshly.
            for submodule in [None, *self._submodules]:
                current = await self._list_untracked(submodule=submodule)
                baseline = (
                    self._baseline_untracked if submodule is None else self._baseline_submodule_untracked[submodule]
                )
                new_paths = sorted(current - baseline)
                if new_paths:
                    # Generated trees can exceed the shell's argument limit. Bound the
                    # quoted byte count; filenames must still be treated literally.
                    batches = [[]]
                    batch_bytes = 0
                    for path in new_paths:
                        path_bytes = len(shlex.quote(path).encode("utf-8")) + 1
                        if batches[-1] and batch_bytes + path_bytes > 32768:
                            batches.append([])
                            batch_bytes = 0
                        batches[-1].append(path)
                        batch_bytes += path_bytes
                    for batch in batches:
                        added = await self._run_git(
                            ["--literal-pathspecs", "add", "-N", "--", *batch],
                            120,
                            submodule=submodule,
                        )
                        _require_git_success(added, "mark new files for patch capture")
                prefix = f"{submodule}/" if submodule is not None else ""
                diff = await self._run_git(
                    [
                        "diff",
                        "--binary",
                        "--no-ext-diff",
                        "--no-textconv",
                        "--ignore-submodules=all",
                        f"--src-prefix=a/{prefix}",
                        f"--dst-prefix=b/{prefix}",
                        "HEAD",
                        "--",
                        ".",
                    ],
                    120,
                    submodule=submodule,
                )
                _require_git_success(diff, "capture patch")
                patches.append(_quote_diff_headers(diff["stdout"]))
                if diff["output_truncated"] or sum(map(len, patches)) > self.max_output_chars:
                    raise SweAgentServiceError(
                        "patch_too_large",
                        f"Captured patch exceeds {self.max_output_chars} characters.",
                        retryable=False,
                    )
            self._captured_patch = "".join(patches)
        return {
            "patch": self._captured_patch,
            "duration_seconds": time.monotonic() - started_at,
        }

    async def evaluate(
        self,
        task_metadata: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Grade the captured patch in a new sandbox the agent never controlled."""
        if not isinstance(task_metadata, dict):
            raise ValueError("task_metadata must be a dictionary")
        if not _positive_number(timeout_seconds):
            raise ValueError("timeout_seconds must be positive")
        if self._evaluation_result is not None:
            return dict(self._evaluation_result)
        if self._captured_patch is None or not self._sealed:
            raise SweAgentServiceError(
                "patch_not_captured",
                "Capture and seal the agent patch before evaluation.",
                retryable=False,
            )
        if not self.template_name or self._sandbox_factory is None:
            raise SweAgentServiceError(
                "evaluator_configuration_missing",
                "The E2B episode is missing its template or sandbox factory.",
                retryable=False,
            )

        started_at = time.monotonic()
        agent_sandbox_id = self.episode_id

        evaluator = None
        evaluator_sandbox = None
        evaluation: dict[str, Any] | None = None
        create_duration = 0.0
        teardown_duration = 0.0
        evaluator_lifetime = 0.0
        evaluator_limiter = self._create_limiter or _shared_create_limiter(self.create_concurrency)
        # A trajectory keeps its live-sandbox slot while switching to a fresh
        # evaluator. Releasing it here would put grading behind queued agents.
        evaluator_permit_held = self._create_permit_held
        self._create_permit_held = False
        try:
            # The replacement cannot start until the agent VM is confirmed dead.
            # This scope also disposes of the transferred slot if close fails.
            await self.close()
            create_started_at = time.monotonic()
            metadata = {"swe-agent": "evaluator", "task-id": self.task_id}
            if self.run_id is not None:
                metadata["run-id"] = self.run_id
            try:
                # The create helper owns release/quarantine on failure. On
                # success its returned VM transfers that ownership back here.
                already_acquired = evaluator_permit_held
                evaluator_permit_held = False
                evaluator_sandbox, evaluator_create_stats, evaluator_started_at = await _create_e2b_sandbox(
                    self._sandbox_factory,
                    self.template_name,
                    {
                        "timeout": math.ceil(self.timeout_seconds),
                        "metadata": metadata,
                        "allow_internet_access": False,
                    },
                    limiter=evaluator_limiter,
                    max_retries=self.create_max_retries,
                    retry_base_seconds=self.create_retry_base_seconds,
                    already_acquired=already_acquired,
                )
                evaluator_permit_held = True
            except asyncio.CancelledError:
                raise
            except Exception as error:
                raise _e2b_error(error, "evaluator create") from error
            create_duration = time.monotonic() - create_started_at
            evaluator = _FreshE2BEvaluator(
                evaluator_sandbox,
                task_id=self.task_id,
                image_name=self.image_name,
                checkout_ref=self.checkout_ref,
                patch=self._captured_patch,
                max_output_chars=self.max_output_chars,
            )
            await evaluator.initialize()
            patch_duration = await evaluator.apply_patch()
            evaluation = await evaluator.evaluate(task_metadata, timeout_seconds)
            evaluation.update(
                {
                    "agent_sandbox_id": agent_sandbox_id,
                    "evaluator_sandbox_id": evaluator.sandbox_id,
                    "agent_closed_before_evaluator": True,
                    "agent_sandbox_lifetime_seconds": self.sandbox_lifetime_seconds,
                    "agent_teardown_duration_seconds": self.teardown_duration_seconds,
                    "evaluator_create_duration_seconds": create_duration,
                    "evaluator_create_attempts": evaluator_create_stats["attempts"],
                    "evaluator_create_retries": evaluator_create_stats["retries"],
                    "evaluator_create_limiter_wait_seconds": evaluator_create_stats["limiter_wait_seconds"],
                    "evaluator_create_rate_limit_wait_seconds": evaluator_create_stats["rate_limit_wait_seconds"],
                    "patch_apply_duration_seconds": patch_duration,
                    "test_duration_seconds": evaluation["duration_seconds"],
                }
            )
        finally:
            if evaluator is not None:
                teardown_started_at = time.monotonic()
                try:
                    await _finish_cleanup(evaluator.close())
                except Exception as error:
                    # Once tests produced an outcome, teardown failure does not
                    # change its reward. Keep capacity reserved until expiry.
                    logger.warning("E2B evaluator cleanup failed: %s", type(error).__name__)
                finally:
                    teardown_duration = time.monotonic() - teardown_started_at
                    evaluator_lifetime = time.monotonic() - evaluator_started_at
                    if evaluator._closed and evaluator_permit_held:
                        evaluator_limiter.release()
                        evaluator_permit_held = False
                    elif evaluator_permit_held:
                        asyncio.get_running_loop().call_later(self.timeout_seconds, evaluator_limiter.release)
                        evaluator_permit_held = False
            elif evaluator_sandbox is not None:
                try:
                    await _finish_cleanup(kill_sandbox(evaluator_sandbox))
                except BaseException:
                    if evaluator_permit_held:
                        asyncio.get_running_loop().call_later(self.timeout_seconds, evaluator_limiter.release)
                        evaluator_permit_held = False
                    raise
                else:
                    if evaluator_permit_held:
                        evaluator_limiter.release()
                        evaluator_permit_held = False
            elif evaluator_permit_held:
                # Agent teardown failed or was cancelled before replacement.
                # A confirmed kill frees the slot; an unknown VM keeps its lease.
                if self._closed:
                    evaluator_limiter.release()
                else:
                    asyncio.get_running_loop().call_later(self.timeout_seconds, evaluator_limiter.release)
                evaluator_permit_held = False

        assert evaluation is not None
        evaluation["evaluator_teardown_duration_seconds"] = teardown_duration
        evaluation["evaluator_sandbox_lifetime_seconds"] = evaluator_lifetime
        evaluation["billed_sandbox_seconds"] = float(self.sandbox_lifetime_seconds or 0.0) + evaluator_lifetime
        evaluation["duration_seconds"] = time.monotonic() - started_at
        self._evaluation_result = evaluation
        return dict(evaluation)

    async def close(self) -> None:
        if self._closed:
            return
        teardown_started_at = time.monotonic()
        try:
            await _finish_cleanup(kill_sandbox(self.sandbox))
        except asyncio.CancelledError:
            # _finish_cleanup waits for kill before propagating cancellation.
            self._closed = True
            raise
        except Exception as error:
            if type(error).__name__ == "SandboxNotFoundException":
                self._closed = True
            else:
                if self._create_permit_held:
                    # An unreachable VM may still run. Reserve its slot for a
                    # full lease, then release it once expiry is certain. This
                    # one-shot timer never renews or changes the E2B timeout.
                    asyncio.get_running_loop().call_later(self.timeout_seconds, self._create_limiter.release)
                    self._create_permit_held = False
                raise _e2b_error(error, "close") from error
        else:
            self._closed = True
        finally:
            self.teardown_duration_seconds = time.monotonic() - teardown_started_at
            self.sandbox_lifetime_seconds = time.monotonic() - self._sandbox_started_at
            if self._closed and self._create_permit_held:
                assert self._create_limiter is not None
                self._create_limiter.release()
                self._create_permit_held = False

    async def _initialize(self, checkout_ref: str | None) -> None:
        head = await self._run_git(["rev-parse", "--verify", "HEAD^{commit}"], 30)
        _require_git_success(
            head,
            f"validate {self.workdir} repository",
            kind="sandbox_initialization_failed",
        )
        if checkout_ref is not None:
            checkout = await self._run_git(["checkout", "--force", checkout_ref], 120)
            _require_git_success(
                checkout,
                f"checkout task ref {checkout_ref}",
                kind="instance_checkout_failed",
            )
        # Keep ignored template artifacts in the baseline too: a later change
        # to .gitignore must not turn existing caches into agent-created files.
        self._baseline_untracked = await self._list_untracked(include_ignored=True)

        # Install the editor once, outside model-writable paths. Each subsequent
        # action performs validation, file access and clipping in one command.
        try:
            editor_source = _EDITOR_SOURCE.replace(b"/testbed", self.workdir.encode())
            await self.sandbox.files.write(RUNNER_PATH, editor_source, user="root")
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise _e2b_error(error, "editor install") from error

        # Images can contain solution-bearing commits. Keep the original Git
        # database root-only and run model commands as an unprivileged user.
        # Trusted patch capture uses the private Git directory explicitly.
        isolate = await self._run_command(
            "set -eu\n"
            f"test -d {self.workdir}/.git && test ! -L {self.workdir}/.git\n"
            f"test ! -e {_HIDDEN_GIT_DIR}\n"
            # Absorb old-form submodule histories before hiding the parent.
            # NUL-separated paths are captured before their pointers go stale.
            "git -c safe.directory='*' submodule absorbgitdirs\n"
            "git -c safe.directory='*' submodule foreach --quiet --recursive "
            '\'printf "%s\\0%s\\0" "$PWD" "$(git rev-parse --absolute-git-dir)"\'\n'
            f"id -u {_AGENT_USER} >/dev/null 2>&1 || useradd -m -s /bin/bash {_AGENT_USER}\n"
            f'test "$(id -u {_AGENT_USER})" -ne 0\n'
            f"mv {self.workdir}/.git {_HIDDEN_GIT_DIR}\n"
            f"chown root:root {_HIDDEN_GIT_DIR}\n"
            f"chmod 700 {_HIDDEN_GIT_DIR}\n"
            f"chown root:root {RUNNER_PATH}\n"
            f"chmod 444 {RUNNER_PATH}\n"
            f"chown -hR {_AGENT_USER}:{_AGENT_USER} {self.workdir}",
            180,
            internal=True,
        )
        _require_git_success(isolate, "isolate agent Git history", kind="sandbox_initialization_failed")
        self._git_hidden = True
        paths = isolate["stdout"].rstrip("\x00").split("\x00") if isolate["stdout"] else []
        if isolate["output_truncated"] or len(paths) % 2:
            raise SweAgentServiceError(
                "sandbox_initialization_failed", "Incomplete submodule inventory.", retryable=False
            )
        for work_tree, git_dir in zip(paths[::2], paths[1::2], strict=False):
            # The trusted inventory must stay inside this sandbox's repository.
            # Never follow model-controlled .git pointers during patch capture.
            try:
                relative_tree = PurePosixPath(work_tree).relative_to(self.workdir)
                relative_git = PurePosixPath(git_dir).relative_to(f"{self.workdir}/.git")
                if (
                    not relative_tree.parts
                    or not relative_git.parts
                    or ".." in (*relative_tree.parts, *relative_git.parts)
                ):
                    raise ValueError("Submodule paths must stay inside the repository.")
            except ValueError as error:
                raise SweAgentServiceError("sandbox_initialization_failed", str(error), retryable=False) from error
            path = relative_tree.as_posix()
            self._submodules[path] = str(PurePosixPath(_HIDDEN_GIT_DIR) / relative_git)
            self._baseline_submodule_untracked[path] = await self._list_untracked(
                submodule=path,
                include_ignored=True,
            )

        if self.allow_git:
            # Agent commits must not change the baseline used by patch capture.
            # Give each repository its own shallow copy, including submodules.
            repositories = [(self.workdir, _HIDDEN_GIT_DIR)]
            repositories += [(f"{self.workdir}/{path}", git_dir) for path, git_dir in self._submodules.items()]
            for work_tree, git_dir in repositories:
                result = await self._run_command(
                    _single_commit_git_commands(work_tree, git_dir)
                    + f"chown -hR {_AGENT_USER}:{_AGENT_USER} {shlex.quote(work_tree + '/.git')}\n",
                    180,
                    internal=True,
                )
                _require_git_success(result, "expose task Git checkout", kind="sandbox_initialization_failed")

    async def _list_untracked(self, *, submodule: str | None = None, include_ignored: bool = False) -> frozenset[str]:
        arguments = ["ls-files", "--others", "-z"]
        if not include_ignored:
            arguments.append("--exclude-standard")
        result = await self._run_git(
            arguments,
            60,
            submodule=submodule,
        )
        _require_git_success(result, "list untracked files")
        if result["output_truncated"]:
            raise SweAgentServiceError(
                "patch_capture_failed",
                "The untracked-file list exceeded the sandbox output limit.",
                retryable=False,
            )
        return frozenset(path for path in result["stdout"].split("\x00") if path)

    async def _run_git(
        self, arguments: list[str], timeout_seconds: float, *, submodule: str | None = None
    ) -> dict[str, Any]:
        location = f"--git-dir={_HIDDEN_GIT_DIR} --work-tree={self.workdir} " if self._git_hidden else ""
        if submodule is not None:
            location = (
                f"--git-dir={shlex.quote(self._submodules[submodule])} "
                f"--work-tree={shlex.quote(f'{self.workdir}/{submodule}')} "
            )
        command = "git -c safe.directory=* " + location + " ".join(shlex.quote(part) for part in arguments)
        return await self._run_command(command, timeout_seconds, internal=True)

    async def _run_command(
        self,
        command: str,
        timeout_seconds: float,
        *,
        internal: bool = False,
    ) -> dict[str, Any]:
        self._ensure_open()
        if self._sealed and not internal:
            raise SweAgentServiceError(
                "episode_sealed",
                "The agent phase is sealed after patch capture.",
                retryable=False,
            )
        return await _run_e2b_command(
            self.sandbox,
            command,
            timeout_seconds,
            self.max_output_chars,
            operation="execute",
            user="root" if internal else _AGENT_USER,
            cwd=self.workdir,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise SweAgentServiceError(
                "episode_closed",
                "The E2B sandbox is already closed.",
                retryable=False,
            )

    def _ensure_agent_active(self) -> None:
        self._ensure_open()
        if self._sealed:
            raise SweAgentServiceError(
                "episode_sealed",
                "The agent phase is sealed after patch capture.",
                retryable=False,
            )


# A fresh evaluator applies the patch and runs trusted tests after the agent exits.
class _FreshE2BEvaluator:
    """Apply one sealed patch and grade it in a fresh E2B sandbox."""

    def __init__(
        self,
        sandbox: Any,
        *,
        task_id: str,
        image_name: str | None,
        checkout_ref: str | None,
        patch: str,
        max_output_chars: int,
    ) -> None:
        sandbox_id = getattr(sandbox, "sandbox_id", None)
        if not isinstance(sandbox_id, str) or not sandbox_id:
            raise SweAgentServiceError(
                "evaluator_initialization_failed",
                "The evaluator sandbox is missing sandbox_id.",
                retryable=True,
            )
        self.sandbox = sandbox
        self.sandbox_id = sandbox_id
        self.task_id = task_id
        self.image_name = image_name
        self.checkout_ref = checkout_ref
        self.patch = patch
        self.max_output_chars = max_output_chars
        self._closed = False

    async def initialize(self) -> None:
        head = await self._run_git(["rev-parse", "--verify", "HEAD^{commit}"], 30)
        _require_git_success(
            head,
            "validate fresh evaluator repository",
            kind="evaluator_initialization_failed",
        )
        if self.checkout_ref is not None:
            checkout = await self._run_git(["checkout", "--force", self.checkout_ref], 120)
            _require_git_success(
                checkout,
                f"checkout evaluator task ref {self.checkout_ref}",
                kind="instance_checkout_failed",
            )

    async def apply_patch(self) -> float:
        started_at = time.monotonic()
        if not self.patch:
            return time.monotonic() - started_at
        try:
            await self.sandbox.files.write(
                "/tmp/swe-agent.patch",
                self.patch.encode("utf-8"),
                user="root",
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise _e2b_error(error, "patch upload") from error

        result = await self._run_git(
            ["apply", "--binary", "--whitespace=nowarn", "/tmp/swe-agent.patch"],
            120,
        )
        _require_git_success(result, "apply captured patch", kind="patch_apply_failed")
        return time.monotonic() - started_at

    async def evaluate(
        self,
        task_metadata: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        dataset_type = task_metadata.get("dataset_type", "swe_smith")
        if dataset_type == "swebench_verified":
            return await self._evaluate_swebench_verified(task_metadata, timeout_seconds)
        if dataset_type != "swe_smith":
            raise SweAgentServiceError(
                "unknown_evaluator",
                f"Unknown evaluator type {dataset_type!r}.",
                retryable=False,
            )
        return await self._evaluate_swe_smith(task_metadata, timeout_seconds)

    async def close(self) -> None:
        if self._closed:
            return
        try:
            await kill_sandbox(self.sandbox)
        except Exception as error:
            if type(error).__name__ != "SandboxNotFoundException":
                raise _e2b_error(error, "evaluator close") from error
        self._closed = True

    async def _evaluate_swe_smith(
        self,
        task_metadata: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Preserve Agent Lightning v1.0.0's F2P-only reward contract."""
        started_at = time.monotonic()
        metadata_instance = task_metadata.get("instance_id", self.task_id)
        if metadata_instance != self.task_id:
            raise SweAgentServiceError(
                "evaluation_metadata_mismatch",
                "Evaluation instance_id does not match the episode branch.",
                retryable=False,
            )
        # Mido's released IDs are relative to the upstream runner's /dev cwd.
        # Translate that one documented prefix, never allow arbitrary traversal.
        test_metadata = dict(task_metadata)
        if self.image_name == "jyangballin/swesmith.x86_64.mido_1776_mido.a0158ff9":
            for field in ("FAIL_TO_PASS", "PASS_TO_PASS"):
                nodes = test_metadata.get(field, [])
                if isinstance(nodes, list) and all(isinstance(node, str) for node in nodes):
                    test_metadata[field] = [node.removeprefix("../dev/") for node in nodes]
        fail_to_pass = _test_nodes(test_metadata.get("FAIL_TO_PASS", []), "FAIL_TO_PASS")
        pass_to_pass = _test_nodes(test_metadata.get("PASS_TO_PASS", []), "PASS_TO_PASS")
        if not fail_to_pass:
            return asdict(
                EvaluationResult(
                    reward=0.0,
                    resolved=False,
                    passed_tests=0,
                    total_tests=0,
                    output="",
                    timed_out=False,
                    duration_seconds=time.monotonic() - started_at,
                    summary="no FAIL_TO_PASS in eval meta",
                )
            )

        f2p_files = sorted({_test_file(node) for node in fail_to_pass})
        pass_to_pass = [node for node in pass_to_pass if _test_file(node) in f2p_files]
        restore = await self._run_git(["checkout", "HEAD~1", "--", *f2p_files], 120)
        _require_git_success(restore, "restore F2P tests", kind="f2p_restore_failed")
        await self._isolate_for_tests()

        xdist_probe = await self._run_command(
            "python -c 'import xdist'",
            min(30.0, timeout_seconds),
            user=_AGENT_USER,
        )
        xdist_args = ["-n1"] if xdist_probe["return_code"] == 0 else ["-p", "no:xdist"]
        # Hydra's optional timing plugin assumes cacheprovider is enabled.
        # It is unrelated to test outcomes and incompatible with this runner.
        plugin_args = (
            ["-p", "no:snail"]
            if self.image_name == ("jyangballin/swesmith.x86_64.facebookresearch_1776_hydra.0f03eb60")
            else []
        )
        # Keep each test module contiguous. Returning to a module after a
        # different file changes import/event-loop state can break good code.
        # Stable sorting preserves the selected tests and their within-file order.
        test_nodes = sorted([*fail_to_pass, *pass_to_pass], key=_test_file)
        arguments = [
            "python",
            "-m",
            "pytest",
            "-rA",
            "-p",
            "no:cacheprovider",
            *plugin_args,
            *xdist_args,
            *test_nodes,
        ]
        # Tornado supplies unittest display IDs and requires its own runner.
        # Preserve the original IDs for scoring; only translate invocation IDs.
        if all(_UNITTEST_NODE_RE.fullmatch(node) for node in test_nodes):
            arguments = ["python", "-m", "tornado.test", "--verbose"]
            for node in test_nodes:
                match = _UNITTEST_NODE_RE.fullmatch(node)
                arguments.append(f"{match[2]}.{match[1]}")
        elif any(_UNITTEST_NODE_RE.fullmatch(node) for node in test_nodes):
            raise SweAgentServiceError("invalid_evaluation_metadata", "Mixed pytest/unittest IDs.", retryable=False)
        elif self.image_name == "jyangballin/swesmith.x86_64.mido_1776_mido.a0158ff9":
            arguments[3:3] = ["-c", "/dev/null", "--rootdir=/testbed"]
        result = await self._run_command(
            " ".join(shlex.quote(part) for part in arguments),
            timeout_seconds,
            user=_AGENT_USER,
        )

        raw_output = result["stdout"] + result["stderr"]
        statuses = _parse_test_statuses(raw_output)
        f2p_passed = sum(statuses.get(node) in ("PASSED", "XFAIL") for node in fail_to_pass)
        p2p_passed = sum(statuses.get(node) in ("PASSED", "XFAIL") for node in pass_to_pass)
        resolved = (
            not result["timed_out"]
            and result["return_code"] == 0
            and f2p_passed == len(fail_to_pass)
            and p2p_passed == len(pass_to_pass)
        )
        prefix = f"EVAL TIMEOUT after {timeout_seconds:g}s — " if result["timed_out"] else ""
        summary = (
            f"{prefix}FAIL_TO_PASS {f2p_passed}/{len(fail_to_pass)} passed, "
            f"PASS_TO_PASS {p2p_passed}/{len(pass_to_pass)} ok (f2p_only)"
        )
        output, output_truncated = _truncate(raw_output, self.max_output_chars)
        return asdict(
            EvaluationResult(
                reward=1.0 if resolved else 0.0,
                resolved=resolved,
                passed_tests=f2p_passed + p2p_passed,
                total_tests=len(fail_to_pass) + len(pass_to_pass),
                output=output,
                timed_out=result["timed_out"],
                duration_seconds=time.monotonic() - started_at,
                output_truncated=result["output_truncated"] or output_truncated,
                fail_to_pass_passed=f2p_passed,
                fail_to_pass_total=len(fail_to_pass),
                pass_to_pass_passed=p2p_passed,
                pass_to_pass_total=len(pass_to_pass),
                summary=summary,
            )
        )

    async def _evaluate_swebench_verified(
        self,
        task_metadata: dict[str, Any],
        timeout_seconds: float,
    ) -> dict[str, Any]:
        """Run the pinned official SWE-bench test script and canonical grader."""
        started_at = time.monotonic()
        if task_metadata.get("instance_id") != self.task_id:
            raise SweAgentServiceError(
                "evaluation_metadata_mismatch",
                "Evaluation instance_id does not match the episode task.",
                retryable=False,
            )
        if (
            task_metadata.get("dataset_revision") != DATASET_REVISION
            or task_metadata.get("harness_revision") != HARNESS_REVISION
        ):
            raise SweAgentServiceError(
                "evaluation_revision_mismatch",
                "The SWE-bench dataset or harness revision is not the pinned revision.",
                retryable=False,
            )

        try:
            from swebench.harness.grading import get_eval_report
            from swebench.harness.utils import make_test_spec
        except ImportError as error:
            raise SweAgentServiceError(
                "evaluator_dependency_missing",
                "swebench==5.0.2 is required on the rollout host.",
                retryable=False,
            ) from error

        try:
            test_spec = await asyncio.to_thread(make_test_spec, task_metadata)
        except Exception as error:
            raise SweAgentServiceError(
                "invalid_evaluation_metadata",
                f"Cannot construct the official SWE-bench test spec: {error}",
                retryable=False,
            ) from error
        expected_image = self.image_name or task_metadata.get("image")
        if test_spec.instance_id != self.task_id or test_spec.image != expected_image:
            raise SweAgentServiceError(
                "evaluation_metadata_mismatch",
                "The SWE-bench test spec does not match the episode image and task.",
                retryable=False,
            )
        if test_spec.image_assets:
            raise SweAgentServiceError(
                "unsupported_evaluation_assets",
                "This evaluator does not support SWE-bench binary image assets.",
                retryable=False,
            )

        # The official script uses Git during setup and cleanup. Give it only a
        # shallow copy of the current baseline, not the image's other history.
        # All script execution (including package setup) is then unprivileged.
        await self._isolate_for_tests(keep_baseline_git=True)
        # A shallow baseline looks like a root commit: bare `git show` would
        # dump the whole repository and crowd the actual tests out of the log.
        # Change only that diagnostic; retain the canonical test commands.
        eval_script = test_spec.eval_script.replace("\ngit show\n", "\ngit show --no-patch\n")
        try:
            await self.sandbox.files.write(
                "/tmp/.swebench_eval.sh",
                eval_script.encode("utf-8"),
                user="root",
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise _e2b_error(error, "evaluator script upload") from error
        result = await self._run_command(
            "bash /tmp/.swebench_eval.sh 2>&1",
            timeout_seconds,
            user=_AGENT_USER,
        )
        raw_output = result["stdout"] + result["stderr"]

        # A timeout is a normal unsuccessful evaluation. Do not turn a partial
        # log that the canonical parser cannot consume into an infra failure.
        if result["timed_out"]:
            fail_to_pass = _metadata_test_list(task_metadata.get("FAIL_TO_PASS", []))
            pass_to_pass = _metadata_test_list(task_metadata.get("PASS_TO_PASS", []))
            output, output_truncated = _truncate(raw_output, self.max_output_chars)
            return asdict(
                EvaluationResult(
                    reward=0.0,
                    resolved=False,
                    passed_tests=0,
                    total_tests=len(fail_to_pass) + len(pass_to_pass),
                    output=output,
                    timed_out=True,
                    duration_seconds=time.monotonic() - started_at,
                    output_truncated=result["output_truncated"] or output_truncated,
                    fail_to_pass_total=len(fail_to_pass),
                    pass_to_pass_total=len(pass_to_pass),
                    summary=f"EVAL TIMEOUT after {timeout_seconds:g}s (official_swebench)",
                )
            )

        prediction = {
            "instance_id": self.task_id,
            "model_patch": self.patch,
            "model_name_or_path": "Qwen/Qwen3.5-4B",
        }
        try:
            with tempfile.TemporaryDirectory(prefix="swe-agent-e2b-") as directory:
                eval_log = Path(directory) / "test_output.txt"
                eval_log.write_text(raw_output, encoding="utf-8")
                report = await asyncio.to_thread(
                    get_eval_report,
                    test_spec,
                    prediction,
                    str(eval_log),
                    True,
                )
            details = report[self.task_id]
        except Exception as error:
            raise SweAgentServiceError(
                "evaluation_grading_failed",
                f"The official SWE-bench grader failed: {error}",
                retryable=False,
            ) from error
        if details.get("infra_failure"):
            raise SweAgentServiceError(
                "evaluation_infrastructure_failure",
                str(details.get("infra_failure_reason", "SWE-bench environment failure")),
                retryable=True,
            )

        statuses = details.get("tests_status") or {}
        f2p = statuses.get("FAIL_TO_PASS") or {"success": [], "failure": []}
        p2p = statuses.get("PASS_TO_PASS") or {"success": [], "failure": []}
        f2p_passed = len(f2p.get("success", []))
        f2p_total = f2p_passed + len(f2p.get("failure", []))
        p2p_passed = len(p2p.get("success", []))
        p2p_total = p2p_passed + len(p2p.get("failure", []))
        resolved = bool(details.get("resolved", False))
        summary = (
            f"FAIL_TO_PASS {f2p_passed}/{f2p_total} passed, "
            f"PASS_TO_PASS {p2p_passed}/{p2p_total} passed (official_swebench)"
        )
        output, output_truncated = _truncate(raw_output, self.max_output_chars)
        return asdict(
            EvaluationResult(
                reward=1.0 if resolved else 0.0,
                resolved=resolved,
                passed_tests=f2p_passed + p2p_passed,
                total_tests=f2p_total + p2p_total,
                output=output,
                timed_out=False,
                duration_seconds=time.monotonic() - started_at,
                output_truncated=result["output_truncated"] or output_truncated,
                fail_to_pass_passed=f2p_passed,
                fail_to_pass_total=f2p_total,
                pass_to_pass_passed=p2p_passed,
                pass_to_pass_total=p2p_total,
                summary=summary,
            )
        )

    async def _isolate_for_tests(self, *, keep_baseline_git: bool = False) -> None:
        """Keep solution-bearing history root-only before executing submitted code."""
        command = (
            "set -eu\n"
            "test -d /testbed/.git && test ! -L /testbed/.git\n"
            f"test ! -e {_HIDDEN_GIT_DIR}\n"
            "git -c safe.directory='*' submodule absorbgitdirs\n"
            f"id -u {_AGENT_USER} >/dev/null 2>&1 || useradd -m -s /bin/bash {_AGENT_USER}\n"
            f'test "$(id -u {_AGENT_USER})" -ne 0\n'
            f"mv /testbed/.git {_HIDDEN_GIT_DIR}\n"
            f"chown root:root {_HIDDEN_GIT_DIR}\n"
            f"chmod 700 {_HIDDEN_GIT_DIR}\n"
        )
        if keep_baseline_git:
            command += (
                "test ! -e /opt/agl_eval\n"
                f"baseline=$(git --git-dir={_HIDDEN_GIT_DIR} rev-parse HEAD)\n"
                "git init /opt/agl_eval\n"
                f'git -C /opt/agl_eval fetch --depth=1 file://{_HIDDEN_GIT_DIR} "$baseline"\n'
                'git -C /opt/agl_eval update-ref HEAD "$baseline"\n'
                "git -C /opt/agl_eval read-tree HEAD\n"
                "mv /opt/agl_eval/.git /testbed/.git\n"
                "rmdir /opt/agl_eval\n"
                # Canonical setup must not traverse now-private submodule Git.
                "git -C /testbed config diff.ignoreSubmodules all\n"
            )
        command += f"chown -hR {_AGENT_USER}:{_AGENT_USER} /testbed\n"
        if keep_baseline_git:
            # Canonical SWE-bench scripts may rebuild the editable package.
            command += f"chown -hR {_AGENT_USER}:{_AGENT_USER} /opt/miniconda3/envs/testbed\n"
        result = await self._run_command(command, 180)
        _require_git_success(result, "isolate evaluator Git history", kind="evaluator_isolation_failed")

    async def _run_git(self, arguments: list[str], timeout_seconds: float) -> dict[str, Any]:
        command = "git -c safe.directory=* " + " ".join(shlex.quote(part) for part in arguments)
        return await self._run_command(command, timeout_seconds)

    async def _run_command(self, command: str, timeout_seconds: float, *, user: str = "root") -> dict[str, Any]:
        return await _run_e2b_command(
            self.sandbox,
            command,
            timeout_seconds,
            self.max_output_chars,
            operation="evaluate",
            user=user,
        )


def _single_commit_git_commands(work_tree: str, git_dir: str) -> str:
    """Expose only the private baseline commit without changing working files."""
    git = f"git -c safe.directory='*' -C {shlex.quote(work_tree)}"
    return (
        "set -eu\n"
        # Relocated submodules have relative worktree paths that fetch cannot use.
        f"git config --file {shlex.quote(git_dir + '/config')} core.worktree {shlex.quote(work_tree)}\n"
        f"baseline=$(git -c safe.directory='*' --git-dir={shlex.quote(git_dir)} rev-parse HEAD)\n"
        # A submodule still has its old pointer; the parent's .git was moved.
        f"rm -f {shlex.quote(work_tree + '/.git')}\n"
        f"git init --quiet {shlex.quote(work_tree)}\n"
        f'{git} fetch --quiet --depth=1 {shlex.quote("file:///" + git_dir.lstrip("/"))} "$baseline"\n'
        f'{git} update-ref HEAD "$baseline"\n'
        f"{git} read-tree HEAD\n"
    )


# SDK command execution, admission limits, and result validation.
async def _run_e2b_command(
    sandbox: Any,
    command: str,
    timeout_seconds: float,
    max_output_chars: int,
    *,
    operation: str,
    user: str = "root",
    cwd: str = _TESTBED,
) -> dict[str, Any]:
    started_at = time.monotonic()
    # The SWE images keep task dependencies in the testbed Conda env.
    # Export after E2B's login shell starts so its profile cannot reset it.
    wrapped = (
        f"export PATH={shlex.quote(_TASK_PATH)}; "
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; "
        f"exec setsid /bin/bash -c {shlex.quote(command)}"
    )
    try:
        handle = await sandbox.commands.run(
            wrapped,
            background=True,
            cwd=cwd,
            user=user,
            timeout=0,
        )
    except asyncio.CancelledError:
        raise
    except Exception as error:
        raise _e2b_error(error, operation) from error

    timed_out = False
    try:
        deadline = asyncio.timeout(timeout_seconds)
        async with deadline:
            result = await handle.wait()
    except TimeoutError as error:
        if not deadline.expired():
            await _finish_cleanup(_stop_e2b_command(sandbox, handle))
            raise _e2b_error(error, operation) from error
        timed_out = True
        await _finish_cleanup(_stop_e2b_command(sandbox, handle))
        result = None
    except asyncio.CancelledError:
        await _finish_cleanup(_stop_e2b_command(sandbox, handle))
        raise
    except Exception as error:
        if _is_command_exit(error):
            result = error
        else:
            await _finish_cleanup(_stop_e2b_command(sandbox, handle))
            raise _e2b_error(error, operation) from error

    stdout = str(getattr(result, "stdout", getattr(handle, "stdout", "")))
    stderr = str(getattr(result, "stderr", getattr(handle, "stderr", "")))
    stdout, stdout_truncated = _truncate(stdout, max_output_chars)
    stderr, stderr_truncated = _truncate(stderr, max_output_chars)
    return {
        "stdout": stdout,
        "stderr": stderr,
        "return_code": 124 if timed_out else int(result.exit_code),
        "timed_out": timed_out,
        "duration_seconds": time.monotonic() - started_at,
        "output_truncated": stdout_truncated or stderr_truncated,
    }


async def _stop_e2b_command(sandbox: Any, handle: Any) -> None:
    # Each command starts a new process group. Kill the group first so child
    # processes cannot outlive a timed-out or cancelled tool call.
    pid = getattr(handle, "pid", None)
    if isinstance(pid, int) and pid > 0:
        try:
            await sandbox.commands.run(
                f"kill -KILL -- -{pid}",
                cwd="/",
                user="root",
                timeout=10,
            )
        except Exception:
            pass
    try:
        await handle.kill()
    except Exception:
        pass


async def _finish_cleanup(awaitable: Any) -> Any:
    """Finish teardown even if the surrounding rollout task is cancelled."""
    cleanup = asyncio.ensure_future(awaitable)
    try:
        return await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        await cleanup
        raise


def _shared_create_limiter(limit: int) -> asyncio.Semaphore:
    """Share one E2B admission limit across episodes in this event loop."""
    loop = asyncio.get_running_loop()
    configured = _CREATE_LIMITERS.get(loop)
    if configured is None:
        limiter = asyncio.Semaphore(limit)
        _CREATE_LIMITERS[loop] = (limit, limiter)
        return limiter
    configured_limit, limiter = configured
    if configured_limit != limit:
        raise ValueError("E2B create_concurrency must be consistent within one rollout process")
    return limiter


async def _create_e2b_sandbox(
    sandbox_factory: Callable[..., Any],
    template_name: str,
    create_kwargs: dict[str, Any],
    *,
    limiter: asyncio.Semaphore,
    max_retries: int,
    retry_base_seconds: float,
    already_acquired: bool = False,
) -> tuple[Any, dict[str, int | float], float]:
    """Create with a new slot, or a transferred slot whose previous VM is dead."""
    attempts = 0
    limiter_wait_seconds = 0.0
    rate_limit_wait_seconds = 0.0
    create_kwargs = dict(create_kwargs)
    create_kwargs["lifecycle"] = {"on_timeout": "kill"}
    metadata = dict(create_kwargs.get("metadata", {}))
    metadata["attempt-id"] = uuid4().hex
    if os.environ.get("SLURM_JOB_ID"):
        metadata["slurm-job-id"] = os.environ["SLURM_JOB_ID"]
    create_kwargs["metadata"] = metadata

    if not already_acquired:
        wait_started_at = time.monotonic()
        await limiter.acquire()
        limiter_wait_seconds = time.monotonic() - wait_started_at
    try:
        for retry_index in range(max_retries + 1):
            try:
                attempts += 1
                sandbox_started_at = time.monotonic()
                sandbox = await sandbox_factory(template_name, **create_kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                if type(error).__name__ != "RateLimitException" or retry_index >= max_retries:
                    raise
            else:
                # The caller owns this permit until the sandbox is killed.
                # This bounds active sandboxes, not just concurrent HTTP calls.
                return (
                    sandbox,
                    {
                        "attempts": attempts,
                        "retries": attempts - 1,
                        "limiter_wait_seconds": limiter_wait_seconds,
                        "rate_limit_wait_seconds": rate_limit_wait_seconds,
                    },
                    sandbox_started_at,
                )

            # The create call is safe to retry because a rate-limit response
            # did not allocate a sandbox. Agent commands and patch application
            # never enter this retry path.
            delay_seconds = min(retry_base_seconds * 2 ** min(retry_index, 10), 10.0)
            rate_limit_wait_seconds += delay_seconds
            await asyncio.sleep(delay_seconds)
    except BaseException as error:
        # A timed-out/cancelled create may have reached E2B without returning its
        # ID. Do not immediately reuse that possibly-live VM's capacity slot.
        ambiguous = isinstance(error, asyncio.CancelledError | TimeoutError | ConnectionError) or type(
            error
        ).__name__ in {
            "TimeoutException",
            "ReadTimeout",
            "ConnectTimeout",
            "RemoteProtocolError",
            "ReadError",
        }
        ambiguous |= type(error).__module__.startswith(("e2b", "httpx", "httpcore", "connectrpc")) and type(
            error
        ).__name__ not in {
            "RateLimitException",
            "AuthenticationException",
            "InvalidArgumentException",
            "TemplateException",
        }
        if ambiguous:
            asyncio.get_running_loop().call_later(create_kwargs["timeout"], limiter.release)
        else:
            limiter.release()
        raise

    raise AssertionError("unreachable E2B create retry state")


def _require_git_success(
    result: dict[str, Any],
    operation: str,
    *,
    kind: str = "patch_capture_failed",
) -> None:
    if not result["timed_out"] and result["return_code"] == 0:
        return
    reason = "timed out" if result["timed_out"] else f"exited with code {result['return_code']}"
    detail = (result["stdout"] + result["stderr"]).strip()
    if detail:
        reason += f": {detail[-2000:]}"
    raise SweAgentServiceError(
        kind,
        f"Could not {operation}: {reason}",
        retryable=False,
    )


def _test_nodes(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(node, str) and node for node in value):
        raise SweAgentServiceError(
            "invalid_evaluation_metadata",
            f"{name} must be a list of nonempty strings.",
            retryable=False,
        )
    for node in value:
        _test_file(node)
    return list(value)


def _test_file(node: str) -> str:
    match = _UNITTEST_NODE_RE.fullmatch(node)
    if match:
        return match[2].rsplit(".", 1)[0].replace(".", "/") + ".py"
    raw_path = node.split("::", 1)[0]
    path = PurePosixPath(raw_path)
    if path.is_absolute() or ".." in path.parts or not path.parts or raw_path.startswith("-"):
        raise SweAgentServiceError(
            "invalid_evaluation_metadata",
            f"Unsafe test node: {node!r}.",
            retryable=False,
        )
    return raw_path


def _metadata_test_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _quote_diff_headers(patch: str) -> str:
    """Disambiguate space-containing paths in Git's same-path binary diff headers."""

    # Binary patches lack +++/--- filenames. Git can misread an unquoted path
    # beginning with a space, even when its own diff command generated it.
    def quote(match: re.Match[str]) -> str:
        path = match[1]
        if " " not in path:
            return match[0]
        return "diff --git " + " ".join(json.dumps(prefix + path, ensure_ascii=False) for prefix in ("a/", "b/"))

    return re.sub(r"^diff --git a/(.+) b/\1$", quote, patch, flags=re.MULTILINE)


def _parse_test_statuses(output: str) -> dict[str, str]:
    # Some repository pytest configs force color even without a terminal.
    # Escape sequences otherwise hide statuses or become part of node IDs.
    output = _ANSI_RE.sub("", output)
    statuses = {}
    for line in output.splitlines():
        unittest_match = re.match(r"(.+ \(.+\)) \.\.\. (ok|skipped.*|FAIL|ERROR)$", line)
        if unittest_match and _UNITTEST_NODE_RE.fullmatch(unittest_match[1]):
            status = unittest_match[2]
            statuses[unittest_match[1]] = (
                "PASSED" if status == "ok" else ("SKIPPED" if status.startswith("skipped") else "FAILED")
            )
            continue
        if line.startswith(("FAIL: ", "ERROR: ")):
            node = line.split(": ", 1)[1]
            if _UNITTEST_NODE_RE.fullmatch(node):
                statuses[node] = "FAILED"
            continue
        match = _STATUS_RE.search(line)
        if match is None:
            continue
        if match.group(1):
            status, node = match.group(1), match.group(2)
        else:
            node, status = match.group(3), match.group(4)
        statuses[node] = status
    return statuses


def _is_command_exit(error: Exception) -> bool:
    return (
        isinstance(getattr(error, "stdout", None), str)
        and isinstance(getattr(error, "stderr", None), str)
        and isinstance(getattr(error, "exit_code", None), int)
    )


def _e2b_error(error: Exception, operation: str) -> SweAgentServiceError:
    if isinstance(error, SweAgentServiceError):
        return error
    error_name = type(error).__name__
    kinds = {
        "AuthenticationException": ("e2b_authentication_failed", False),
        "InvalidArgumentException": ("e2b_invalid_request", False),
        "RateLimitException": ("e2b_rate_limited", True),
        "SandboxNotFoundException": ("e2b_sandbox_not_found", True),
        "TimeoutException": ("e2b_timeout", True),
        "TimeoutError": ("e2b_timeout", True),
    }
    if (
        error_name not in kinds
        and not isinstance(error, OSError | ConnectionError)
        and not type(error).__module__.startswith(("e2b", "httpx", "httpcore", "connectrpc"))
    ):
        raise error
    kind, retryable = kinds.get(error_name, ("e2b_infrastructure_error", True))
    detail = str(error)
    api_key = os.environ.get("E2B_API_KEY")
    if api_key:
        detail = detail.replace(api_key, "[redacted]")
    return SweAgentServiceError(
        kind,
        f"E2B {operation} failed: {error_name}: {detail}",
        retryable=retryable,
    )


def _positive_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return False
    try:
        return math.isfinite(float(value)) and value > 0
    except OverflowError:
        return False


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    marker = "\n...[output truncated]...\n"
    payload = max(0, limit - len(marker))
    head = payload // 2
    tail = payload - head
    if limit <= len(marker):
        return marker[:limit], True
    return text[:head] + marker + text[-tail:], True


# Bounded teardown for individual episodes and Slurm job exit.
async def kill_sandbox(sandbox):
    """Kill is idempotent; retry transport failures without hanging teardown."""
    for attempt in range(3):
        try:
            return await asyncio.wait_for(sandbox.kill(), timeout=10)
        except Exception as error:
            if type(error).__name__ == "SandboxNotFoundException":
                return False
            if attempt == 2:
                raise
            await asyncio.sleep(1)


async def cleanup_job(sandbox_class, job_id, run_id):
    """Never touch another job/run, even if it uses the same task templates."""
    if not job_id or not run_id:
        raise ValueError("Both Slurm job ID and SWE run ID are required for cleanup.")
    killed = 0
    # A second pass catches creates that completed while Ray was shutting down.
    for sweep in range(2):
        paginator = sandbox_class.list()
        while paginator.has_next:
            for info in await paginator.next_items():
                metadata = info.metadata or {}
                if (
                    metadata.get("slurm-job-id") != job_id
                    or metadata.get("run-id") != run_id
                    or metadata.get("swe-agent") not in {"rollout", "evaluator"}
                ):
                    continue
                # Use the class API by ID; do not connect or extend the deadline.
                for attempt in range(3):
                    try:
                        killed += bool(await asyncio.wait_for(sandbox_class.kill(info.sandbox_id), timeout=10))
                        break
                    except Exception as error:
                        if type(error).__name__ == "SandboxNotFoundException":
                            break
                        if attempt == 2:
                            logger.warning(
                                "Could not kill job-owned sandbox %s: %s", info.sandbox_id, type(error).__name__
                            )
                        else:
                            await asyncio.sleep(1)
        if sweep == 0:
            await asyncio.sleep(2)
    return killed


if __name__ == "__main__":
    import sys

    if sys.argv[1:] != ["--cleanup-job"]:
        raise SystemExit("usage: python e2b_episode.py --cleanup-job")
    from e2b import AsyncSandbox

    print(
        "E2B job cleanup: killed",
        asyncio.run(
            cleanup_job(
                AsyncSandbox,
                os.environ["SLURM_JOB_ID"],
                os.environ["SLURM_JOB_NAME"],
            )
        ),
    )
