"""Run one complete editor operation inside the E2B sandbox."""

from __future__ import annotations

import base64
import json
from typing import Any, Mapping

from .editor_runner import EditorError

RUNNER_PATH = "/opt/swe_agent_editor.py"


class AsyncE2BEditor:
    """Execute the installed stdlib runner with one command RPC per operation."""

    def __init__(self, sandbox: Any, max_output_chars: int = 20_000, *, workdir: str = "/testbed") -> None:
        if max_output_chars <= 0:
            raise ValueError("max_output_chars must be positive.")
        self.sandbox = sandbox
        self.user = "swe-agent"
        self.max_output_chars = max_output_chars
        self.workdir = workdir

    async def run(self, arguments: Mapping[str, Any]) -> str:
        """Pass model text as data and restore the existing editor result API."""
        if not isinstance(arguments, Mapping):
            raise EditorError("Tool arguments must be an object.")

        # Base64 has no shell metacharacters. Isolated Python cannot import
        # model-written modules from the repository or the user's environment.
        payload = base64.b64encode(json.dumps(dict(arguments), ensure_ascii=True).encode("ascii")).decode("ascii")
        command = f"/usr/bin/python3 -I {RUNNER_PATH} {payload} {self.max_output_chars}"
        # E2B passes this command as a single bash argument; Linux caps its size.
        if len(command) > 120_000:
            raise EditorError("Editor arguments are too large. Split the edit into smaller operations.")
        result = await self.sandbox.commands.run(command, cwd=self.workdir, user=self.user, timeout=30)
        response = json.loads(result.stdout)
        if response["error"]:
            raise EditorError(response["output"])
        return response["output"]
