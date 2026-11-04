# SWE-agent

Read `env.py` first. `SweAgentEnv.reset()` loads a task and creates its sandbox;
`step()` validates and executes one model action; `truncate()` evaluates the
current patch when the outer loop reaches a limit. `GymAgentLoop` owns model
generation and context compaction.

The environment always uses E2B. Installation, frozen input preparation and
training commands are in [the environment guide](../../../scripts/swe_agent/README.md).

The implementation has five main files:

| File | Responsibility |
| --- | --- |
| `env.py` | Task lifecycle, tool dispatch, terminal reward and metadata |
| `prompts.py` | Task instructions, tool schemas, action parsing and observation text |
| `tasks.py` | SWE-Smith and SWE-bench data loading, image/template names and solution records |
| `e2b_episode.py` | E2B creation, commands, patch capture, fresh-sandbox evaluation and cleanup |
| `editor.py` | Send one editor operation to the sandbox's `editor_runner.py` |

`__init__.py` contains the small shared episode interface, error and evaluation
result types. Importing them does not load a backend.

`e2b_episode.py` is ordered by execution: the agent episode, the fresh evaluator,
SDK helpers, and teardown. Model instructions and tool definitions are together
in `prompts.py`; the prompt is a literal string, with no separate template file.

The E2B editor installs `editor_runner.py` once as a root-owned, read-only file
outside `/testbed`. Each editor action makes one command RPC running isolated
system Python as `swe-agent`; validation, file access and output clipping happen
inside the sandbox. Arguments travel as encoded JSON data. Commands exceeding
120,000 bytes are rejected as editor feedback before the RPC; split such edits
into smaller operations. 

The editor accepts absolute `/testbed/...` paths and paths relative to
`/testbed`, such as `src/file.py` or `./src/file.py`. Use `.` to view the
repository root. Paths containing `..` and symlinks that leave the repository
are rejected; successful operation messages use absolute `/testbed/...` paths.

The E2B semaphore bounds live VMs, including time spent waiting for the model.
A trajectory keeps its permit when its agent VM is killed and replaced by a
fresh evaluator, so evaluation does not queue behind newly admitted agents.
The old VM must be confirmed dead before replacement. Normal evaluator cleanup
releases the permit; uncertain creation or teardown keeps the existing bounded
quarantine behavior. No used VM is reset or shared with another trajectory.

SWE-bench Verified on E2B exposes a single-commit Git checkout at `base_commit`,
including a separate shallow checkout for each initialized submodule. Git commands
are allowed. The original Git
metadata remains root-only, so patch capture uses the initial baseline even when
the agent commits, resets, or changes branches. Network access remains disabled;
the visible repositories have no remotes. SWE-Smith retains its existing Git policy.

The Slurm launcher runs the E2B job cleanup through the absolute path to
`e2b_episode.py --cleanup-job`. It requires `SLURM_JOB_ID`, `SLURM_JOB_NAME`, and
the E2B SDK credentials. The absolute path avoids importing an old checkout
from the working directory; cleanup still matches both job and run identity.

## Training failure policy

Tool mistakes remain model feedback, and genuine test failures keep their normal
reward. Model command/filename rejections are handled at the action boundary;
internal Git, evaluator, authentication, and transport errors remain distinct.
The gym loop runs each episode once. E2B sandbox creation retains its bounded
internal retries; once a patch has been captured, grading gets at most three
attempts on that same patch. These operations share the episode deadline.
An oversized patch ends the trajectory with reward zero and is not retried or
counted as an infrastructure outage. Patch capture/apply failures remain
environment failures. Known nonfatal environment failures receive reward zero
and keep their place in the GRPO group, retaining generated actions and masks.
Reset failures and episode timeouts have no training tokens. Unexpected errors
propagate; the loop does not turn code errors into zero-reward samples.

The fully async rollouter counts final outcomes across workers: **ten environment
failures within the latest fifty logical rollouts** stop generation. It counts
each compacted trajectory once, excludes padding, and does not count benchmark
test failures or oversized patches. The threshold also applies during startup
as soon as ten failures have accumulated. A single all-failed group of eight
does not stop the run by itself, and there is no extra unlimited group retry loop.

Shared authentication/configuration failures during generation use the same
stop path immediately, without publishing an incomplete group. The shared
exception preserves its classification across Ray serialization. Startup
validation and model/trainer invariant errors still raise normally.

On stopping, in-flight groups and their unfinished sibling trajectories are
cancelled and cleaned up before the rollout queue is closed. The sample-count
wait also returns on an outage, so the trainer can reach the termination signal
even when the queue is below its normal threshold. The trainer consumes the
remaining complete batches, saves a checkpoint, and exits without final
validation. Check the logged failure reason before restarting from the
checkpoint; the new run starts with a fresh failure window.
`swe/environment_failures_recent` and `swe/rollouts_recent` expose the window
counts in rollouter statistics.

Patch capture records pre-existing ignored files in the initial inventory, for
the parent repository and each submodule. Changing ignore rules cannot introduce
template caches into the patch; new files still follow the current ignore rules.
