"""Evaluate Qwen3.5-4B's native vLLM tool calling on SWE-bench Verified.

Each Slurm task owns one node and a disjoint set of episodes. The SWE environment
still owns prompts, tool validation, patch capture and isolated test execution.
Use --prepare before launching nodes, and --summarize after they have exited.
"""

import argparse
import asyncio
import importlib.metadata
import json
import os
import secrets
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from uuid import uuid4

import httpx

from agent_system.envs.swe_agent import SweAgentServiceError
from agent_system.envs.swe_agent.env import SweAgentEnv
from agent_system.envs.swe_agent.prompts import tool_schemas
from agent_system.envs.swe_agent.tasks import SweBenchVerifiedTaskStore

from .evaluate import pending_items, summarize, write_json


def prepare(config):
    """Freeze model, task order and protocol; never mix runs with changed settings."""
    for key in (
        "nodes",
        "gpus_per_node",
        "max_active_episodes",
        "samples_per_task",
        "max_tokens",
        "max_model_len",
        "max_episode_tokens",
        "max_turns",
    ):
        assert type(config[key]) is int and config[key] > 0, key
    model = config["model"]
    assert config["gpus_per_node"] % model["tp"] == 0
    assert config["max_active_episodes"] >= config["nodes"]
    assert "seed" not in model["template_kwargs"]
    store = SweBenchVerifiedTaskStore(config["data_path"])
    tasks = store.list_task_specs()[: config["max_tasks"]]
    rows = [dict(extra_info=dict(task, index=i)) for i, task in enumerate(tasks)]
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode()
    protocol = dict(
        tasks=len(rows),
        samples_per_task=config["samples_per_task"],
        total_episodes=len(rows) * config["samples_per_task"],
        model=model,
        tool_parser="qwen3_coder",
        git_checkout="single_base_commit",
        sampling_seed_policy="fresh_os_entropy_engine_no_request_seed",
        episode_budget="generated tokens (including reasoning) plus tokenized observation text",
        context_budget="complete native chat prompt plus generated tokens",
    )
    run = Path(config["output_dir"])
    if (run / "config.json").exists():
        assert json.loads((run / "config.json").read_text()) == config, "Changed evaluation config"
        assert json.loads((run / "protocol.json").read_text()) == protocol, "Changed evaluation protocol"
        assert (run / "rows.jsonl").read_bytes() == payload, "Changed task identities"
    else:
        assert not (run / "results").exists(), "Results without frozen inputs"
        write_json(run / "protocol.json", protocol)
        (run / "rows.jsonl").write_bytes(payload)
        write_json(run / "config.json", config)
    return rows


async def post(client, url, payload):
    """Retry transport/server failures without executing any repository action twice."""
    for attempt in range(3):
        try:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.TransportError, httpx.HTTPStatusError) as error:
            if isinstance(error, httpx.HTTPStatusError) and error.response.status_code < 500:
                raise RuntimeError(f"{url}: {error.response.text[:2000]}") from error
            if attempt == 2:
                raise
            await asyncio.sleep(2**attempt)


def request_body(config, messages, max_tokens):
    """Explicit generation fields prevent fixed seeds or HF sampling defaults leaking in."""
    model = config["model"]
    return dict(
        model=model["name"],
        messages=messages,
        tools=tool_schemas(),
        tool_choice="auto",
        parallel_tool_calls=False,
        temperature=config["temperature"],
        top_p=config["top_p"],
        top_k=config["top_k"],
        repetition_penalty=1.0,
        max_tokens=max_tokens,
        chat_template_kwargs=model["template_kwargs"],
    )


def environment_action(message, finish_reason):
    """Normalize server-parsed calls; truncated or malformed arguments stay model errors."""
    calls = []
    # Parsers can recover partial JSON. A length-truncated call must not execute.
    for call in (message.get("tool_calls") or []) if finish_reason != "length" else []:
        function = call["function"]
        arguments = function["arguments"]
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        calls.append(dict(name=function["name"], arguments=arguments, tool_call_id=call["id"]))
    return dict(assistant_content=message.get("content") or "", tool_calls=calls)


def history_observations(message, observations, info):
    """Rejected native calls still need tool replies to keep model templates well formed."""
    calls = message.get("tool_calls") or []
    if calls and not info.get("format_ok", True):
        text = "\n".join(o["content"] for o in observations) or info.get("error", "Invalid tool call.")
        return [dict(role="tool", tool_call_id=c["id"], name=c["function"]["name"], content=text) for c in calls]
    return observations


def save_trajectory(path, record, messages):
    """Write one complete, chronological UTF-8 text file per episode attempt."""
    counts = {role: sum(m["role"] == role for m in messages) for role in ("user", "assistant", "tool")}
    header = {key: value for key, value in record.items() if key != "events"}
    lines = [json.dumps(header, ensure_ascii=False, indent=2)]
    for i, message in enumerate(messages):
        lines.extend([f"\n--- Message {i}: {message['role']} ---", json.dumps(message, ensure_ascii=False, indent=2)])
    lines.append("\n--- Summary ---\n" + json.dumps(dict(messages=len(messages), turns_by_role=counts)))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def episode(config, client, url, item, attempt_dir, slots):
    """Grade finished and budget-truncated solutions; exclude infrastructure failures."""
    index, sample, row = item
    record = dict(
        task_id=row["extra_info"]["task_id"],
        sample_index=index,
        rollout_index=sample,
        job_id=os.environ.get("SLURM_JOB_ID"),
        status="environment_failure",
        reward=None,
        metadata=dict(episode_tool_call_count=0),
        events=[],
    )
    model = config["model"]
    started = time.monotonic()
    messages = []
    env = SweAgentEnv(
        **config["environment"],
        max_turns=config["max_turns"],
        data_path=config["data_path"],
        dataset="swebench_verified",
        split="test",
        native_tool_calls=True,
        model_name=model["repo"] + "@" + model["revision"],
        e2b_create_concurrency=slots,
        e2b_run_id=os.environ.get("SLURM_JOB_NAME", Path(config["output_dir"]).name),
    )
    try:
        async with asyncio.timeout(config["episode_timeout_seconds"]):
            messages, reset_info = await env.reset(record["task_id"])
            record["reset_info"] = reset_info
            generated = observed = 0
            done = False
            for turn in range(config["max_turns"]):
                body = request_body(config, messages, config["max_tokens"])
                tokenized = await post(
                    client,
                    url + "/tokenize",
                    {key: body[key] for key in ("model", "messages", "tools", "chat_template_kwargs")},
                )
                prompt_tokens = tokenized["count"]
                remaining = min(
                    config["max_model_len"] - prompt_tokens, config["max_episode_tokens"] - generated - observed
                )
                if remaining <= 0:
                    observations, reward, done, info = await env.truncate("context_or_episode_budget")
                    messages.extend(observations)
                    break
                body["max_tokens"] = min(config["max_tokens"], remaining)
                generation_started = time.monotonic()
                response = await post(client, url + "/v1/chat/completions", body)
                usage = response["usage"]
                if usage["prompt_tokens"] != prompt_tokens:
                    raise RuntimeError(
                        f"Tokenizer/chat prompt mismatch: {prompt_tokens} vs {usage['prompt_tokens']}"
                    )
                choice = response["choices"][0]
                message = choice["message"]
                messages.append(message)
                generated += usage["completion_tokens"]
                event = dict(
                    turn=turn,
                    response=response,
                    max_tokens=body["max_tokens"],
                    generation_seconds=time.monotonic() - generation_started,
                )
                record["events"].append(event)
                capped = choice["finish_reason"] == "length" or usage["completion_tokens"] >= body["max_tokens"]
                action = environment_action(message, "length" if capped else choice["finish_reason"])
                observations, reward, done, info = await env.step(action)
                record["metadata"]["episode_tool_call_count"] += len(info.get("tool_calls", []))
                event["environment"] = info
                if any(call["arguments"] is None for call in action["tool_calls"]):
                    # Invalid JSON cannot be re-rendered as a native call. Keep
                    # its exact response in evidence and show it as plain text.
                    messages[-1] = dict(role="assistant", content=json.dumps(message, ensure_ascii=False))
                else:
                    observations = history_observations(message, observations, info)
                messages.extend(observations)
                print(
                    f"TURN task={record['task_id']} sample={sample} turn={turn + 1} "
                    f"action={info.get('action_type')} done={done}",
                    flush=True,
                )
                if done:
                    break
                # Match the existing SWE evaluator: a capped action ends the
                # episode and grades its current patch, rather than continuing it.
                if capped:
                    observations, reward, done, info = await env.truncate("response_length")
                    messages.extend(observations)
                    record["metadata"]["response_length_hit"] = True
                    break
                observation_tokens = await post(
                    client,
                    url + "/tokenize",
                    dict(
                        model=model["name"],
                        prompt="\n".join(o["content"] for o in observations),
                        add_special_tokens=False,
                    ),
                )
                observed += observation_tokens["count"]
            assert done and reward in (0, 1)
            record.update(status="completed", reward=reward, final_info=info)
            record["metadata"].update(
                finish_reason=info["termination_reason"], generated_tokens=generated, observation_tokens=observed
            )
    except SweAgentServiceError as error:
        record.update(error=str(error), error_kind=error.kind)
        if error.kind == "patch_too_large":
            record.update(status="completed", reward=0)
        elif error.is_fatal:
            raise
    except (httpx.TransportError, httpx.HTTPStatusError, TimeoutError) as error:
        record.update(error=str(error), error_kind=type(error).__name__)
    except Exception:
        record["error"] = traceback.format_exc()
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        attempt = attempt_dir / f"{index:03d}_{sample}"
        attempt.mkdir(parents=True, exist_ok=True)
        write_json(attempt / "record.json", record)
        save_trajectory(attempt / "000_trajectory.txt", record, messages)
        try:
            await env.close()
        except Exception as error:
            print(f"Cleanup task={record['task_id']}: {error}", flush=True)
    # Keep fixed episode metadata once; full API responses stay in attempt evidence.
    result = {key: value for key, value in record.items() if key != "events"}
    result["trajectory"] = str(attempt / "000_trajectory.txt")
    write_json(Path(config["output_dir"]) / "results" / f"{index:03d}_{sample}.json", result)
    print(json.dumps({k: result[k] for k in ("task_id", "rollout_index", "status", "reward")}), flush=True)
    return result


async def run_node(config, rows, node, urls, attempt_dir, servers):
    """One queue and fixed worker count bound both inference requests and live sandboxes."""
    slots = config["max_active_episodes"] // config["nodes"] + (node < config["max_active_episodes"] % config["nodes"])
    queue = asyncio.Queue()
    for item in pending_items(rows, Path(config["output_dir"]), config["samples_per_task"]):
        index, sample, _ = item
        if (sample * len(rows) + index) % config["nodes"] == node:
            queue.put_nowait(item)
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(1800, connect=30), limits=httpx.Limits(max_connections=slots + len(urls))
    ) as client:

        async def worker(worker_index):
            while not queue.empty():
                if any(server.poll() is not None for server in servers):
                    raise RuntimeError("An inference server exited; preserve pending episodes for continuation")
                item = queue.get_nowait()
                await episode(config, client, urls[worker_index % len(urls)], item, attempt_dir, slots)

        async with asyncio.TaskGroup() as group:
            for i in range(slots):
                group.create_task(worker(i))


def launch_servers(config, attempt_dir):
    """Run independent TP replicas on each node, with fresh engine RNG state."""
    model = config["model"]
    servers, logs, urls = [], [], []
    try:
        for replica in range(config["gpus_per_node"] // model["tp"]):
            port = 8123 + replica
            url = f"http://127.0.0.1:{port}"
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model["path"],
                "--served-model-name",
                model["name"],
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--tensor-parallel-size",
                str(model["tp"]),
                "--max-model-len",
                str(config["max_model_len"]),
                "--max-num-seqs",
                "8",
                "--max-num-batched-tokens",
                "4096",
                "--gpu-memory-utilization",
                "0.85",
                "--enforce-eager",
                "--generation-config",
                "vllm",
                "--seed",
                str(secrets.randbelow(2**31)),
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "qwen3_coder",
            ]
            environment = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=",".join(
                    str(i) for i in range(replica * model["tp"], (replica + 1) * model["tp"])
                ),
            )
            log = (attempt_dir / f"server_{replica}.log").open("w")
            logs.append(log)
            write_json(attempt_dir / f"server_{replica}_command.json", command)
            servers.append(
                subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            )
            urls.append(url)
        with httpx.Client(timeout=5) as client:
            ready = set()
            for _ in range(240):
                if any(server.poll() is not None for server in servers):
                    raise RuntimeError("vLLM exited during startup; inspect server logs")
                for url in urls:
                    try:
                        if client.get(url + "/health").status_code == 200:
                            ready.add(url)
                    except httpx.TransportError:
                        pass
                if len(ready) == len(urls):
                    return servers, logs, urls
                time.sleep(5)
        raise TimeoutError("vLLM startup exceeded twenty minutes")
    except BaseException:
        stop_servers(servers, logs)
        raise


def stop_servers(servers, logs):
    for server in servers:
        if server.poll() is None:
            os.killpg(server.pid, signal.SIGTERM)
    for server in servers:
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(server.pid, signal.SIGKILL)
            server.wait()
    for log in logs:
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.prepare:
        rows = prepare(config)
        print(f"Prepared {len(rows)} tasks x {config['samples_per_task']} rollouts")
        return
    run = Path(config["output_dir"])
    assert json.loads((run / "config.json").read_text()) == config
    rows = [json.loads(line) for line in (run / "rows.jsonl").read_text().splitlines()]
    if args.summarize:
        print(json.dumps(summarize(rows, run, config["samples_per_task"]), indent=2))
        return
    import fcntl

    node = int(os.environ.get("SLURM_NODEID", "0"))
    assert int(os.environ.get("SLURM_JOB_NUM_NODES", "1")) == config["nodes"]
    if not any(
        (sample * len(rows) + index) % config["nodes"] == node
        for index, sample, _ in pending_items(rows, run, config["samples_per_task"])
    ):
        print(f"Node {node}: no pending episodes", flush=True)
        return
    with (run / f"node_{node}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        attempt_dir = run / "attempts" / f"{os.environ.get('SLURM_JOB_ID', 'local')}_{node}_{uuid4().hex[:8]}"
        attempt_dir.mkdir(parents=True)
        write_json(
            attempt_dir / "runtime.json",
            dict(
                node=node, versions={p: importlib.metadata.version(p) for p in ("vllm", "transformers", "torch", "e2b")}
            ),
        )
        servers, logs, urls = launch_servers(config, attempt_dir)
        try:
            asyncio.run(run_node(config, rows, node, urls, attempt_dir, servers))
        finally:
            stop_servers(servers, logs)


if __name__ == "__main__":
    main()
