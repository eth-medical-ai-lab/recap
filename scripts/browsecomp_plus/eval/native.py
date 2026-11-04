"""Full-context BC+ evaluation using Qwen3.5-4B's native tool calling in vLLM.

The existing environment owns prompts, retrieval and answer submission. This
backend only replaces model transport/history rendering; grading stays shared.
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import httpx
from omegaconf import OmegaConf

from agent_system.envs.browsecomp_plus.env import BrowseCompPlusEnv
from agent_system.envs.browsecomp_plus.prompts import BROWSECOMP_PLUS_TOOL_SCHEMAS

from .evaluate import sampling_params
from .records import write_json


def request_body(config, messages, max_tokens):
    settings = config.evaluation
    return dict(
        model="policy",
        messages=messages,
        tools=BROWSECOMP_PLUS_TOOL_SCHEMAS,
        tool_choice="auto",
        parallel_tool_calls=False,
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        repetition_penalty=1.0,
        max_tokens=max_tokens,
        chat_template_kwargs=OmegaConf.to_container(config.data.apply_chat_template_kwargs),
    )


async def post(client, url, body):
    response = await client.post(url, json=body)
    if response.is_error:
        raise RuntimeError(f"{url}: HTTP {response.status_code}: {response.text[:2000]}")
    return response.json()


def environment_action(message):
    calls = []
    for call in message.get("tool_calls") or []:
        function = call["function"]
        arguments = function["arguments"]
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        calls.append(dict(name=function["name"], arguments=arguments))
    return calls


def history_observations(message, observations, info):
    """Attach native call identities without changing BC+'s observation contents."""
    calls = message.get("tool_calls") or []
    if not calls:
        return observations
    if not info["format_ok"]:
        text = "\n".join(o["content"] for o in observations)
        return [dict(role="tool", name=c["function"]["name"], tool_call_id=c["id"], content=text) for c in calls]
    # A finish ends the episode, so BC+ intentionally ignores later calls.
    assert len(observations) <= len(calls)
    return [
        dict(o, name=c["function"]["name"], tool_call_id=c["id"]) for c, o in zip(calls, observations, strict=False)
    ]


def save_trajectory(path, record, messages):
    counts = {role: sum(m["role"] == role for m in messages) for role in ("system", "user", "assistant", "tool")}
    lines = [json.dumps(record, ensure_ascii=False, indent=2)]
    for index, message in enumerate(messages):
        lines += [f"\n--- Message {index}: {message['role']} ---", json.dumps(message, ensure_ascii=False, indent=2)]
    lines += ["\n--- Summary ---", json.dumps(dict(messages=len(messages), turns_by_role=counts))]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def episode(config, client, url, item, attempt):
    """Keep failed/non-submitted model outcomes; infrastructure errors stay resumable."""
    index, sample, row = item
    settings = config.evaluation
    run = Path(settings.output_dir)
    evidence = run / "attempts" / attempt / f"{index:03d}_{sample}"
    evidence.mkdir(parents=True, exist_ok=True)
    env = BrowseCompPlusEnv(
        data_path=str(run / "tasks.jsonl"),
        split=row["extra_info"]["split"],
        mcp_url=config.trainer.gym_env_kwargs.mcp_url,
        defer_judge=True,
    )
    metadata = dict(
        turn_token_limit_hit=False,
        compact_response_tokens=[],
        invalid_action_count=0,
        step_response_lengths=[],
        max_context_token_count=0,
    )
    record = dict(
        index=index,
        sample=sample,
        task_id=str(row["extra_info"]["task_id"]),
        job_id=attempt,
        status="error",
        final_answer=None,
        finish_reason="turn_limit",
        tool_calls=0,
        sampling=sampling_params(config),
        metadata=metadata,
    )
    messages = []
    started = time.monotonic()
    try:
        messages, reset_info = await asyncio.to_thread(env.reset, record["task_id"])
        record["reset_info"] = reset_info
        for turn in range(settings.max_turns):
            body = request_body(config, messages, settings.max_response_tokens)
            token_body = {key: body[key] for key in ("model", "messages", "tools", "chat_template_kwargs")}
            tokenized = await post(client, url + "/tokenize", token_body)
            count = tokenized["count"]
            remaining = settings.max_model_len - count
            if remaining <= 0:
                record["finish_reason"] = "context_length"
                break
            body["max_tokens"] = min(settings.max_response_tokens, remaining)
            write_json(evidence / f"request_{turn:03d}.json", body)
            response = await post(client, url + "/v1/chat/completions", body)
            write_json(evidence / f"response_{turn:03d}.json", response)
            usage = response["usage"]
            assert usage["prompt_tokens"] == count, (
                f"Tokenize/chat history mismatch: {count} vs {usage['prompt_tokens']}"
            )
            tokens = usage["completion_tokens"]
            metadata["step_response_lengths"].append(tokens)
            metadata["max_context_token_count"] = max(metadata["max_context_token_count"], count + tokens)
            choice = response["choices"][0]
            message = choice["message"]
            messages.append(message)
            observations, _, done, info = await asyncio.to_thread(env.step, environment_action(message))
            record["tool_calls"] += len(info.get("tool_calls", []))
            metadata["invalid_action_count"] += int(not info["format_ok"])
            write_json(evidence / f"environment_{turn:03d}.json", info)
            messages.extend(history_observations(message, observations, info))
            capped = choice["finish_reason"] == "length" or tokens >= body["max_tokens"]
            metadata["turn_token_limit_hit"] |= capped
            # Like GymAgentLoop, execute the generated action before applying a cap.
            if done:
                record.update(final_answer=info["final_answer"], finish_reason="done")
                break
            if capped:
                record["finish_reason"] = (
                    "context_length" if remaining <= settings.max_response_tokens else "response_length"
                )
                break
        record["status"] = "completed"
    except Exception:
        record.update(finish_reason="infrastructure_error", error=traceback.format_exc())
    finally:
        env.close()
        record["elapsed_seconds"] = time.monotonic() - started
        write_json(evidence / "record.json", record)
        save_trajectory(evidence / "000_trajectory.txt", record, messages)
        write_json(run / "rollouts" / f"{index:03d}_{sample}.json", record)
        print(
            json.dumps({k: record[k] for k in ("index", "sample", "status", "finish_reason", "tool_calls")}), flush=True
        )
    return record


async def collect_from_servers(config, client, urls, items, attempt):
    """Assign complete episodes to the local model replicas."""
    queue = asyncio.Queue()
    for item in items:
        queue.put_nowait(item)

    # Each worker stays on the same replica for the complete episode.
    async def worker(index):
        while not queue.empty():
            await episode(config, client, urls[index % len(urls)], queue.get_nowait(), attempt)

    await asyncio.gather(*(worker(i) for i in range(min(config.evaluation.workers, len(items)))))


async def collect_native(config, items, attempt):
    """Own all TP replicas on the rollout node; always terminate owned servers."""
    settings, rollout = config.evaluation, config.actor_rollout_ref.rollout
    attempt_dir = Path(settings.output_dir) / "attempts" / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    assert rollout.nnodes == 1 and rollout.n_gpus_per_node % rollout.tensor_model_parallel_size == 0
    replicas = rollout.n_gpus_per_node // rollout.tensor_model_parallel_size
    servers, logs, urls = [], [], []
    try:
        for replica in range(replicas):
            port = 8123 + replica
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                settings.model_path,
                "--served-model-name",
                "policy",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--tensor-parallel-size",
                str(rollout.tensor_model_parallel_size),
                "--max-model-len",
                str(settings.max_model_len),
                "--dtype",
                "bfloat16",
                "--max-num-seqs",
                str(rollout.max_num_seqs),
                "--max-num-batched-tokens",
                str(rollout.max_num_batched_tokens),
                "--gpu-memory-utilization",
                str(rollout.standalone_gpu_memory_utilization),
                "--generation-config",
                "vllm",
                "--seed",
                str(rollout.seed + replica),
                "--enable-auto-tool-choice",
                "--tool-call-parser",
                "qwen3_coder",
                "--language-model-only",
            ]
            if rollout.enforce_eager:
                command += ["--enforce-eager"]
            first_gpu = replica * rollout.tensor_model_parallel_size
            environment = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=",".join(
                    str(i) for i in range(first_gpu, first_gpu + rollout.tensor_model_parallel_size)
                ),
            )
            log = (attempt_dir / f"server_{replica}.log").open("w")
            logs.append(log)
            write_json(attempt_dir / f"server_{replica}_command.json", command)
            servers.append(
                subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            )
            urls.append(f"http://127.0.0.1:{port}")
        async with httpx.AsyncClient(timeout=httpx.Timeout(900, connect=30)) as client:
            ready = set()
            for _ in range(240):
                if any(server.poll() is not None for server in servers):
                    raise RuntimeError("Native vLLM exited; inspect server logs")
                for url in urls:
                    if url not in ready:
                        try:
                            response = await client.get(url + "/health", timeout=5)
                            if response.status_code == 200:
                                ready.add(url)
                        except httpx.TransportError:
                            pass
                if len(ready) == replicas:
                    break
                await asyncio.sleep(5)
            else:
                raise TimeoutError("Native vLLM startup exceeded twenty minutes")
            await collect_from_servers(config, client, urls, items, attempt)
    finally:
        for server in servers:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
        for server in servers:
            try:
                await asyncio.to_thread(server.wait, timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                await asyncio.to_thread(server.wait)
        for log in logs:
            log.close()
