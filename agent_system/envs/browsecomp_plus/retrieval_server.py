from __future__ import annotations

import argparse
import atexit
import glob
import multiprocessing as mp
import os
import pickle
import queue
import threading
import time
import traceback
from concurrent.futures import Future
from contextlib import nullcontext
from typing import Any

TASK_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"


def truncate_text(tokenizer, text: str, max_tokens: int) -> str:
    """Return the first max_tokens content tokens without adding model tokens."""
    if max_tokens <= 0:
        return text
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)


def _model_cache_dir() -> str | None:
    """Use the standard Hugging Face hub cache instead of duplicating large models."""
    if cache_dir := os.environ.get("HF_HUB_CACHE"):
        return cache_dir
    if hf_home := os.environ.get("HF_HOME"):
        return os.path.join(hf_home, "hub")
    return None


class DenseSearcher:
    """Qwen query encoder and read-only FAISS index for BrowseComp-Plus."""

    def __init__(
        self,
        index_path: str,
        model_name: str,
        dataset_name: str,
        max_length: int,
        torch_dtype: str,
    ) -> None:
        import numpy as np
        import torch
        from datasets import load_dataset
        from tevatron.retriever.driver.encode import DenseModel
        from tevatron.retriever.searcher import FaissFlatSearcher
        from transformers import AutoTokenizer

        self.torch = torch
        self.max_length = max_length
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[torch_dtype]

        # Load every hosted embedding shard into one CPU inner-product index.
        # The corpus is only about 100K documents, so GPU FAISS is unnecessary.
        index_files = sorted(glob.glob(index_path))
        if not index_files:
            raise FileNotFoundError(f"No BrowseComp-Plus index files matched: {index_path}")

        self.lookup: list[str] = []
        self.retriever = None
        for path in index_files:
            with open(path, "rb") as file:
                representations, lookup = pickle.load(file)
            representations = np.asarray(representations, dtype=np.float32)
            if self.retriever is None:
                self.retriever = FaissFlatSearcher(representations)
            self.retriever.add(representations)
            self.lookup.extend(str(docid) for docid in lookup)

        cache_dir = _model_cache_dir()
        self.model = DenseModel.load(
            model_name,
            pooling="eos",
            normalize=True,
            cache_dir=cache_dir,
            torch_dtype=self.dtype,
            attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            padding_side="left",
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        dataset_cache = os.environ.get("HF_DATASETS_CACHE")
        corpus = load_dataset(dataset_name, split="train", cache_dir=dataset_cache)
        self.docid_to_text = {str(row["docid"]): str(row["text"]) for row in corpus}

    def search_batch(self, queries: list[str], k: int) -> list[list[dict[str, Any]]]:
        """Encode one GPU batch and search the CPU FAISS index as one matrix."""
        batch = self.tokenizer(
            [TASK_PREFIX + query for query in queries],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}

        use_autocast = self.device == "cuda" and self.dtype != self.torch.float32
        autocast = self.torch.amp.autocast("cuda", dtype=self.dtype) if use_autocast else nullcontext()
        with autocast, self.torch.no_grad():
            query_representation = self.model.encode_query(batch)
        query_representation = query_representation.float().cpu().numpy()

        scores, indices = self.retriever.search(query_representation, k)
        batch_results = []
        for query_scores, query_indices in zip(scores, indices, strict=False):
            results = []
            for score, index in zip(query_scores, query_indices, strict=False):
                docid = self.lookup[int(index)]
                results.append(
                    {
                        "docid": docid,
                        "score": float(score),
                        "text": self.docid_to_text[docid],
                    }
                )
            batch_results.append(results)
        return batch_results

    def search(self, query: str, k: int) -> list[dict[str, Any]]:
        return self.search_batch([query], k)[0]

    def get_document(self, docid: str) -> dict[str, str] | None:
        text = self.docid_to_text.get(str(docid))
        if text is None:
            return None
        return {"docid": str(docid), "text": text}


def _gpu_search_worker(
    worker_id: int,
    index_path: str,
    model_name: str,
    max_length: int,
    torch_dtype: str,
    corpus_torch_dtype: str,
    batch_queue,
    result_queue,
    ready_queue,
) -> None:
    """Own one embedding model and one exact corpus matrix on one GPU."""
    try:
        import numpy as np
        import torch
        from tevatron.retriever.driver.encode import DenseModel
        from transformers import AutoTokenizer

        device = torch.device(f"cuda:{worker_id}")
        torch.cuda.set_device(device)
        model_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[torch_dtype]
        corpus_dtype = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[corpus_torch_dtype]

        # Every worker keeps a complete corpus matrix. The fixed index is small
        # enough to replicate, which avoids a shared CPU FAISS bottleneck.
        index_files = sorted(glob.glob(index_path))
        if not index_files:
            raise FileNotFoundError(f"No BrowseComp-Plus index files matched: {index_path}")
        matrices = []
        lookup = []
        for path in index_files:
            with open(path, "rb") as file:
                representations, shard_lookup = pickle.load(file)
            matrices.append(np.asarray(representations, dtype=np.float32))
            lookup.extend(str(docid) for docid in shard_lookup)
        corpus_embeddings = torch.from_numpy(np.concatenate(matrices, axis=0)).to(
            device=device,
            dtype=corpus_dtype,
        )
        del matrices

        cache_dir = _model_cache_dir()
        model = DenseModel.load(
            model_name,
            pooling="eos",
            normalize=True,
            cache_dir=cache_dir,
            torch_dtype=model_dtype,
            attn_implementation="flash_attention_2",
        )
        model = model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            padding_side="left",
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        ready_queue.put((worker_id, None))
        while True:
            work = batch_queue.get()
            if work is None:
                return
            batch_id, queries, k = work
            start = time.perf_counter()
            try:
                batch = tokenizer(
                    [TASK_PREFIX + query for query in queries],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                batch = {key: value.to(device) for key, value in batch.items()}
                tokenize_seconds = time.perf_counter() - start

                encode_start = time.perf_counter()
                use_autocast = model_dtype != torch.float32
                autocast = torch.amp.autocast("cuda", dtype=model_dtype) if use_autocast else nullcontext()
                with autocast, torch.no_grad():
                    query_embeddings = model.encode_query(batch)
                torch.cuda.synchronize(device)
                encode_seconds = time.perf_counter() - encode_start

                search_start = time.perf_counter()
                query_embeddings = query_embeddings.to(dtype=corpus_dtype)
                similarities = torch.matmul(query_embeddings, corpus_embeddings.T)
                scores, indices = torch.topk(
                    similarities,
                    k=min(k, len(lookup)),
                    dim=1,
                )
                scores = scores.float().cpu().tolist()
                indices = indices.cpu().tolist()
                search_seconds = time.perf_counter() - search_start

                results = []
                for query_scores, query_indices in zip(scores, indices, strict=False):
                    results.append(
                        [
                            {
                                "docid": lookup[index],
                                "score": float(score),
                            }
                            for score, index in zip(query_scores, query_indices, strict=False)
                        ]
                    )
                result_queue.put(
                    (
                        batch_id,
                        results,
                        None,
                        {
                            "worker": worker_id,
                            "batch_size": len(queries),
                            "tokenize_seconds": tokenize_seconds,
                            "encode_seconds": encode_seconds,
                            "search_seconds": search_seconds,
                            "total_seconds": time.perf_counter() - start,
                        },
                    )
                )
            except Exception:
                result_queue.put((batch_id, None, traceback.format_exc(), None))
    except Exception:
        ready_queue.put((worker_id, traceback.format_exc()))


class GpuSearchPool:
    """Batch concurrent searches across one independent process per GPU."""

    def __init__(
        self,
        index_path: str,
        model_name: str,
        dataset_name: str,
        max_length: int,
        torch_dtype: str,
        corpus_torch_dtype: str,
        k: int,
        num_workers: int,
        max_batch_size: int,
        batch_timeout_ms: float,
    ) -> None:
        from datasets import load_dataset

        dataset_cache = os.environ.get("HF_DATASETS_CACHE")
        corpus = load_dataset(dataset_name, split="train", cache_dir=dataset_cache)
        self.docid_to_text = {str(row["docid"]): str(row["text"]) for row in corpus}
        self.k = k
        self.num_workers = num_workers
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout_ms / 1000.0
        self.requests: queue.Queue[tuple[str, Future] | None] = queue.Queue()
        self.pending: dict[int, list[Future]] = {}
        self.pending_lock = threading.Lock()
        self.next_batch_id = 0
        self.closed = False

        context = mp.get_context("spawn")
        self.batch_queue = context.Queue(maxsize=1024)
        self.result_queue = context.Queue(maxsize=1024)
        self.ready_queue = context.Queue()
        self.workers = []
        for worker_id in range(num_workers):
            worker = context.Process(
                target=_gpu_search_worker,
                args=(
                    worker_id,
                    index_path,
                    model_name,
                    max_length,
                    torch_dtype,
                    corpus_torch_dtype,
                    self.batch_queue,
                    self.result_queue,
                    self.ready_queue,
                ),
                daemon=True,
            )
            worker.start()
            self.workers.append(worker)

        # Do not expose the MCP endpoint until every model and index is ready.
        for _ in self.workers:
            worker_id, error = self.ready_queue.get(timeout=3600)
            if error is not None:
                self.close()
                raise RuntimeError(f"Retrieval worker {worker_id} failed to start:\n{error}")

        self.dispatcher = threading.Thread(
            target=self._dispatch,
            name="gpu-search-dispatcher",
            daemon=True,
        )
        self.collector = threading.Thread(
            target=self._collect,
            name="gpu-search-collector",
            daemon=True,
        )
        self.dispatcher.start()
        self.collector.start()
        atexit.register(self.close)

    def search(self, query: str) -> list[dict[str, Any]]:
        if self.closed:
            raise RuntimeError("The GPU search pool is closed")
        future = Future()
        self.requests.put((query, future))
        return future.result()

    def get_document(self, docid: str) -> dict[str, str] | None:
        text = self.docid_to_text.get(str(docid))
        if text is None:
            return None
        return {"docid": str(docid), "text": text}

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if hasattr(self, "dispatcher"):
            self.requests.put(None)
            self.dispatcher.join(timeout=30)
        else:
            for _ in getattr(self, "workers", []):
                self.batch_queue.put(None)
        for worker in getattr(self, "workers", []):
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=10)
        if hasattr(self, "collector"):
            self.result_queue.put(None)
            self.collector.join(timeout=10)

    def _dispatch(self) -> None:
        stop = False
        while not stop:
            first = self.requests.get()
            if first is None:
                break

            batch = [first]
            deadline = time.perf_counter() + self.batch_timeout
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    request = self.requests.get(timeout=remaining)
                except queue.Empty:
                    break
                if request is None:
                    stop = True
                    break
                batch.append(request)

            batch_id = self.next_batch_id
            self.next_batch_id += 1
            with self.pending_lock:
                self.pending[batch_id] = [future for _, future in batch]
            self.batch_queue.put((batch_id, [query for query, _ in batch], self.k))

        for _ in self.workers:
            self.batch_queue.put(None)

    def _collect(self) -> None:
        while True:
            item = self.result_queue.get()
            if item is None:
                return
            batch_id, results, error, timing = item
            with self.pending_lock:
                futures = self.pending.pop(batch_id)

            if error is not None:
                exception = RuntimeError(error)
                for future in futures:
                    future.set_exception(exception)
                continue

            for candidates, future in zip(results, futures, strict=False):
                for candidate in candidates:
                    candidate["text"] = self.docid_to_text[candidate["docid"]]
                future.set_result(candidates)

            print(
                "retrieval_batch "
                f"timestamp={time.time():.3f} "
                f"worker={timing['worker']} "
                f"size={timing['batch_size']} "
                f"tokenize_ms={timing['tokenize_seconds'] * 1000:.1f} "
                f"encode_ms={timing['encode_seconds'] * 1000:.1f} "
                f"search_ms={timing['search_seconds'] * 1000:.1f} "
                f"total_ms={timing['total_seconds'] * 1000:.1f}",
                flush=True,
            )


class SearchBatcher:
    """Collect concurrent tool calls briefly and run one embedding forward pass."""

    def __init__(
        self,
        searcher: DenseSearcher,
        k: int,
        max_batch_size: int,
        batch_timeout_ms: float,
    ) -> None:
        self.searcher = searcher
        self.k = k
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout_ms / 1000.0
        self.requests: queue.Queue[tuple[str, Future] | None] = queue.Queue()
        self.worker = threading.Thread(target=self._run, name="search-batcher", daemon=True)
        self.worker.start()

    def search(self, query: str) -> list[dict[str, Any]]:
        future = Future()
        self.requests.put((query, future))
        return future.result()

    def close(self) -> None:
        self.requests.put(None)
        self.worker.join()

    def _run(self) -> None:
        while True:
            first = self.requests.get()
            if first is None:
                return

            batch = [first]
            deadline = time.perf_counter() + self.batch_timeout

            # Wait only a few milliseconds so concurrent trajectories can share
            # one GPU forward pass without adding noticeable tool latency.
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    request = self.requests.get(timeout=remaining)
                except queue.Empty:
                    break
                if request is None:
                    self.requests.put(None)
                    break
                batch.append(request)

            try:
                results = self.searcher.search_batch(
                    [query for query, _ in batch],
                    self.k,
                )
            except Exception as error:
                for _, future in batch:
                    future.set_exception(error)
                continue

            for result, (_, future) in zip(results, batch, strict=False):
                future.set_result(result)


def create_mcp_server(
    search_backend,
    document_store,
    k: int,
    snippet_max_tokens: int,
    document_max_tokens: int,
):
    """Create MCP tools plus direct HTTP routes for latency-sensitive rollouts."""
    from fastmcp import FastMCP
    from starlette.concurrency import run_in_threadpool
    from starlette.requests import Request
    from starlette.responses import JSONResponse
    from transformers import AutoTokenizer

    snippet_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-0.6B",
        cache_dir=_model_cache_dir(),
    )
    mcp = FastMCP(name="search-server")

    def run_search(query: str) -> list[dict[str, Any]]:
        candidates = search_backend.search(query)
        results = []
        for candidate in candidates:
            text = truncate_text(snippet_tokenizer, candidate["text"], snippet_max_tokens)
            results.append(
                {
                    "docid": candidate["docid"],
                    "score": candidate["score"],
                    "snippet": text,
                }
            )
        return results

    def run_get_document(docid: str) -> dict[str, str] | None:
        document = document_store.get_document(docid)
        if document is None:
            return None
        document["text"] = truncate_text(
            snippet_tokenizer,
            document["text"],
            document_max_tokens,
        )
        return document

    @mcp.tool(
        name="search",
        description=(f"Search the BrowseComp-Plus collection. Returns top-{k} hits with docid, score, and snippet."),
    )
    def search(query: str) -> list[dict[str, Any]]:
        return run_search(query)

    @mcp.tool(
        name="get_document",
        description=f"Retrieve the first {document_max_tokens} tokens of a document by its docid.",
    )
    def get_document(docid: str) -> dict[str, str] | None:
        return run_get_document(docid)

    # Rollout workers use these routes with persistent connections. Running
    # blocking retrieval in Starlette's thread pool lets concurrent requests
    # reach the 5 ms GPU micro-batcher together.
    @mcp.custom_route("/search", methods=["POST"])
    async def search_http(request: Request) -> JSONResponse:
        payload = await request.json()
        results = await run_in_threadpool(run_search, str(payload["query"]))
        return JSONResponse(results)

    @mcp.custom_route("/get_document", methods=["POST"])
    async def get_document_http(request: Request) -> JSONResponse:
        payload = await request.json()
        document = await run_in_threadpool(run_get_document, str(payload["docid"]))
        return JSONResponse(document)

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser(description="BrowseComp-Plus dense retrieval MCP server")
    parser.add_argument("--index-path", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--dataset-name", default="Tevatron/browsecomp-plus-corpus")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--snippet-max-tokens", type=int, default=512)
    parser.add_argument("--document-max-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-timeout-ms", type=float, default=5.0)
    parser.add_argument(
        "--gpu-workers",
        type=int,
        default=-1,
        help="GPU corpus-search workers; -1 uses every visible GPU, 0 keeps the legacy backend.",
    )
    parser.add_argument(
        "--corpus-torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float32",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    import torch

    visible_gpus = torch.cuda.device_count()
    gpu_workers = visible_gpus if args.gpu_workers < 0 else args.gpu_workers
    if gpu_workers > visible_gpus:
        raise ValueError(f"Requested {gpu_workers} GPU workers, but only {visible_gpus} GPUs are visible.")

    if gpu_workers > 0:
        search_backend = GpuSearchPool(
            index_path=args.index_path,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            max_length=args.max_length,
            torch_dtype=args.torch_dtype,
            corpus_torch_dtype=args.corpus_torch_dtype,
            k=args.k,
            num_workers=gpu_workers,
            max_batch_size=args.batch_size,
            batch_timeout_ms=args.batch_timeout_ms,
        )
        document_store = search_backend
    else:
        document_store = DenseSearcher(
            index_path=args.index_path,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            max_length=args.max_length,
            torch_dtype=args.torch_dtype,
        )
        search_backend = SearchBatcher(
            searcher=document_store,
            k=args.k,
            max_batch_size=args.batch_size,
            batch_timeout_ms=args.batch_timeout_ms,
        )

    mcp = create_mcp_server(
        search_backend=search_backend,
        document_store=document_store,
        k=args.k,
        snippet_max_tokens=args.snippet_max_tokens,
        document_max_tokens=args.document_max_tokens,
    )
    try:
        mcp.run(
            transport="streamable-http",
            host=args.host,
            port=args.port,
            path="/mcp",
        )
    finally:
        search_backend.close()


if __name__ == "__main__":
    main()
