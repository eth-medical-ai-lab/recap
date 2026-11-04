from __future__ import annotations

import os
import pickle
from pathlib import Path

from huggingface_hub import snapshot_download

from agent_system.envs.browsecomp_plus.retrieval_server import DenseSearcher

INDEX_REPO = "Tevatron/browsecomp-plus-indexes"
INDEX_ROOT = Path(os.environ.get("BROWSECOMP_PLUS_INDEX_DIR", "data/browsecomp_plus/indexes"))
INDEX_DIR = INDEX_ROOT / "qwen3-embedding-8b"
INDEX_GLOB = str(INDEX_DIR / "corpus.shard*.pkl")
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
DATASET_NAME = "Tevatron/browsecomp-plus-corpus"


def main() -> None:
    """Download the hosted 8B retrieval artifacts and verify one real lookup."""
    INDEX_ROOT.mkdir(parents=True, exist_ok=True)

    # The corpus is already embedded upstream, so only download the matching shards.
    snapshot_download(
        repo_id=INDEX_REPO,
        repo_type="dataset",
        allow_patterns="qwen3-embedding-8b/*",
        local_dir=INDEX_ROOT,
        token=os.environ.get("HF_TOKEN"),
    )
    snapshot_download(
        repo_id=MODEL_NAME,
        token=os.environ.get("HF_TOKEN"),
    )

    # Check every shard before loading the larger model and running the smoke query.
    index_files = sorted(INDEX_DIR.glob("corpus.shard*.pkl"))
    if len(index_files) != 4:
        raise RuntimeError(f"Expected four Qwen3-Embedding-8B shards, found {index_files}")

    embedding_dim = None
    document_count = 0
    for index_file in index_files:
        with index_file.open("rb") as file:
            representations, lookup = pickle.load(file)
        if representations.ndim != 2 or len(representations) != len(lookup):
            raise RuntimeError(f"Invalid embedding shard: {index_file}")
        if embedding_dim is None:
            embedding_dim = representations.shape[1]
        elif representations.shape[1] != embedding_dim:
            raise RuntimeError(f"Embedding dimensions differ in {index_file}")
        document_count += len(lookup)
        del representations, lookup

    print(f"verified {len(index_files)} index shards, {document_count} documents, dimension {embedding_dim}")

    searcher = DenseSearcher(
        index_path=INDEX_GLOB,
        model_name=MODEL_NAME,
        dataset_name=DATASET_NAME,
        max_length=512,
        torch_dtype="bfloat16",
    )
    results = searcher.search("capital of Switzerland", k=5)
    if len(results) != 5:
        raise RuntimeError(f"Expected five retrieval results, got {results}")
    document = searcher.get_document(results[0]["docid"])
    if document is None:
        raise RuntimeError(f"Could not load retrieved document {results[0]['docid']}")
    print("retrieval smoke test docids:", [result["docid"] for result in results])


if __name__ == "__main__":
    main()
