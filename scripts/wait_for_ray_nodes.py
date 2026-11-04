"""Wait until every Slurm node has joined the Ray cluster."""

import os
import time

import ray

address = os.environ["RAY_ADDRESS"]
expected_nodes = int(os.environ["EXPECTED_RAY_NODES"])
deadline = time.time() + 600

ray.init(address=address, logging_level="ERROR")

while time.time() < deadline:
    alive_nodes = sum(node["Alive"] for node in ray.nodes())
    print(f"Ray nodes ready: {alive_nodes}/{expected_nodes}", flush=True)
    if alive_nodes >= expected_nodes:
        break
    time.sleep(5)
else:
    raise TimeoutError(f"Only {alive_nodes}/{expected_nodes} Ray nodes joined within 600 seconds")
