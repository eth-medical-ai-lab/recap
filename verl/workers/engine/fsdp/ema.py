"""FP32 EMA shards for scoring with the actor's existing FSDP2 model."""

from contextlib import contextmanager

import torch
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor


class FSDPTeacherEMA:
    """Keep EMA on CPU; temporarily swap local shards for no-gradient scoring."""

    def __init__(self, module, rate: float):
        if not 0 < rate <= 1:
            raise ValueError("Teacher EMA rate must be in (0, 1].")
        self.module, self.rate, self.version = module, rate, 0
        self.weights = {
            name: shard.detach().to(device="cpu", dtype=torch.float32, copy=True)
            for name, shard in self._shards().items()
        }

    def _shards(self):
        # Discard any full-parameter cache before changing local shards. This
        # matters when the training configuration disables reshard-after-forward.
        for child in self.module.modules():
            if isinstance(child, FSDPModule):
                child.reshard()
        return {
            name: param.to_local() if isinstance(param, DTensor) else param
            for name, param in self.module.named_parameters()
        }

    @torch.no_grad()
    def update(self, version: int):
        if version == self.version:
            return
        if version != self.version + 1:
            raise ValueError(f"Teacher EMA expected version {self.version + 1}, received {version}.")
        for name, shard in self._shards().items():
            self.weights[name].lerp_(shard.detach().to(device="cpu", dtype=torch.float32), self.rate)
        self.version = version

    @contextmanager
    def apply(self):
        # Preserve the exact student shards and optimizer parameter objects.
        with torch.no_grad():
            shards = self._shards()
            student = {name: shard.detach().to(device="cpu", copy=True) for name, shard in shards.items()}
            try:
                for name, shard in shards.items():
                    shard.copy_(self.weights[name])
                yield
            finally:
                for name, shard in self._shards().items():
                    shard.copy_(student[name])

    def save(self, path, version):
        assert self.version == version
        torch.save(dict(rate=self.rate, version=self.version, weights=self.weights), path)

    def load(self, path):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["rate"] != self.rate:
            raise ValueError("Resumed teacher EMA rate differs from its checkpoint.")
        self.version, self.weights = state["version"], state["weights"]
