# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared test fixtures for rsl_rl tests."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from collections.abc import Callable
from pathlib import Path
from tensordict import TensorDict


def make_obs(num_envs: int, obs_dim: int = 8, device: str = "cpu") -> TensorDict:
    """Create an observation TensorDict with a single 'policy' key."""
    return TensorDict(
        {"policy": torch.randn(num_envs, obs_dim, device=device)},
        batch_size=[num_envs],
        device=device,
    )


def run_distributed(fn: Callable, world_size: int, tmp_path: Path, *args: object) -> list:
    """Run ``fn(rank, world_size, *args)`` in ``world_size`` gloo processes and return each rank's result."""
    mp.spawn(_init_and_run, args=(world_size, tmp_path, fn, args), nprocs=world_size, join=True)
    return [torch.load(tmp_path / f"rank{rank}.pt") for rank in range(world_size)]


def _init_and_run(rank: int, world_size: int, tmp_path: Path, fn: Callable, args: tuple) -> None:
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/pg", rank=rank, world_size=world_size)
    try:
        torch.save(fn(rank, world_size, *args), tmp_path / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()
