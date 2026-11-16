# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for broadcasting the PPO models across processes."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path

from rsl_rl.algorithms.ppo import PPO
from tests.algorithms.test_ppo import _make_actor, _make_critic
from tests.conftest import make_obs


def _run(rank: int, world_size: int, tmp_path: Path) -> None:
    """Build differently seeded models on one rank, broadcast them and save the result."""
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/pg", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(rank)
        obs = make_obs(4)
        obs_groups = {"actor": ["policy"], "critic": ["policy"]}
        actor = _make_actor(obs, obs_groups, obs_normalization=True)
        critic = _make_critic(obs, obs_groups, obs_normalization=True)
        actor.update_normalization(make_obs(4))
        ppo = PPO(
            actor, critic, None, multi_gpu_cfg={"global_rank": rank, "local_rank": rank, "world_size": world_size}
        )
        ppo.broadcast_parameters()
        state = {f"{n}.{k}": v for n, m in (("actor", actor), ("critic", critic)) for k, v in m.state_dict().items()}
        torch.save(state, tmp_path / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_broadcast_parameters_copies_rank0_models(tmp_path: Path) -> None:
    """Every rank holds rank 0's parameters and normalizer buffers after the broadcast."""
    mp.spawn(_run, args=(2, tmp_path), nprocs=2, join=True)
    rank0, rank1 = (torch.load(tmp_path / f"rank{r}.pt") for r in range(2))
    assert rank0.keys() == rank1.keys()
    for key in rank0:
        torch.testing.assert_close(rank1[key], rank0[key], rtol=0, atol=0, msg=key)
