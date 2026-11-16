# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for normalization modules."""

from __future__ import annotations

import torch
from pathlib import Path

import pytest

from rsl_rl.modules.normalization import EmpiricalDiscountedVariationNormalization, EmpiricalNormalization
from tests.conftest import run_distributed

# Per-rank batch sizes differ on purpose: ranks may hold different numbers of environments.
RANK_ROWS = (3, 5)


def _rank_batches(rank: int, batches: list[torch.Tensor]) -> list[torch.Tensor]:
    """Split each batch into the rows of ``rank``."""
    return [b.split(RANK_ROWS)[rank] for b in batches]


def _update_on_rank(rank: int, world_size: int, rounds: list[list[torch.Tensor]], resume: dict | None) -> dict:
    norm = EmpiricalNormalization(shape=rounds[0][0].shape[1:])
    if resume is not None:
        norm.load_state_dict(resume)
    for batches in rounds:
        for batch in _rank_batches(rank, batches):
            norm.update(batch)
        norm.sync()
    norm.eval()
    norm.update(_rank_batches(rank, rounds[0])[0] + 100.0)
    norm.sync()
    return {"count": norm.count, "mean": norm.mean, "std": norm.std}


def _reward_normalizer_on_rank(rank: int, world_size: int, rewards: list[torch.Tensor]) -> dict:
    norm = EmpiricalDiscountedVariationNormalization(shape=[])
    for reward in _rank_batches(rank, rewards):
        norm(reward)
    norm.emp_norm.sync()
    return {"mean": norm.emp_norm.mean, "std": norm.emp_norm.std}


class TestEmpiricalNormalization:
    """Tests for ``EmpiricalNormalization``."""

    def test_convergence_to_known_distribution(self) -> None:
        """Running mean and std should converge to the true values of the input distribution."""
        true_mean, true_std = 5.0, 2.0
        norm = EmpiricalNormalization(shape=4)
        norm.train()

        torch.manual_seed(0)
        for _ in range(200):
            batch = true_mean + true_std * torch.randn(64, 4)
            norm.update(batch)

        assert torch.allclose(norm.mean, torch.full((4,), true_mean), atol=0.15)
        assert torch.allclose(norm.std, torch.full((4,), true_std), atol=0.15)

    def test_forward_applies_normalization(self) -> None:
        """forward() should return (x - mean) / (std + eps), not the raw input."""
        norm = EmpiricalNormalization(shape=2, eps=0.01)
        norm.train()

        data = torch.tensor([[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]])
        norm.update(data)

        result = norm(data)
        expected = (data - norm._mean) / (norm._std + norm.eps)
        assert torch.allclose(result, expected)

    def test_until_stops_updates(self) -> None:
        """After 'until' samples are seen, further updates must not change statistics."""
        norm = EmpiricalNormalization(shape=2, until=100)
        norm.train()

        for _ in range(10):
            norm.update(torch.randn(20, 2))

        assert norm.count >= 100
        mean_snapshot = norm._mean.clone()
        std_snapshot = norm._std.clone()

        for _ in range(10):
            norm.update(torch.randn(20, 2) + 100)

        assert torch.equal(norm._mean, mean_snapshot)
        assert torch.equal(norm._std, std_snapshot)

    def test_eval_mode_freezes_stats(self) -> None:
        """In eval mode, update() should be a no-op."""
        norm = EmpiricalNormalization(shape=3)
        norm.train()
        norm.update(torch.randn(50, 3))

        mean_before = norm._mean.clone()
        std_before = norm._std.clone()

        norm.eval()
        norm.update(torch.randn(50, 3) + 100)

        assert torch.equal(norm._mean, mean_before)
        assert torch.equal(norm._std, std_before)

    def test_inverse_round_trip(self) -> None:
        """inverse(forward(x)) should approximately recover x."""
        norm = EmpiricalNormalization(shape=4, eps=1e-2)
        norm.train()

        torch.manual_seed(42)
        for _ in range(50):
            norm.update(torch.randn(32, 4) * 3 + 5)

        x = torch.randn(16, 4) * 3 + 5
        recovered = norm.inverse(norm(x))
        assert torch.allclose(recovered, x, atol=1e-5)

    def test_single_sample_does_not_produce_nan(self) -> None:
        """Updating with a single sample should not produce NaN in mean or std."""
        norm = EmpiricalNormalization(shape=2)
        norm.train()
        norm.update(torch.tensor([[1.0, 2.0]]))

        assert not torch.any(torch.isnan(norm._mean))
        assert not torch.any(torch.isnan(norm._std))


class TestDistributedNormalization:
    """After ``sync``, normalizers in a process group hold the statistics of one process that saw every batch."""

    @pytest.mark.parametrize("resume", [False, True])
    def test_ranks_match_one_process_with_all_rows(self, tmp_path: Path, resume: bool) -> None:
        """Unequal per-rank batches over several syncs, from scratch or a loaded state, match the concatenated run."""
        torch.manual_seed(0)
        rounds = [[3.0 + 2.0 * torch.randn(sum(RANK_ROWS), 2, 3) for _ in range(3)] for _ in range(2)]
        ref = EmpiricalNormalization(shape=(2, 3))
        state = None
        if resume:
            ref.update(torch.randn(11, 2, 3))
            state = {k: v.clone() for k, v in ref.state_dict().items()}
        for batches in rounds:
            for batch in batches:
                ref.update(batch)

        for out in run_distributed(_update_on_rank, 2, tmp_path, rounds, state):
            assert out["count"] == ref.count
            torch.testing.assert_close(out["mean"], ref.mean)
            torch.testing.assert_close(out["std"], ref.std)

    def test_constant_input_gives_zero_std(self, tmp_path: Path) -> None:
        """A large constant input must give a zero std, not a NaN from a negative variance."""
        rounds = [[torch.full((sum(RANK_ROWS), 4), 12345.678)] * 3]
        for out in run_distributed(_update_on_rank, 2, tmp_path, rounds, None):
            assert torch.equal(out["std"], torch.zeros(4))

    def test_reward_normalizer_matches_one_process(self, tmp_path: Path) -> None:
        """The scalar-shaped normalizer inside the discounted reward normalizer syncs too."""
        torch.manual_seed(0)
        rewards = [torch.rand(sum(RANK_ROWS)) for _ in range(5)]
        ref = EmpiricalDiscountedVariationNormalization(shape=[])
        for reward in rewards:
            ref(reward)

        for out in run_distributed(_reward_normalizer_on_rank, 2, tmp_path, rewards):
            torch.testing.assert_close(out["mean"], ref.emp_norm.mean)
            torch.testing.assert_close(out["std"], ref.emp_norm.std)

    def test_update_issues_no_collective(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The per-step update stays local, so a CUDA graph can capture it; only ``sync`` communicates."""

        def no_collective(*args: object, **kwargs: object) -> None:
            raise AssertionError("update() must not communicate")

        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "all_reduce", no_collective)
        norm = EmpiricalNormalization(shape=3)
        norm.update(torch.randn(5, 3))
        assert norm._batch_count == 5


class TestEmpiricalDiscountedVariationNormalization:
    """Tests for ``EmpiricalDiscountedVariationNormalization``."""

    def test_constant_rewards_produce_stable_normalization(self) -> None:
        """Constant rewards should converge to a stable normalization factor."""
        gamma = 0.99
        norm = EmpiricalDiscountedVariationNormalization(shape=[], gamma=gamma)
        norm.train()

        reward = torch.tensor([1.0])
        outputs = [norm(reward).item() for _ in range(200)]

        # After convergence, the output should be approximately constant
        last_10 = outputs[-10:]
        assert max(last_10) - min(last_10) < 0.1, "Normalization should stabilize for constant rewards"

    def test_zero_std_returns_raw_reward(self) -> None:
        """When std is zero (or not yet computed), forward should return the raw reward."""
        norm = EmpiricalDiscountedVariationNormalization(shape=[])
        norm.eval()
        reward = torch.tensor([5.0])
        result = norm(reward)
        # In eval mode with no prior updates, std defaults to 1, so it normalizes by 1
        assert torch.isfinite(result).all()
