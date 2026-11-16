# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2020 Preferred Networks, Inc.


from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape: int | tuple[int, ...] | list[int], eps: float = 1e-2, until: int | None = None) -> None:
        """Initialize EmpiricalNormalization module.

        .. note:: The normalization parameters are computed over the whole batch, not for each environment separately.

        Args:
            shape: Shape of input values except batch axis.
            eps: Small value for stability.
            until: If this arg is specified, the module learns input values until the sum of batch sizes exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        # Statistics at the last sync and the local moments since then, for sync() in a process group
        self.register_buffer("_synced_mean", self._mean.clone(), persistent=False)
        self.register_buffer("_synced_var", self._var.clone(), persistent=False)
        self.register_buffer("_synced_count", self.count.double(), persistent=False)
        self.register_buffer("_batch_count", torch.tensor(0.0, dtype=torch.float64), persistent=False)
        self.register_buffer("_batch_sum", torch.zeros_like(self._mean, dtype=torch.float64), persistent=False)
        self.register_buffer("_batch_sum_sq", torch.zeros_like(self._mean, dtype=torch.float64), persistent=False)

    @property
    def mean(self) -> torch.Tensor:
        """Return the current running mean."""
        return self._mean.squeeze(0).clone()  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        """Return the current running standard deviation."""
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize mean and variance of values based on empirical values."""
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        """Learn input values without computing the output values of them.

        In a process group, the batch is also added to the local moments that :meth:`sync` reduces across processes.
        """
        if not self.training:
            return
        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)
        if dist.is_available() and dist.is_initialized():
            x64 = x.double()
            self._batch_count += count_x
            self._batch_sum += x64.sum(dim=0, keepdim=True)
            self._batch_sum_sq += x64.square().sum(dim=0, keepdim=True)

    @torch.jit.unused
    def sync(self) -> None:
        """Merge the batches every process has seen since the last sync into the statistics, the same on all processes.

        Every process of the default group must call it at the same point. The statistics restart from those of the
        last sync (or load), so the local updates in between only serve the forward passes until this call.
        """
        with torch.inference_mode():
            moments = torch.cat((self._batch_count.view(1), self._batch_sum.flatten(), self._batch_sum_sq.flatten()))
            dist.all_reduce(moments)
            count_b, sum_b, sum_sq_b = moments[0], *moments[1:].view(2, *self._batch_sum.shape)
            count = self._synced_count + count_b
            rate = count_b / count.clamp_min(1)
            mean_b = sum_b / count_b.clamp_min(1)
            var_b = (sum_sq_b / count_b.clamp_min(1) - mean_b.square()).clamp_min(0.0)
            mean0, var0 = self._synced_mean.double(), self._synced_var.double()
            delta_mean = mean_b - mean0
            mean = mean0 + rate * delta_mean
            var = var0 + rate * (var_b - var0 + delta_mean * (mean_b - mean))
            self._mean.copy_(mean)
            self._var.copy_(var)
            self._std.copy_(var.sqrt())
            self.count.copy_(count.round())
            self._snapshot()

    @torch.jit.unused
    def _snapshot(self) -> None:
        """Record the statistics :meth:`sync` restarts from and clear the local moments."""
        self._synced_mean.copy_(self._mean)
        self._synced_var.copy_(self._var)
        self._synced_count.copy_(self.count)
        self._batch_count.zero_()
        self._batch_sum.zero_()
        self._batch_sum_sq.zero_()

    def _load_from_state_dict(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        super()._load_from_state_dict(*args, **kwargs)
        with torch.no_grad():
            self._snapshot()

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """De-normalize values based on empirical values."""
        return y * (self._std + self.eps) + self._mean


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize the scale of the
    rewards so that the value function can learn quickly. We did this by dividing the rewards by a running estimate of
    the standard deviation of the sum of discounted rewards.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...] | list[int],
        eps: float = 1e-2,
        gamma: float = 0.99,
        until: int | None = None,
    ) -> None:
        """Initialize discounted-reward normalization with running moments."""
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = _DiscountedAverage(gamma)

    def forward(self, rew: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using the running std of discounted returns."""
        if self.training:
            # Update discounted rewards
            avg = self.disc_avg.update(rew)
            # Update moments from discounted rewards
            self.emp_norm.update(avg)

        # Normalize rewards with the empirical std
        if self.emp_norm._std > 0:  # type: ignore
            return rew / self.emp_norm._std  # type: ignore
        else:
            return rew


class _DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t
    """

    def __init__(self, gamma: float) -> None:
        """Initialize discounted accumulation with a fixed discount factor."""
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        """Update and return the discounted running average."""
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg
