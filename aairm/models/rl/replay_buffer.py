"""Experience replay buffer for the PPO / TD learning pipeline.

Stores transition tuples ``(s_t, a_t, r_t, s_{t+1}, done)`` collected
by the simulation environment.  Used by the Learning Agent (A3) to batch
policy gradient updates.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class ReplayBuffer:
    """Fixed-capacity FIFO replay buffer.

    Args:
        capacity: Maximum number of transitions to store.
        obs_dim: Observation space dimension.
        act_dim: Action space dimension.
    """

    def __init__(
        self,
        capacity: int = 10_000,
        obs_dim: int = 5,
        act_dim: int = 1,
    ) -> None:
        self._capacity = capacity
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._buffer: deque[dict[str, Any]] = deque(maxlen=capacity)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single transition to the buffer.

        Args:
            obs: Current observation of shape ``(obs_dim,)``.
            action: Action taken of shape ``(act_dim,)``.
            reward: Scalar reward received.
            next_obs: Next observation of shape ``(obs_dim,)``.
            done: Whether the episode terminated after this transition.
        """
        self._buffer.append(
            {
                "obs": np.asarray(obs, dtype=np.float32),
                "action": np.asarray(action, dtype=np.float32),
                "reward": float(reward),
                "next_obs": np.asarray(next_obs, dtype=np.float32),
                "done": bool(done),
            }
        )

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a random mini-batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Dict of stacked arrays:
            ``{obs, action, reward, next_obs, done}``
            each of shape ``(batch_size, dim)``.
        """
        indices = np.random.choice(len(self._buffer), size=batch_size, replace=False)
        batch = [self._buffer[i] for i in indices]
        return {
            "obs":      np.stack([b["obs"]      for b in batch]),
            "action":   np.stack([b["action"]   for b in batch]),
            "reward":   np.array([b["reward"]   for b in batch], dtype=np.float32),
            "next_obs": np.stack([b["next_obs"] for b in batch]),
            "done":     np.array([b["done"]     for b in batch], dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_ready(self) -> bool:
        """True when the buffer contains at least ``capacity // 10`` transitions."""
        return len(self._buffer) >= max(64, self._capacity // 10)
