"""Value function approximator for the TD update (Eq. 7).

A simple two-layer MLP V_φ(s) used by the Learning Agent (A3) to compute
the temporal-difference loss L_TD(φ) = (r_t + γ·V_φ(s_{t+1}) − V_φ(s_t))².

References
----------
Paper Eq. 7; Repo Guide Section 5.8.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_OBS_DIM = 5
_HIDDEN = 64


class ValueNetwork:
    """Lightweight MLP value function approximator.

    Uses NumPy only (no PyTorch dependency) for maximum compatibility.
    Weights are trained via gradient descent on the TD loss.

    Args:
        obs_dim: Observation dimension (default 5).
        hidden_size: Hidden layer size (default 64).
        learning_rate: Gradient descent step size (default 1e-3).
    """

    def __init__(
        self,
        obs_dim: int = _OBS_DIM,
        hidden_size: int = _HIDDEN,
        learning_rate: float = 1e-3,
    ) -> None:
        rng = np.random.default_rng(42)
        # Xavier initialisation
        scale1 = np.sqrt(2.0 / (obs_dim + hidden_size))
        scale2 = np.sqrt(2.0 / (hidden_size + 1))
        self._W1 = rng.normal(0, scale1, (obs_dim, hidden_size))
        self._b1 = np.zeros(hidden_size)
        self._W2 = rng.normal(0, scale2, (hidden_size, 1))
        self._b2 = np.zeros(1)
        self._lr = learning_rate

    def __call__(self, obs: np.ndarray) -> float:
        """Compute V(s) for a single observation.

        Args:
            obs: Observation vector of shape ``(obs_dim,)``.

        Returns:
            Scalar value estimate.
        """
        h = np.tanh(obs @ self._W1 + self._b1)
        v = h @ self._W2 + self._b2
        return float(v[0])

    def update(
        self,
        obs: np.ndarray,
        target: float,
    ) -> float:
        """One gradient descent step on the MSE loss (V(s) - target)².

        Args:
            obs: Current state observation.
            target: TD target r + γ·V(s').

        Returns:
            Loss before the update.
        """
        h = np.tanh(obs @ self._W1 + self._b1)
        v = float((h @ self._W2 + self._b2)[0])
        loss = (v - target) ** 2

        # Backprop
        dv = 2.0 * (v - target)
        dW2 = h[:, None] * dv
        db2 = np.array([dv])
        dh = (self._W2.ravel() * dv) * (1 - h**2)
        dW1 = obs[:, None] * dh[None, :]
        db1 = dh

        self._W1 -= self._lr * dW1
        self._b1 -= self._lr * db1
        self._W2 -= self._lr * dW2
        self._b2 -= self._lr * db2
        return float(loss)
