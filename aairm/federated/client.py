"""Federated client and parameterized demand model for FDL.

``LinearDemandModel`` is a lightweight, fully parameterized C1 surrogate used
for the federated experiments: a linear regressor over standardized demand
features trained by SGD. Its flat parameter vector is what FedAvg/FedProx
average. Keeping it numpy-only means the FDL evaluation runs anywhere, while
the federation logic (local epochs, proximal regularization, weighted
aggregation) is identical to what a deep C1 model would use.

``FederatedClient`` wraps one store's private dataset and performs ``E`` local
epochs, optionally with the FedProx proximal term that anchors local
parameters to the broadcast global parameters.

References
----------
product.tex Section "Federated Demand Learning"; FedProx (Li et al. 2020).
"""

from __future__ import annotations

import numpy as np


class LinearDemandModel:
    """Linear demand model: ``y_hat = x . w + b`` with SGD training.

    Args:
        n_features: Input feature dimension.
        seed: RNG seed for weight initialization.
    """

    def __init__(self, n_features: int, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.n_features = n_features
        self.w = rng.normal(0.0, 0.01, size=n_features)
        self.b = 0.0

    # -- parameter (de)serialization for federation -------------------
    def get_params(self) -> np.ndarray:
        """Flatten ``[w, b]`` into one parameter vector."""
        return np.concatenate([self.w, [self.b]])

    def set_params(self, params: np.ndarray) -> None:
        """Load a flat ``[w, b]`` vector into the model."""
        params = np.asarray(params, dtype=float)
        self.w = params[:-1].copy()
        self.b = float(params[-1])

    # -- training / inference -----------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.w + self.b

    def sgd_epoch(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
        global_params: np.ndarray | None = None,
        mu: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        """One epoch of mini-batch SGD on MSE, with optional FedProx term.

        The proximal term ``(mu/2)||theta - theta_global||^2`` adds a gradient
        ``mu * (theta - theta_global)`` that pulls local params toward the
        broadcast global params, limiting client drift under non-IID data.
        """
        rng = rng or np.random.default_rng(0)
        n = len(y)
        idx = rng.permutation(n)
        batch = 64
        gw = global_params[:-1] if global_params is not None else None
        gb = float(global_params[-1]) if global_params is not None else None
        for start in range(0, n, batch):
            bi = idx[start:start + batch]
            xb, yb = X[bi], y[bi]
            err = self.predict(xb) - yb
            grad_w = (xb.T @ err) / len(bi)
            grad_b = float(np.mean(err))
            if mu > 0.0 and gw is not None:
                grad_w = grad_w + mu * (self.w - gw)
                grad_b = grad_b + mu * (self.b - gb)
            self.w -= lr * grad_w
            self.b -= lr * grad_b


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error (%), the paper's FDL accuracy metric."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.sum(np.abs(y_true))
    if denom <= 0:
        return 0.0
    return 100.0 * float(np.sum(np.abs(y_true - y_pred)) / denom)


class FederatedClient:
    """One store participating in federated demand learning.

    Args:
        client_id: Store identifier.
        X: Local feature matrix (private; never leaves the client).
        y: Local demand targets (private).
        n_features: Feature dimension.
        lr: Local SGD learning rate.
        seed: RNG seed.
    """

    def __init__(
        self,
        client_id: str,
        X: np.ndarray,
        y: np.ndarray,
        n_features: int,
        lr: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.client_id = client_id
        self._X = np.asarray(X, dtype=float)
        self._y = np.asarray(y, dtype=float)
        self._lr = lr
        self._rng = np.random.default_rng(seed)
        self.model = LinearDemandModel(n_features, seed=seed)

    @property
    def n_samples(self) -> int:
        return len(self._y)

    def local_train(
        self, global_params: np.ndarray, local_epochs: int = 2, mu: float = 0.0
    ) -> np.ndarray:
        """Run ``E`` local epochs starting from the broadcast global params.

        Args:
            global_params: Current global parameter vector ``theta^(u)``.
            local_epochs: Number of local SGD epochs ``E``.
            mu: FedProx proximal coefficient (0 = FedAvg).

        Returns:
            Updated local parameter vector ``theta_k^(u+1)``.
        """
        self.model.set_params(global_params)
        for _ in range(local_epochs):
            self.model.sgd_epoch(
                self._X, self._y, lr=self._lr,
                global_params=global_params if mu > 0 else None,
                mu=mu, rng=self._rng,
            )
        return self.model.get_params()

    def evaluate(self, params: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray) -> float:
        """Held-out WAPE for the given parameter vector."""
        self.model.set_params(params)
        return wape(y_eval, self.model.predict(X_eval))
