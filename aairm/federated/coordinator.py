"""FDL coordinator — orchestrates communication rounds and evaluation.

Implements the four training regimes compared in product.tex Table 9:
centralized (pooled), FedAvg, FedProx, and local-only, over non-IID store
partitions built with a Dirichlet category mix (concentration ``beta``). The
coordinator measures held-out WAPE per round (the convergence curve of
Figure fig:fl-convergence) and, when a Blockchain Trust Ledger is supplied,
anchors each round's aggregated parameter digest so any dispute about which
model version produced a forecast can be resolved against the ledger.

References
----------
product.tex Sections "Federated Demand Learning" and "Federated Demand
Learning Evaluation"; Table 9; Figure fig:fl-convergence.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from aairm.federated.aggregator import fedavg, fedprox
from aairm.federated.client import FederatedClient, LinearDemandModel, wape
from aairm.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FDLResult:
    """Outcome of one FDL regime."""

    regime: str
    final_wape: float
    convergence: list[tuple[int, float]] = field(default_factory=list)
    mb_per_round: float = 0.0
    rounds: int = 0


class FDLCoordinator:
    """Coordinates federated demand learning across store clients.

    Args:
        n_features: Feature dimension of the demand model.
        rounds: Communication rounds ``U`` (paper: 60).
        local_epochs: Local epochs ``E`` per round (paper: 2).
        mu: FedProx proximal coefficient (paper: 0.01).
        lr: Local SGD learning rate.
        seed: RNG seed.
        ledger: Optional object with ``anchor(event_type, payload)`` to anchor
            per-round parameter digests (e.g. the Blockchain Trust Ledger).
    """

    def __init__(
        self,
        n_features: int,
        rounds: int = 60,
        local_epochs: int = 2,
        mu: float = 0.01,
        lr: float = 0.05,
        seed: int = 42,
        ledger: Any | None = None,
    ) -> None:
        self.n_features = n_features
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.mu = mu
        self.lr = lr
        self.seed = seed
        self.ledger = ledger
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    def partition_dirichlet(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categories: np.ndarray,
        n_clients: int = 8,
        beta: float = 0.5,
    ) -> list[FederatedClient]:
        """Split samples across clients with non-IID category mixes.

        Each client's category proportions are drawn from a Dirichlet(beta)
        distribution, so stores share broad offerings but differ in local
        demand proportions (paper: moderate heterogeneity at beta=0.5).
        """
        cats = np.unique(categories)
        # client_id -> list of sample indices
        assignment: list[list[int]] = [[] for _ in range(n_clients)]
        for c in cats:
            c_idx = np.where(categories == c)[0]
            self._rng.shuffle(c_idx)
            props = self._rng.dirichlet([beta] * n_clients)
            cuts = (np.cumsum(props) * len(c_idx)).astype(int)[:-1]
            for k, chunk in enumerate(np.split(c_idx, cuts)):
                assignment[k].extend(chunk.tolist())

        clients: list[FederatedClient] = []
        for k in range(n_clients):
            idx = np.array(assignment[k], dtype=int)
            if idx.size == 0:
                idx = np.array([int(self._rng.integers(len(y)))])
            clients.append(
                FederatedClient(
                    client_id=f"store_{k}", X=X[idx], y=y[idx],
                    n_features=self.n_features, lr=self.lr, seed=self.seed + k,
                )
            )
        return clients

    # ------------------------------------------------------------------
    def fit_centralized(
        self, X: np.ndarray, y: np.ndarray, X_eval: np.ndarray, y_eval: np.ndarray
    ) -> FDLResult:
        """Upper bound: pool all client data and train one model."""
        model = LinearDemandModel(self.n_features, seed=self.seed)
        rng = np.random.default_rng(self.seed)
        trace = []
        for u in range(1, self.rounds + 1):
            for _ in range(self.local_epochs):
                model.sgd_epoch(X, y, lr=self.lr, rng=rng)
            trace.append((u, wape(y_eval, model.predict(X_eval))))
        return FDLResult("centralized", trace[-1][1], trace, rounds=self.rounds)

    def fit_local_only(
        self, clients: list[FederatedClient], X_eval: np.ndarray, y_eval: np.ndarray
    ) -> FDLResult:
        """Lower bound: each store trains only on its own data; report mean WAPE."""
        wapes = []
        for client in clients:
            params = client.model.get_params()
            params = client.local_train(params, local_epochs=self.local_epochs * self.rounds, mu=0.0)
            wapes.append(client.evaluate(params, X_eval, y_eval))
        return FDLResult("local_only", float(np.mean(wapes)), rounds=self.rounds)

    def fit_federated(
        self,
        clients: list[FederatedClient],
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        regime: str = "fedavg",
        param_bytes: int = 4,
    ) -> FDLResult:
        """Run FedAvg or FedProx for ``rounds`` communication rounds.

        Args:
            clients: store clients (private data each).
            X_eval, y_eval: shared held-out evaluation set.
            regime: ``"fedavg"`` or ``"fedprox"``.
            param_bytes: bytes per parameter (for the communication estimate).

        Returns:
            :class:`FDLResult` with per-round held-out WAPE and MB/round/store.
        """
        if regime not in ("fedavg", "fedprox"):
            raise ValueError(f"regime must be 'fedavg' or 'fedprox'; got {regime}")
        aggregate = fedavg if regime == "fedavg" else fedprox
        mu = self.mu if regime == "fedprox" else 0.0

        global_params = LinearDemandModel(self.n_features, seed=self.seed).get_params()
        eval_model = LinearDemandModel(self.n_features, seed=self.seed)
        trace: list[tuple[int, float]] = []

        for u in range(1, self.rounds + 1):
            updates, sizes = [], []
            for client in clients:
                updates.append(client.local_train(global_params, self.local_epochs, mu))
                sizes.append(client.n_samples)
            global_params = aggregate(updates, sizes)

            if self.ledger is not None:
                digest = hashlib.sha256(global_params.tobytes()).hexdigest()
                try:
                    self.ledger.anchor("fdl_round", {"round": u, "param_digest": digest})
                except Exception:  # noqa: BLE001
                    pass

            eval_model.set_params(global_params)
            trace.append((u, wape(y_eval, eval_model.predict(X_eval))))

        n_params = len(global_params)
        mb_per_round = n_params * param_bytes / (1024 * 1024)
        return FDLResult(regime, trace[-1][1], trace, mb_per_round=mb_per_round, rounds=self.rounds)
