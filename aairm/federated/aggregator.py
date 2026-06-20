"""Federated parameter aggregation rules: FedAvg and FedProx.

Both operate on flat parameter vectors so they are agnostic to the underlying
demand model. FedProx differs from FedAvg only in the *client* objective (it
adds a proximal term ``(mu/2)||theta_k - theta_global||^2``); the server-side
aggregation is the same data-weighted average. The proximal term is applied
inside :class:`aairm.federated.client.FederatedClient`, so the two server
functions below share the weighted-average implementation and exist as
distinct names for clarity and traceability to the paper's two regimes.

References
----------
McMahan et al. (2017) FedAvg; Li et al. (2020) FedProx; product.tex Eq. (FedAvg).
"""

from __future__ import annotations

import numpy as np


def _weighted_average(
    client_params: list[np.ndarray], client_sizes: list[int]
) -> np.ndarray:
    """Data-weighted average of client parameter vectors (Eq. FedAvg).

    .. math:: \\theta^{(u+1)} = \\sum_k \\frac{|D_k|}{\\sum_{k'} |D_{k'}|} \\theta_k^{(u+1)}

    Args:
        client_params: list of flat parameter vectors, one per participating client.
        client_sizes: list of local dataset sizes ``|D_k|`` (same order).

    Returns:
        Aggregated global parameter vector.
    """
    if not client_params:
        raise ValueError("No client parameters to aggregate.")
    total = float(sum(client_sizes))
    if total <= 0:
        raise ValueError("Sum of client sizes must be positive.")
    stacked = np.stack(client_params, axis=0)
    weights = np.asarray(client_sizes, dtype=float) / total
    return np.tensordot(weights, stacked, axes=(0, 0))


def fedavg(client_params: list[np.ndarray], client_sizes: list[int]) -> np.ndarray:
    """FedAvg server aggregation (data-weighted parameter average)."""
    return _weighted_average(client_params, client_sizes)


def fedprox(client_params: list[np.ndarray], client_sizes: list[int]) -> np.ndarray:
    """FedProx server aggregation.

    Identical to FedAvg on the server; client drift is limited by the proximal
    term applied during local training (see ``FederatedClient.local_train``).
    """
    return _weighted_average(client_params, client_sizes)
