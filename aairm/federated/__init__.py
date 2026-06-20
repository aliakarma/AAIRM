"""Federated Demand Learning (FDL) layer.

Trains the C1 demand-forecasting model across store clients without
centralizing raw sales data (product.tex Section "Federated Demand Learning").
Implements federated averaging (FedAvg, McMahan et al. 2017) and its proximal
variant (FedProx, Li et al. 2020) over a parameterized demand model, plus a
coordinator that builds non-IID store partitions, runs communication rounds,
measures held-out WAPE, and anchors each round's parameter digest in the
Blockchain Trust Ledger.

Public API:
    LinearDemandModel  — parameterized C1 surrogate trainable per client.
    FederatedClient    — one store; holds private data, runs local epochs.
    fedavg, fedprox    — parameter aggregation rules.
    FDLCoordinator     — orchestrates rounds and evaluation (Table 9).
"""

from aairm.federated.aggregator import fedavg, fedprox
from aairm.federated.client import FederatedClient, LinearDemandModel
from aairm.federated.coordinator import FDLCoordinator

__all__ = [
    "LinearDemandModel",
    "FederatedClient",
    "fedavg",
    "fedprox",
    "FDLCoordinator",
]
