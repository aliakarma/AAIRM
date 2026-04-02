"""Global random seed management for full experiment reproducibility.

Call :func:`set_global_seed` once at the start of every experiment script
and at the top of every stochastic test module.  The default seed value
(42) is the value used in all paper experiments.

References
----------
Paper Section 5.1: "All experiments were conducted with seed = 42."
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int = 42) -> None:
    """Set random seeds for Python, NumPy, PyTorch (if available), and the OS.

    This function is the single call required to make any AAIRM experiment
    fully deterministic.  It must be the first substantive call in any
    script that produces stochastic outputs.

    Args:
        seed: Integer seed value.  Paper uses ``seed=42``.  Any non-negative
            integer produces a valid, fully reproducible run.

    Examples:
        >>> set_global_seed(42)
        >>> import numpy as np
        >>> np.random.rand()  # always 0.374... with seed=42
        0.3745401188473625
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch — imported conditionally so the package works without GPU deps
    try:
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Make cuDNN deterministic.  Small performance penalty; mandatory for
        # full reproducibility on GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed; simulation-only mode is fine
