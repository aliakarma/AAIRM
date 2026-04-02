"""Stochastic Demand Generator.

Implements the data-generating process (DGP) described in Section 5.1
of the paper:

    y_{i,t} = base_i · seasonal(t, cat_i) · promo_uplift(t, i) · ε_{i,t}

where:
    base_i         ~ Uniform(20, 200) units/day — fixed per SKU
    seasonal(t)    = 1 + amp * sin(2π·(t % 7)/7)   [weekly]
                   + holiday_spike(t)               [Saudi retail calendar]
    promo_uplift   = Uniform(1.5, 2.5) on 10% of days, else 1.0
    ε_{i,t}        ~ LogNormal(0, σ_i), σ_i ~ Uniform(0.05, 0.25)

Holiday calendar reflects the Saudi retail context of the paper's
institutional setting (Islamic University of Madinah, Saudi Arabia).

References
----------
Paper Section 5.1; demand_params_seed.json.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from aairm.simulation.sku_catalog import SKUCatalog

# ---------------------------------------------------------------------------
# Holiday calendar — Saudi retail context (approximate day-of-year)
# ---------------------------------------------------------------------------

# Each entry: (start_doy, duration_days, uplift_multiplier, name)
_HOLIDAYS: list[tuple[int, int, float, str]] = [
    (1, 1, 1.2, "New_Year"),
    (100, 3, 2.0, "Eid_Al_Fitr"),
    (175, 3, 1.8, "Eid_Al_Adha"),
    (60, 30, 1.3, "Ramadan"),
    (272, 2, 1.4, "National_Day"),
]

# Pre-compute holiday day-of-year lookup for speed
_HOLIDAY_UPLIFT: dict[int, float] = {}
for _start, _dur, _up, _name in _HOLIDAYS:
    for _d in range(_dur):
        _doy = ((_start + _d - 1) % 365) + 1
        # Take max if multiple holidays overlap
        _HOLIDAY_UPLIFT[_doy] = max(_HOLIDAY_UPLIFT.get(_doy, 1.0), _up)


class DemandGenerator:
    """Generates synthetic daily demand for all SKUs in a catalog.

    The full two-year time series is generated at initialisation time
    for speed (vectorised NumPy).  Individual days are then returned via
    :meth:`get_demand` and :meth:`get_history`.

    Args:
        catalog: :class:`~aairm.simulation.sku_catalog.SKUCatalog` instance.
        n_days: Total simulation horizon in days (default 730 = 2 years).
        promo_fraction: Fraction of days that are promotional (default 0.10).
        promo_uplift_range: (min, max) uplift on promo days (default (1.5, 2.5)).
        seed: Random seed.

    Examples:
        >>> catalog = SKUCatalog(n_skus=10, seed=42)
        >>> gen = DemandGenerator(catalog, n_days=30, seed=42)
        >>> demand = gen.get_demand("GRO-0001", day=0)
        >>> isinstance(demand, float)
        True
    """

    def __init__(
        self,
        catalog: SKUCatalog,
        n_days: int = 730,
        promo_fraction: float = 0.10,
        promo_uplift_range: tuple[float, float] = (1.5, 2.5),
        seed: int = 42,
    ) -> None:
        self._catalog = catalog
        self._n_days = n_days
        self._rng = np.random.default_rng(seed)
        self._promo_frac = promo_fraction
        self._promo_range = promo_uplift_range

        # Pre-generate the full demand matrix: shape (n_skus, n_days)
        self._sku_ids = catalog.sku_ids
        self._demand_matrix = self._generate_all()

        # Trend signals: day → list of {product_id, trend_score, category, ...}
        self._trend_cache: dict[int, list[dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_demand(self, sku_id: str, day: int) -> float:
        """Return realised demand for one SKU on one day.

        Args:
            sku_id: SKU identifier.
            day: Simulation day index (0-based).

        Returns:
            Demand realisation (non-negative float).
        """
        idx = self._idx_of(sku_id)
        if idx is None or day >= self._n_days:
            return 0.0
        expected = float(self._demand_matrix[idx, day])
        if expected <= 0.0:
            return 0.0

        # Add online stochasticity so trajectories are not overly smooth.
        if expected < 50.0:
            sampled = float(self._rng.poisson(lam=max(expected, 0.1)))
        else:
            sampled = float(self._rng.normal(loc=expected, scale=max(0.15 * expected, 1.0)))
        return float(max(0.0, round(sampled, 2)))

    def get_history(self, sku_id: str, n_days: int, up_to_day: int) -> np.ndarray:
        """Return the demand history for one SKU up to (not including) a day.

        Args:
            sku_id: SKU identifier.
            n_days: Number of history days to return.
            up_to_day: The current simulation day (exclusive upper bound).

        Returns:
            NumPy array of shape ``(min(n_days, up_to_day),)``.
        """
        idx = self._idx_of(sku_id)
        if idx is None:
            return np.zeros(n_days)
        start = max(0, up_to_day - n_days)
        end = min(up_to_day, self._n_days)
        series = self._demand_matrix[idx, start:end]
        # Pad with zeros if fewer than n_days available
        if len(series) < n_days:
            series = np.concatenate([np.zeros(n_days - len(series)), series])
        return series.copy()

    def get_demand_stats(self, sku_id: str, up_to_day: int) -> dict[str, float]:
        """Return rolling demand mean and std for ROP computation.

        Uses the last 30 days (or all available) of history.

        Args:
            sku_id: SKU identifier.
            up_to_day: Current simulation day.

        Returns:
            Dict with ``mu_d`` and ``sigma_d``.
        """
        history = self.get_history(sku_id, 30, up_to_day)
        nonzero = history[history > 0]
        if len(nonzero) == 0:
            rec = self._catalog.get(sku_id)
            mu = float(rec.base_demand_daily) if rec else 10.0
            return {"mu_d": mu, "sigma_d": mu * 0.15}
        return {
            "mu_d": float(np.mean(nonzero)),
            "sigma_d": float(np.std(nonzero) + 1e-8),
        }

    def get_trend_signals(self, day: int) -> list[dict[str, Any]]:
        """Return simulated trend signals for day.

        Trend scores are derived from promotional uplift state and
        seasonality peaks, modelling what a real trend API would return.

        Args:
            day: Simulation day index.

        Returns:
            List of trend signal dicts.
        """
        if day in self._trend_cache:
            return self._trend_cache[day]

        doy = (day % 365) + 1
        holiday_boost = _HOLIDAY_UPLIFT.get(doy, 1.0)
        signals = []

        for sku_id in self._sku_ids[:50]:  # top-50 as external trending products
            rec = self._catalog.get(sku_id)
            if rec is None:
                continue
            base_score = 0.40 + self._rng.random() * 0.40
            score = float(np.clip(base_score * holiday_boost * 0.8, 0.0, 1.0))
            signals.append(
                {
                    "product_id": sku_id,
                    "product_name": f"{rec.category.title()} Item {sku_id}",
                    "category": rec.category,
                    "trend_score": round(score, 4),
                    "source": "simulated_trend_api",
                }
            )

        self._trend_cache[day] = signals
        return signals

    # ------------------------------------------------------------------
    # Vectorised DGP
    # ------------------------------------------------------------------

    def _generate_all(self) -> np.ndarray:
        """Generate the full (n_skus × n_days) demand matrix.

        Returns:
            Non-negative float array of shape ``(n_skus, n_days)``.
        """
        n_skus = len(self._sku_ids)
        n_days = self._n_days

        # --- Base demand (n_skus,) ---
        base = np.array(
            [self._catalog[s].base_demand_daily for s in self._sku_ids],
            dtype=float,
        )

        # --- Seasonality: weekly sinusoidal + category amplitude ---
        amp = np.array(
            [self._catalog[s].seasonality_amp for s in self._sku_ids],
            dtype=float,
        )  # shape (n_skus,)
        days_idx = np.arange(n_days, dtype=float)
        weekly = np.sin(2.0 * np.pi * days_idx / 7.0)  # shape (n_days,)
        seasonal = 1.0 + amp[:, None] * weekly[None, :]  # (n_skus, n_days)

        # --- Holiday uplift ---
        holiday = np.ones(n_days)
        for d in range(n_days):
            doy = (d % 365) + 1
            holiday[d] = _HOLIDAY_UPLIFT.get(doy, 1.0)
        seasonal = seasonal * holiday[None, :]

        # --- Promotional uplift: 10% of days, random per SKU ---
        promo_days = self._rng.random((n_skus, n_days)) < self._promo_frac
        promo_mult = self._rng.uniform(
            self._promo_range[0], self._promo_range[1], size=(n_skus, n_days)
        )
        promo = np.where(promo_days, promo_mult, 1.0)

        # --- Multiplicative log-normal noise ---
        sigma = np.array(
            [self._catalog[s].demand_sigma_frac for s in self._sku_ids],
            dtype=float,
        )
        noise = self._rng.lognormal(
            mean=0.0,
            sigma=sigma[:, None] * np.ones((1, n_days)),
        )

        # --- Combine ---
        demand = base[:, None] * seasonal * promo * noise
        return np.maximum(0.0, np.round(demand, 2))

    def _idx_of(self, sku_id: str) -> int | None:
        """Return the row index of a SKU in the demand matrix."""
        try:
            return self._sku_ids.index(sku_id)
        except ValueError:
            return None
