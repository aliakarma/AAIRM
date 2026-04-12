"""Centralised configuration management via Pydantic v2 Settings.

All parameters are documented and default to values that reproduce the
paper's experimental conditions exactly.  Override via:

    1. YAML file  (recommended) — load with OmegaConf and pass as a dict.
    2. Environment variables   — prefix ``AAIRM_`` (e.g. ``AAIRM_LOG_LEVEL``).
    3. ``.env`` file           — loaded automatically from the working directory.

Usage
-----
    from aairm.utils.config import AAIRMConfig

    # Defaults only (reproduces paper)
    config = AAIRMConfig()

    # From YAML
    from omegaconf import OmegaConf
    raw = OmegaConf.to_container(OmegaConf.load("configs/default.yaml"))
    config = AAIRMConfig.model_validate(raw)

References
----------
AAIRM Repo Guide, Section 4.3; Paper Section 5.1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Sub-configurations
# ---------------------------------------------------------------------------

class SimulationConfig(BaseSettings):
    """Synthetic retail simulation parameters (paper Section 5.1)."""

    model_config = SettingsConfigDict(env_prefix="AAIRM_SIM_")

    n_skus: int = Field(1200, gt=0,
        description="Total SKU count across all categories.")
    n_categories: int = Field(5, gt=0,
        description="Number of product categories.")
    category_names: list[str] = Field(
        default=["grocery", "frozen_food", "apparel", "cosmetics", "dry_fruits"],
        description="Category identifiers.  Length must equal n_categories.",
    )
    skus_per_category: int = Field(240, gt=0,
        description="SKUs per category.  Computed as n_skus // n_categories.")
    simulation_horizon_days: int = Field(730, gt=0,
        description="Total simulation horizon in days (train + test).")
    test_horizon_days: int = Field(365, gt=0,
        description="Test horizon in days.  Must be < simulation_horizon_days.")
    n_suppliers_min: int = Field(3, ge=1,
        description="Minimum number of suppliers generated per SKU.")
    n_suppliers_max: int = Field(5, ge=1,
        description="Maximum number of suppliers generated per SKU.")
    seed: int = Field(42, ge=0,
        description="Global random seed.  Paper uses 42.")
    full_coverage: bool = Field(False,
        description="If true, InventoryMonitorAgent selects all SKUs every cycle for aggressive coverage.")
    stockout_penalty_weight: float = Field(5.0, gt=0.0,
        description="Alpha multiplier applied to stockout units when computing evaluation penalty cost.")

    @field_validator("category_names")
    @classmethod
    def _validate_category_count(cls, v: list[str], info) -> list[str]:  # noqa: ANN001
        n = info.data.get("n_categories", 5)
        if len(v) != n:
            raise ValueError(
                f"category_names length ({len(v)}) must equal "
                f"n_categories ({n})."
            )
        return v

    @model_validator(mode="after")
    def _validate_sku_divisibility(self) -> "SimulationConfig":
        if self.n_skus % self.n_categories != 0:
            raise ValueError(
                f"n_skus ({self.n_skus}) must be divisible by "
                f"n_categories ({self.n_categories})."
            )
        self.skus_per_category = self.n_skus // self.n_categories
        return self

    @model_validator(mode="after")
    def _validate_horizon(self) -> "SimulationConfig":
        if self.test_horizon_days >= self.simulation_horizon_days:
            raise ValueError(
                "test_horizon_days must be less than simulation_horizon_days."
            )
        return self


class ForecastingConfig(BaseSettings):
    """Demand Forecasting Agent (C1) parameters (paper Section 4.2.1)."""

    model_config = SettingsConfigDict(env_prefix="AAIRM_FORECAST_")

    architecture: Literal["tft", "lstm", "naive"] = Field(
        "tft",
        description=(
            "Forecasting model architecture.  "
            "'tft' = Temporal Fusion Transformer (paper default); "
            "'lstm' = LSTM encoder-decoder; "
            "'naive' = seasonal naive (fast baseline)."
        ),
    )
    forecast_horizon: int = Field(7, gt=0,
        description="Forecast horizon H in days (paper: 7 days).")
    context_length: int = Field(60, gt=0,
        description="Input history window in days fed to the model.")
    hidden_size: int = Field(128, gt=0,
        description="Hidden layer size for TFT / LSTM architectures.")
    attention_head_size: int = Field(4, gt=0,
        description="Number of attention heads (TFT only).")
    dropout: float = Field(0.1, ge=0.0, le=0.5)
    batch_size: int = Field(64, gt=0)
    max_epochs: int = Field(50, gt=0)
    learning_rate: float = Field(1e-3, gt=0.0)
    loss: Literal["mse", "pinball"] = Field(
        "mse",
        description=(
            "Training loss function.  "
            "'mse' for point forecasts; 'pinball' for quantile forecasts."
        ),
    )
    quantiles: list[float] = Field(
        default=[0.1, 0.5, 0.9],
        description="Quantile levels output by the model (used when loss='pinball').",
    )


class OptimisationConfig(BaseSettings):
    """Reorder Optimisation Agent (C2) parameters (paper Section 4.2.2)."""

    model_config = SettingsConfigDict(env_prefix="AAIRM_OPT_")

    mode: Literal["rl", "analytical"] = Field(
        "rl",
        description=(
            "Optimisation mode.  "
            "'rl' = PPO-based learned policy (paper default); "
            "'analytical' = direct cost minimisation of Eq. 3."
        ),
    )
    budget: float = Field(1_000_000.0, gt=0.0,
        description="Purchasing budget B in Eq. 4 of the paper (currency units).")
    warehouse_capacity: float = Field(100_000.0, gt=0.0,
        description="Warehouse volume capacity V in Eq. 4 (cubic units).")
    service_level: float = Field(0.95, gt=0.0, lt=1.0,
        description="Target cycle service level used in ROP computation.")
    holding_cost_rate: float = Field(0.25, gt=0.0,
        description="Annual holding cost as a fraction of unit cost h_i.")
    penalty_cost_multiplier: float = Field(3.0, gt=0.0,
        description="Stockout penalty as a multiple of unit cost p_i = mult * c_i.")
    min_order_quantity: float = Field(50.0, ge=0.0,
        description="Minimum order quantity safeguard to prevent zero-order behavior.")
    discount_factor: float = Field(0.99, gt=0.0, lt=1.0,
        description="RL discount factor γ in Eq. 5 of the paper.")
    rl_training_episodes: int = Field(400, gt=0,
        description="Number of PPO training episodes (paper: 400, convergence at ~250).")
    rl_learning_rate: float = Field(3e-4, gt=0.0,
        description="PPO optimiser learning rate.")
    rl_n_steps: int = Field(2048, gt=0,
        description="Number of environment steps per PPO update.")
    rl_batch_size: int = Field(64, gt=0)
    rl_n_epochs: int = Field(10, gt=0,
        description="Number of PPO epochs per update.")


class SupplierRankingConfig(BaseSettings):
    """Supplier Ranking Agent (C3) weight parameters (paper Eq. 6)."""

    model_config = SettingsConfigDict(env_prefix="AAIRM_SUPP_")

    alpha_1: float = Field(0.35, ge=0.0, le=1.0,
        description="Weight on normalised unit cost c_ij.")
    alpha_2: float = Field(0.30, ge=0.0, le=1.0,
        description="Weight on normalised adjusted lead time L_hat_ij.")
    alpha_3: float = Field(0.25, ge=0.0, le=1.0,
        description="Weight on reliability r_ij (subtracted — higher is better).")
    alpha_4: float = Field(0.10, ge=0.0, le=1.0,
        description="Penalty weight for MOQ violation indicator.")

    @model_validator(mode="after")
    def _weights_sum_check(self) -> "SupplierRankingConfig":
        total = self.alpha_1 + self.alpha_2 + self.alpha_3 + self.alpha_4
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Supplier ranking weights must sum to 1.0; got {total:.6f}."
            )
        return self


class GovernanceConfig(BaseSettings):
    """Governance and Policy Agent (C5) constraint parameters."""

    model_config = SettingsConfigDict(env_prefix="AAIRM_GOV_")

    max_single_supplier_fraction: float = Field(0.60, gt=0.0, le=1.0,
        description=(
            "Maximum fraction of a category's procurement volume that may "
            "go to a single supplier before a diversification flag is raised."
        ),
    )
    shelf_life_safety_margin: float = Field(0.80, gt=0.0, le=1.0,
        description=(
            "For perishables, the order quantity Q* must not exceed "
            "shelf_life_demand * safety_margin.  Default 0.80 (80\\%)."
        ),
    )
    human_approval_threshold: float = Field(50_000.0, gt=0.0,
        description=(
            "Orders with total value (Q* × unit_cost) above this threshold "
            "require human approval before execution."
        ),
    )
    frozen_zone_capacity: float = Field(20_000.0, gt=0.0,
        description="Cold-storage capacity in volumetric units.")
    ambient_zone_capacity: float = Field(80_000.0, gt=0.0,
        description="Ambient-storage capacity in volumetric units.")


class LLMConfig(BaseSettings):
    """LLM backbone configuration for all agent reasoning."""

    model_config = SettingsConfigDict(env_prefix="AAIRM_LLM_")

    model: str = Field("gpt-4o",
        description="LLM model identifier.  Must be an OpenAI-compatible model name.")
    temperature: float = Field(0.0, ge=0.0, le=2.0,
        description=(
            "Sampling temperature.  0.0 = fully deterministic reasoning "
            "(paper default); increase for more varied negotiation responses."
        ),
    )
    max_tokens: int = Field(1024, gt=0,
        description="Maximum tokens per LLM response.")
    timeout: float = Field(30.0, gt=0.0,
        description="Per-request LLM API timeout in seconds.")


# ---------------------------------------------------------------------------
# Top-level aggregated configuration
# ---------------------------------------------------------------------------

class AAIRMConfig(BaseSettings):
    """Top-level AAIRM configuration.

    Aggregates all sub-configurations and exposes global settings.
    Loads values from (in priority order):

        1. Explicit kwargs at construction time.
        2. Environment variables prefixed with ``AAIRM_``.
        3. ``.env`` file in the current working directory.
        4. Hard-coded defaults (reproduce paper results exactly).

    Examples:
        >>> config = AAIRMConfig()
        >>> config.simulation.n_skus
        1200
        >>> config.supplier_ranking.alpha_1
        0.35
    """

    model_config = SettingsConfigDict(
        env_prefix="AAIRM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    forecasting: ForecastingConfig = Field(default_factory=ForecastingConfig)
    optimisation: OptimisationConfig = Field(default_factory=OptimisationConfig)
    supplier_ranking: SupplierRankingConfig = Field(default_factory=SupplierRankingConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    results_dir: Path = Field(Path("experiments/results"),
        description="Directory for writing experiment outputs.")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO")
    log_format: Literal["console", "json"] = Field("console")
    fast_dev_run: bool = Field(False,
        description="Enable fast development mode with reduced training and logging.")
