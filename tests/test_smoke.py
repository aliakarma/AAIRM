"""Lightweight smoke test for the AAIRM pipeline.

Ensures key components can be imported and instantiated without crashing.
"""

from pathlib import Path
import sys


def test_imports():
    """Test that all key modules can be imported."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Core imports
    from aairm.simulation.environment import RetailEnv
    from aairm.baselines.rop_eoq import ROPEOQPolicy
    from aairm.baselines.ml_static import MLStaticPolicy
    from aairm.agents.meta_orchestrator import MetaOrchestrator
    from aairm.evaluation.benchmarker import Benchmarker
    from aairm.utils.config import AAIRMConfig
    from aairm.utils.seed import set_global_seed
    from experiments.run_paper_experiment import (
        load_config,
        build_baselines,
    )
    
    assert all([
        RetailEnv,
        ROPEOQPolicy,
        MLStaticPolicy,
        MetaOrchestrator,
        Benchmarker,
        AAIRMConfig,
        set_global_seed,
        load_config,
        build_baselines,
    ]), "All imports failed"


def test_config_loading():
    """Test that configuration can be loaded."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from experiments.run_paper_experiment import load_config
    
    # Load config in fast mode
    config, reward_tuning, raw_payload = load_config(
        "configs/simulation_1200sku.yaml",
        fast=True,
        seed_override=42
    )
    
    assert config is not None, "Config loading failed"
    assert config.simulation.n_skus == 10, "Fast mode not applied"
    assert config.simulation.test_horizon_days == 30, "Fast mode test horizon not set"


def test_seed_setting():
    """Test that global seed setting works."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from aairm.utils.seed import set_global_seed
    
    # Should not raise
    set_global_seed(42)
    set_global_seed(123)
    assert True, "Seed setting succeeded"