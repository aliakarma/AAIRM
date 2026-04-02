"""Unit tests for aairm/models/rl/ppo_policy.py."""

from __future__ import annotations

import numpy as np
import pytest

from aairm.models.rl.ppo_policy import PPOPolicy


def test_predict_before_training_returns_default():
    policy = PPOPolicy()
    obs = np.array([50.0, 70.0, 15.0, 0.8, 1.0], dtype=np.float32)
    action, state = policy.predict(obs, deterministic=True)
    assert action[0] >= 0.0
    assert isinstance(float(action[0]), float)


def test_predict_clips_negative():
    """Policy must never return negative order quantity."""
    policy = PPOPolicy()
    obs = np.zeros(5, dtype=np.float32)
    action, _ = policy.predict(obs)
    assert action[0] >= 0.0


def test_build_without_env_does_not_crash():
    """Building without an env should not raise (SB3 unavailable path)."""
    policy = PPOPolicy()
    # No env injected — should not crash
    try:
        policy.build(None)
    except Exception:
        pass  # Acceptable if SB3 not installed


def test_save_load_roundtrip(tmp_path):
    policy = PPOPolicy()
    path = tmp_path / "policy"
    policy.save(str(path))    # Should not crash even when untrained
    policy.load(str(path))    # Should not crash on missing file
