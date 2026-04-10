"""PPO Reorder Ordering Policy for the Reorder Optimisation Agent (C2).

Wraps Stable-Baselines3's PPO implementation with the observation and
action spaces defined by the AAIRM simulation environment.

Observation space (per SKU, 5-dimensional):
    [effective_available, forecast_mean, forecast_std,
     days_to_expiry_normalised, budget_remaining_fraction]

Action space:
    Continuous order quantity Q ∈ [0, Q_max] (Box, 1-dimensional).

Training follows Eq. 5 of the paper:
    max_φ E[Σ γ^t r(s_t, a_t)]
    where r = -C_i(Q_i) (negative expected cost from Eq. 3).

References
----------
Paper Section 4.2.2; Eq. 5; Repo Guide Section 5.4.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from aairm.utils.logging import get_logger

logger = get_logger(__name__)

_OBS_DIM = 5     # observation space dimension
_ACT_HIGH = 1e4  # maximum order quantity in action space


class PPOPolicy:
    """PPO-based reorder ordering policy.

    Wraps ``stable_baselines3.PPO`` so that C2 can call
    ``policy.predict(obs)`` using the standard SB3 interface.

    Args:
        learning_rate: PPO optimiser learning rate (default 3e-4).
        n_steps: Steps per PPO update batch (default 2048).
        batch_size: Mini-batch size (default 64).
        n_epochs: PPO epochs per update (default 10).
        gamma: Discount factor (default 0.99, paper value).
        verbose: SB3 verbosity level.
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        verbose: int = 0,
    ) -> None:
        self._lr = learning_rate
        self._n_steps = n_steps
        self._batch = batch_size
        self._n_epochs = n_epochs
        self._gamma = gamma
        self._verbose = verbose
        self._model: Any = None
        self._trained = False

    def build(self, env: Any) -> "PPOPolicy":
        """Construct the PPO model for a given environment.

        Args:
            env: A gymnasium-compatible environment (e.g. RetailEnv).

        Returns:
            Self.
        """
        try:
            from stable_baselines3 import PPO  # type: ignore

            self._model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=self._lr,
                n_steps=self._n_steps,
                batch_size=self._batch,
                n_epochs=self._n_epochs,
                gamma=self._gamma,
                verbose=self._verbose,
                tensorboard_log=None,
            )
            logger.info("ppo.model_built", obs_dim=_OBS_DIM)
        except ImportError:
            logger.warning("ppo.stable_baselines3_not_available")
        return self

    def train(self, total_timesteps: int) -> "PPOPolicy":
        """Train the PPO policy.

        Args:
            total_timesteps: Total environment steps to train for.
                Paper: 400 episodes × ~365 steps/episode ≈ 146,000 steps.

        Returns:
            Self.
        """
        if self._model is None:
            logger.warning("ppo.train.model_not_built")
            return self
        try:
            logger.info("ppo.training_started", total_timesteps=total_timesteps)
            
            # Create callback for logging
            from stable_baselines3.common.callbacks import BaseCallback
            
            class TrainingCallback(BaseCallback):
                def __init__(self, verbose=0):
                    super().__init__(verbose)
                    self.episode_rewards = []
                    self.episode_lengths = []
                    self.current_episode_reward = 0
                    self.current_episode_length = 0
                
                def _on_step(self) -> bool:
                    # Log step info
                    reward = self.locals.get('rewards', [0])[0] if 'rewards' in self.locals else 0
                    done = self.locals.get('dones', [False])[0] if 'dones' in self.locals else False
                    
                    self.current_episode_reward += reward
                    self.current_episode_length += 1
                    
                    if done:
                        self.episode_rewards.append(self.current_episode_reward)
                        self.episode_lengths.append(self.current_episode_length)
                        
                        # Log episode reward
                        logger.info("ppo.episode_completed", 
                                  episode=len(self.episode_rewards),
                                  reward=round(self.current_episode_reward, 2),
                                  length=self.current_episode_length)
                        
                        # Check if rewards are constant (indicates no learning)
                        if len(self.episode_rewards) >= 5:
                            recent_rewards = self.episode_rewards[-5:]
                            if len(set(recent_rewards)) == 1:
                                logger.warning("ppo.constant_rewards_detected", 
                                             recent_rewards=recent_rewards,
                                             message="PPO may not be learning - rewards are constant")
                        
                        self.current_episode_reward = 0
                        self.current_episode_length = 0
                    
                    # Log loss if available
                    if 'loss' in self.locals:
                        loss = self.locals['loss']
                        if hasattr(loss, 'item'):
                            loss_val = loss.item()
                        else:
                            loss_val = float(loss)
                        logger.debug("ppo.training_loss", loss=round(loss_val, 6))
                    
                    return True
                
                def _on_training_end(self):
                    # Log final statistics
                    if self.episode_rewards:
                        avg_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                        logger.info("ppo.training_summary", 
                                  total_episodes=len(self.episode_rewards),
                                  avg_episode_reward=round(avg_reward, 2),
                                  min_reward=min(self.episode_rewards),
                                  max_reward=max(self.episode_rewards))
                    
                    # Log action distribution (sample some predictions)
                    try:
                        sample_obs = np.random.randn(10, _OBS_DIM).astype(np.float32)
                        actions, _ = self.model.predict(sample_obs, deterministic=True)
                        action_stats = {
                            'mean': round(float(np.mean(actions)), 2),
                            'std': round(float(np.std(actions)), 2),
                            'min': round(float(np.min(actions)), 2),
                            'max': round(float(np.max(actions)), 2)
                        }
                        logger.info("ppo.action_distribution", **action_stats)
                    except Exception as e:
                        logger.debug("ppo.action_distribution_failed", error=str(e))
            
            callback = TrainingCallback()
            self._model.learn(total_timesteps=total_timesteps, callback=callback)
            self._trained = True
            logger.info("ppo.training_completed", total_timesteps=total_timesteps)
        except Exception as exc:  # noqa: BLE001
            logger.error("ppo.train.failed", error=str(exc))
        return self

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Any]:
        """Predict the order quantity for a given observation.

        Args:
            obs: Observation vector of shape ``(5,)`` or ``(1, 5)``.
            deterministic: If ``True``, use the mean policy action
                (no exploration noise).  Always ``True`` at evaluation.

        Returns:
            Tuple of ``(action, state)`` matching the SB3 interface.
            ``action[0]`` is the predicted order quantity Q*.
        """
        if self._model is None:
            raise RuntimeError("PPO model not built. Call build() first.")
        if not self._trained:
            raise RuntimeError("PPO policy not trained. Call train() first.")

        try:
            action, state = self._model.predict(obs, deterministic=deterministic)
            action = np.clip(action, 0.0, _ACT_HIGH).astype(np.float32)
            return action, state
        except Exception as exc:  # noqa: BLE001
            logger.error("ppo.predict.failed", error=str(exc))
            return np.array([50.0], dtype=np.float32), None

    def save(self, path: str | Path) -> None:
        """Persist policy weights.

        Args:
            path: File path (SB3 adds ``.zip`` extension automatically).
        """
        if self._model is not None:
            self._model.save(str(path))
            logger.info("ppo.saved", path=str(path))

    def load(self, path: str | Path) -> "PPOPolicy":
        """Load policy weights from a SB3 checkpoint.

        Args:
            path: File path written by :meth:`save`.

        Returns:
            Self with loaded weights.
        """
        p = Path(path)
        zip_path = p if p.suffix == ".zip" else Path(str(p) + ".zip")
        if zip_path.exists():
            try:
                from stable_baselines3 import PPO  # type: ignore
                self._model = PPO.load(str(path))
                self._trained = True
                logger.info("ppo.loaded", path=str(path))
            except Exception as exc:  # noqa: BLE001
                logger.error("ppo.load.failed", error=str(exc))
        else:
            logger.warning("ppo.load.file_not_found", path=str(path))
        return self
