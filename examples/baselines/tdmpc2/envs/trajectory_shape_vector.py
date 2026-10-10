import copy

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

from mani_skill.envs.tasks.trajectory_shape import TrajectoryShapePOMDPV2CoreRewardEnv

TRAJECTORY_SHAPE_TASKS = {
    "TrajectoryShapePOMDPV2CoreReward-v1": TrajectoryShapePOMDPV2CoreRewardEnv,
}

class TorchTrajectoryShapeVectorEnv:
	"""
	Tensor vector adapter for batched closed-path shaping.
	Uses the task dynamics directly and adds fixed-horizon episode statistics
	without constructing a simulator or automatically resetting completed rows.
	"""

	def __init__(self, cfg, num_envs: int, is_eval: bool = False):
		"""Create task buffers and batched Gymnasium spaces for the selected device."""
		task_cls = TRAJECTORY_SHAPE_TASKS[cfg.env_id]
		# The adapter supplies batch storage and episode bookkeeping itself.
		self._task = task_cls.__new__(task_cls)
		self.env_id = cfg.env_id
		self.num_envs = int(num_envs)
		self.device = torch.device("cuda" if cfg.env_type == "gpu" else "cpu")
		self.obs_mode = cfg.obs
		self.render_mode = cfg.get("render_mode", None)
		self.max_episode_steps = int(cfg.get("time_limit", 256))
		self._max_episode_steps = self.max_episode_steps
		self._episode_returns = torch.zeros((self.num_envs,), device=self.device)
		self._episode_lengths = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
		self._init_task(task_cls, cfg)
		self.reset()
		self.single_observation_space = gym.spaces.Box(
			low=-np.inf,
			high=np.inf,
			shape=(self._task._state_dim,),
			dtype=np.float32,
		)
		self.observation_space = batch_space(self.single_observation_space, n=self.num_envs)

	def _init_task(self, task_cls, cfg):
		"""
		Configure the task directly from defaults and traj_* overrides.
		Share the action spaces with the task while leaving episode limits here.
		"""
		task = self._task
		task.num_envs = self.num_envs
		task.device = self.device
		task._obs_mode = self.obs_mode
		task.render_mode = self.render_mode
		task._elapsed_steps = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
		task._raw_kwargs = {}
		task._traj_cfg = copy.deepcopy(task_cls.TASK_DEFAULTS)
		# Only task and scene settings are forwarded to the environment.
		for key in cfg.keys():
			if str(key).startswith("traj_") or key in {"size", "action_repeat", "time_limit", "seed"}:
				task._traj_cfg[key] = cfg[key]
		task._configure_from_dict(task._traj_cfg)
		if hasattr(task, "_set_source_seed"):
			task._set_source_seed(int(cfg.get("seed", 0)))
		self.single_action_space = task._make_single_action_space()
		self.action_space = batch_space(self.single_action_space, n=self.num_envs)
		self._orig_single_action_space = self.single_action_space
		task.single_action_space = self.single_action_space
		task.action_space = self.action_space
		task._orig_single_action_space = self._orig_single_action_space

	@property
	def control_mode(self):
		"""Return the shaping task control interface."""
		return "default"

	@property
	def unwrapped(self):
		"""Return the tensor adapter without an additional wrapper layer."""
		return self

	def reset(self, seed=None, options=None):
		"""Reset all rows and counters, optionally reseeding task random streams."""
		if seed is not None:
			if isinstance(seed, (list, tuple, np.ndarray)):
				seed = int(seed[0])
			torch.manual_seed(int(seed))
			np.random.seed(int(seed) % (2**32 - 1))
			if hasattr(self._task, "_set_source_seed"):
				self._task._set_source_seed(int(seed))
		task = self._task
		task._elapsed_steps.zero_()
		self._episode_returns.zero_()
		self._episode_lengths.zero_()
		env_idx = torch.arange(self.num_envs, device=self.device)
		task._initialize_episode(env_idx, options or {})
		obs = task.get_obs()
		task._last_obs = obs
		return obs, task.get_info()

	def step(self, action):
		"""
		Advance [B, 3] actions and return observations, rewards, and timeout flags.
		Completed observations remain available for replay; the trainer resets
		the batch after consuming final_observation and final_info.
		"""
		obs, reward, terminated, truncated, info = self._task.step(action)
		self._episode_returns += reward
		self._episode_lengths += 1
		# Use truncation for fixed-horizon episodes; the coverage task has no terminal success.
		timeout = self._task._elapsed_steps >= self.max_episode_steps
		terminated = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
		truncated = timeout.clone()
		info = dict(info)
		# Preserve the final state before the trainer starts the next episode.
		if timeout.any():
			info["final_observation"] = obs.clone() if torch.is_tensor(obs) else copy.deepcopy(obs)
			info["final_info"] = {"episode": self._episode_metrics(info)}
		return obs, reward, terminated, truncated, info

	def _episode_metrics(self, info):
		"""Collect returns, lengths, and task diagnostics for the completed batch."""
		episode = {
			"return": self._episode_returns.clone(),
			"episode_len": self._episode_lengths.float().clone(),
			"success_once": torch.zeros((self.num_envs,), device=self.device),
			"fail_once": torch.zeros((self.num_envs,), device=self.device),
		}
		for key, value in info.items():
			if torch.is_tensor(value) and value.shape[:1] == (self.num_envs,):
				episode[key] = value.detach().clone()
		return episode

	def render(self):
		"""Return the task diagnostic frames without a rendering backend."""
		return self._task.render()

	def close(self):
		"""Close the tensor adapter; no simulator resources are held."""
		return None


def make_trajectory_shape_vector_env(cfg, num_envs: int, is_eval: bool = False):
	"""Construct a tensor vector environment for training or evaluation."""
	return TorchTrajectoryShapeVectorEnv(cfg, num_envs, is_eval=is_eval)
