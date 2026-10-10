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
	"""SAPIEN-free vector adapter for the tensor-only trajectory-shape tasks."""

	def __init__(self, cfg, num_envs: int, is_eval: bool = False):
		task_cls = TRAJECTORY_SHAPE_TASKS[cfg.env_id]
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
		task = self._task
		task.num_envs = self.num_envs
		task.device = self.device
		task._obs_mode = self.obs_mode
		task.render_mode = self.render_mode
		task._elapsed_steps = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
		task._raw_kwargs = {}
		task._traj_cfg = copy.deepcopy(task_cls.TASK_DEFAULTS)
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
		return "default"

	@property
	def unwrapped(self):
		return self

	def reset(self, seed=None, options=None):
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
		obs, reward, terminated, truncated, info = self._task.step(action)
		self._episode_returns += reward
		self._episode_lengths += 1
		timeout = self._task._elapsed_steps >= self.max_episode_steps
		terminated = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
		truncated = timeout.clone()
		info = dict(info)
		if timeout.any():
			info["final_observation"] = obs.clone() if torch.is_tensor(obs) else copy.deepcopy(obs)
			info["final_info"] = {"episode": self._episode_metrics(info)}
		return obs, reward, terminated, truncated, info

	def _episode_metrics(self, info):
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
		return self._task.render()

	def close(self):
		return None


def make_trajectory_shape_vector_env(cfg, num_envs: int, is_eval: bool = False):
	return TorchTrajectoryShapeVectorEnv(cfg, num_envs, is_eval=is_eval)
