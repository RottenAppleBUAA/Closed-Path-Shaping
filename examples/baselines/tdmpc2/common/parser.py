import os
import re
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from common import MODEL_SIZE


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
	"""
	Parses a Hydra config. Mostly for convenience.
	"""

	# Logic
	for k in cfg.keys():
		try:
			v = cfg[k]
			if v == None:
				v = True
		except:
			pass

	# Algebraic expressions
	for k in cfg.keys():
		try:
			v = cfg[k]
			if isinstance(v, str):
				match = re.match(r"(\d+)([+\-*/])(\d+)", v)
				if match:
					cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
					if isinstance(cfg[k], float) and cfg[k].is_integer():
						cfg[k] = int(cfg[k])
		except:
			pass

	# Convenience
	try:
		original_cwd = hydra.utils.get_original_cwd()
	except ValueError:
		original_cwd = os.getcwd()
	cfg.work_dir = Path(original_cwd) / 'logs' / cfg.env_id / str(cfg.seed) / cfg.exp_name
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins-1) # Bin size for discrete regression

	# Model size
	if cfg.get('model_size', None) is not None:
		assert cfg.model_size in MODEL_SIZE.keys(), \
			f'Invalid model size {cfg.model_size}. Must be one of {list(MODEL_SIZE.keys())}'
		for k, v in MODEL_SIZE[cfg.model_size].items():
			cfg[k] = v

	cfg.multitask = False
	cfg.task_dim = 0
	cfg.tasks = [cfg.env_id]

	# Tensor environment metadata
	cfg.env_cfg.env_id = cfg.eval_env_cfg.env_id = cfg.env_id
	cfg.env_cfg.obs_mode = cfg.eval_env_cfg.obs_mode = cfg.obs # compact state
	cfg.env_cfg.reward_mode = cfg.eval_env_cfg.reward_mode = 'normalized_dense'
	cfg.env_cfg.num_envs = cfg.num_envs
	cfg.eval_env_cfg.num_envs = cfg.num_eval_envs
	cfg.env_cfg.sim_backend = cfg.eval_env_cfg.sim_backend = cfg.env_type

	cfg.eval_env_cfg.num_eval_episodes = cfg.eval_episodes_per_env * cfg.num_eval_envs

	# cfg.(eval_)env_cfg.control_mode is defined in envs/__init__.py
	# cfg.(eval_)env_cfg.env_horizon is defined in envs/__init__.py
	# cfg.discount is defined in tdmpc2.py

	return cfg
