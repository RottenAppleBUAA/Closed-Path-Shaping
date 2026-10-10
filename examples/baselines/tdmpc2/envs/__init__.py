"""Tensor environment factory for the closed-path shaping task."""
from envs.trajectory_shape_vector import make_trajectory_shape_vector_env


def make_envs(cfg, num_envs, is_eval=False):
    """
    Build the task and populate dimensions required by the TD-MPC2 model.
    The seed budget collects at least one complete vector episode before
    replay-backed learning begins.
    """
    if cfg.env_id != "TrajectoryShapePOMDPV2CoreReward-v1":
        raise ValueError(f"Unsupported environment: {cfg.env_id}")
    if cfg.obs != "state":
        raise ValueError("Closed-path shaping uses compact state observations.")
    if cfg.env_type not in {"cpu", "gpu"}:
        raise ValueError("env_type must be cpu or gpu")
    env = make_trajectory_shape_vector_env(cfg, num_envs, is_eval=is_eval)
    # Derive model dimensions from the actual environment spaces.
    cfg.obs_shape = {"state": env.single_observation_space.shape}
    cfg.action_dim = env.single_action_space.shape[0]
    cfg.episode_length = env.max_episode_steps
    # Replay is populated at episode boundaries rather than individual vector steps.
    cfg.seed_steps = max(1000, cfg.num_envs * cfg.episode_length)
    cfg.env_cfg.control_mode = cfg.eval_env_cfg.control_mode = env.control_mode
    cfg.env_cfg.env_horizon = cfg.eval_env_cfg.env_horizon = env.max_episode_steps
    return env
