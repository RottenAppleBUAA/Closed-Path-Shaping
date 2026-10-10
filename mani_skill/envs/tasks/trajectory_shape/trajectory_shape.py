import copy
import math
from collections import OrderedDict
from typing import Any
import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

def _parse_xy_points(value: Any, name: str) -> np.ndarray:
    if value is None or (isinstance(value, str) and not value.strip()):
        return np.zeros((0, 2), dtype=np.float32)
    if isinstance(value, str):
        rows = [[float(item.strip()) for item in row.split(",")] for row in value.split(";") if row.strip()]
        points = np.asarray(rows, dtype=np.float32)
    else:
        points = np.asarray(value, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"{name} must contain x,y pairs; got shape={points.shape}")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return points

def _as_pair(value):
    if isinstance(value, str):
        parts = [int(x.strip()) for x in value.strip("()[]").split(",") if x.strip()]
        if len(parts) == 2:
            return tuple(parts)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    return int(value), int(value)

TRAJECTORY_SHAPE_POMDP_DEFAULTS = {'size': (64, 64),
 'action_repeat': 1,
 'time_limit': 256,
 'traj_num_angles': 128,
 'traj_radius_min': 1.0,
 'traj_radius_max': 7.5,
 'traj_domain_radius': 10.0,
 'traj_kernel_sigma': 1.5,
 'traj_kernel_cutoff': 2.0,
 'traj_subsamples': 2,
 'traj_num_agents': 3,
 'traj_heat_lambda': 0.25,
 'traj_resource_scale': 1.0,
 'traj_phi_sat': 1.0,
 'traj_potential_eps': 1e-06,
 'traj_reward_abs_weight': 0.45,
 'traj_min_poi_coverage_radius': 1.8,
 'traj_directional_target_sigma': 0.35,
 'traj_alpha_max': 0.3,
 'traj_lambda_min': 0.2,
 'traj_lambda_max': 1.0,
 'traj_obs_mode': 'partial',
 'traj_poi_sample_interval': 3,
 'traj_poi_max_obs': 8,
 'traj_r_soft': 1.5,
 'traj_interest_sigma': 1.5,
 'traj_center_mode': 'fixed',
 'traj_center_history': 20,
 'traj_center_weight_alpha': 0.2,
 'traj_poi_num_min': 4,
 'traj_poi_num_max': 8,
 'traj_poi_dt': 0.05,
 'traj_poi_dynamic_enabled': True,
 'traj_poi_motion_mode': 'hotspot',
 'traj_poi_script_start_xy': '',
 'traj_poi_script_end_xy': '',
 'traj_poi_script_duration_steps': 256,
 'traj_poi_script_easing': 'smoothstep',
 'traj_poi_random_strength': 0.2,
 'traj_poi_inertia': 0.8,
 'traj_poi_hotspot_num': 4,
 'traj_poi_hotspot_speed': 1.2,
 'traj_poi_hotspot_arrival_tol': 0.2,
 'traj_poi_hotspot_radius_min_ratio': 0.25,
 'traj_poi_hotspot_radius_max_ratio': 0.9,
 'traj_poi_hotspot_min_dist': 5.0,
 'traj_poi_retarget_interval': 50,
 'traj_poi_target_min_unique': 2,
 'traj_poi_init_min_dist': 1.0,
 'traj_nofly_num': 0,
 'traj_nofly_radius_min_ratio': 0.5,
 'traj_nofly_radius_max_ratio': 1.0,
 'traj_nofly_penalty': 3.0,
 'traj_source_compatible_rng': True,
 'traj_obs_vector_mode': 'compact',
 'traj_curve_repr': 'fft',
 'traj_fft_k': 12,
 'traj_stack_len': 4,
 'traj_obs_noise_std': 0.05,
 'traj_obs_dropout_prob': 0.05,
 'traj_obs_source': 'trajectory',
 'traj_uav_num': 4,
 'traj_uav_dt': 0.1,
 'traj_uav_speed': 1.8,
 'traj_uav_omega_max': 2.4,
 'traj_uav_phase_speed': 0.1,
 'traj_uav_init_radius_ratio': 0.2,
 'traj_uav_init_noise_ratio': 0.03,
 'traj_hide_hotspots': True,
 'traj_privileged_stats_in_obs': False,
 'traj_reward_t_theta_weight': 0.3}

class TrajectoryShapePOMDPV2CoreRewardEnv(gym.Env):
    """Final partial-observation orbit task with visible-only state statistics."""
    SUPPORTED_OBS_MODES = ("state", "state_dict", "none")
    SUPPORTED_REWARD_MODES = ("normalized_dense", "dense", "none")
    SUPPORTED_ROBOTS = ["none"]
    TASK_DEFAULTS = TRAJECTORY_SHAPE_POMDP_DEFAULTS

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, num_envs=1, device="cpu", obs_mode="state", render_mode=None, **kwargs):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self._obs_mode = obs_mode
        self.render_mode = render_mode
        self._raw_kwargs = dict(kwargs)
        self._traj_cfg = copy.deepcopy(self.TASK_DEFAULTS)
        self._traj_cfg.update(kwargs)
        self._configure_from_dict(self._traj_cfg)
        self._elapsed_steps = torch.zeros((self.num_envs,), dtype=torch.int32, device=self.device)
        self._set_source_seed(int(kwargs.get("seed", 0)))
        self.single_action_space = self._make_single_action_space()
        self.action_space = batch_space(self.single_action_space, n=self.num_envs)
        self._orig_single_action_space = self.single_action_space
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._state_dim,), dtype=np.float32,
        )
        self.observation_space = batch_space(self.single_observation_space, n=self.num_envs)
        self.reset()

    @property
    def obs_mode(self):
        return self._obs_mode

    @property
    def control_mode(self):
        return "default"

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed) % (2**32 - 1))
            self._set_source_seed(int(seed))
        self._elapsed_steps.zero_()
        self._initialize_episode(torch.arange(self.num_envs, device=self.device), options or {})
        self._last_obs = self.get_obs()
        return self._last_obs, self.get_info()

    def close(self):
        return None

    def _configure_from_dict(self, cfg):
        self._height, self._width = _as_pair(cfg.get("size", (64, 64)))
        self._action_repeat = max(1, int(cfg.get("action_repeat", 1)))
        self._num_angles = int(cfg.get("traj_num_angles", 128))
        self._radius_min = float(cfg.get("traj_radius_min", 1.0))
        self._radius_max = float(cfg.get("traj_radius_max", 7.5))
        self._domain_radius = float(cfg.get("traj_domain_radius", 10.0))
        self._kernel_sigma_phys = float(cfg.get("traj_kernel_sigma", 1.5))
        self._traj_subsamples = max(1, int(cfg.get("traj_subsamples", 2)))
        self._num_agents = float(cfg.get("traj_num_agents", 3.0))
        self._heat_lambda = float(cfg.get("traj_heat_lambda", 0.25))
        self._resource_eta = float(cfg.get("traj_resource_scale", 1.0))
        self._phi_sat = float(cfg.get("traj_phi_sat", 1.0))
        self._psi_eps = float(cfg.get("traj_potential_eps", 1e-6))
        self._reward_abs_weight = float(cfg.get("traj_reward_abs_weight", 0.45))
        self._min_poi_coverage_radius = float(cfg.get("traj_min_poi_coverage_radius", 1.8))
        self._directional_target_sigma = float(cfg.get("traj_directional_target_sigma", 0.35))
        self._phi_max = math.pi
        self._alpha_max = float(cfg.get("traj_alpha_max", 0.3))
        self._lambda_min = float(cfg.get("traj_lambda_min", 0.2))
        self._lambda_max = float(cfg.get("traj_lambda_max", 1.0))
        self._obs_mode_task = str(cfg.get("traj_obs_mode", "full")).lower()
        self._poi_sample_interval = max(1, int(cfg.get("traj_poi_sample_interval", 1)))
        self._poi_num_min = max(0, int(cfg.get("traj_poi_num_min", 4)))
        self._poi_num_max = max(self._poi_num_min, int(cfg.get("traj_poi_num_max", 8)))
        self._poi_max_obs = max(int(cfg.get("traj_poi_max_obs", self._poi_num_max)), self._poi_num_max, 1)
        self._r_soft = float(cfg.get("traj_r_soft", self._kernel_sigma_phys))
        self._poi_dt = float(cfg.get("traj_poi_dt", 0.05))
        self._poi_dynamic_enabled = bool(cfg.get("traj_poi_dynamic_enabled", True))
        self._poi_motion_mode = str(cfg.get("traj_poi_motion_mode", "hotspot")).strip().lower()
        if self._poi_motion_mode not in {"hotspot", "scripted_endpoints"}:
            raise ValueError(f"unknown traj_poi_motion_mode={self._poi_motion_mode!r}")
        self._poi_script_start_xy = _parse_xy_points(cfg.get("traj_poi_script_start_xy", ""), "traj_poi_script_start_xy")
        self._poi_script_end_xy = _parse_xy_points(cfg.get("traj_poi_script_end_xy", ""), "traj_poi_script_end_xy")
        self._poi_script_duration_steps = max(1, int(cfg.get("traj_poi_script_duration_steps", 256)))
        self._poi_script_easing = str(cfg.get("traj_poi_script_easing", "smoothstep")).strip().lower()
        if self._poi_script_easing not in {"linear", "smoothstep"}:
            raise ValueError(f"unknown traj_poi_script_easing={self._poi_script_easing!r}")
        if self._poi_motion_mode == "scripted_endpoints":
            if self._poi_script_start_xy.shape != self._poi_script_end_xy.shape or self._poi_script_start_xy.shape[0] == 0:
                raise ValueError(
                    "scripted_endpoints requires non-empty traj_poi_script_start_xy and traj_poi_script_end_xy with matching shapes"
                )
            scripted_count = int(self._poi_script_start_xy.shape[0])
            if scripted_count > self._poi_max_obs:
                raise ValueError(f"scripted PoI count {scripted_count} exceeds traj_poi_max_obs={self._poi_max_obs}")
            if np.abs(self._poi_script_start_xy).max() > self._domain_radius or np.abs(self._poi_script_end_xy).max() > self._domain_radius:
                raise ValueError("scripted PoI coordinates must remain inside traj_domain_radius")
            self._poi_num_min = scripted_count
            self._poi_num_max = scripted_count
        self._poi_random_strength = float(cfg.get("traj_poi_random_strength", 0.2))
        self._poi_inertia = float(cfg.get("traj_poi_inertia", 0.8))
        self._poi_hotspot_num = max(1, int(cfg.get("traj_poi_hotspot_num", 4)))
        self._poi_hotspot_speed = float(cfg.get("traj_poi_hotspot_speed", 1.2))
        self._poi_hotspot_arrival_tol = float(cfg.get("traj_poi_hotspot_arrival_tol", 0.2))
        self._poi_hotspot_radius_min_ratio = float(cfg.get("traj_poi_hotspot_radius_min_ratio", 0.25))
        self._poi_hotspot_radius_max_ratio = float(cfg.get("traj_poi_hotspot_radius_max_ratio", 0.90))
        self._poi_hotspot_min_dist = float(max(1e-6, cfg.get("traj_poi_hotspot_min_dist", 5.0)))
        self._poi_retarget_interval = max(0, int(cfg.get("traj_poi_retarget_interval", 50)))
        self._poi_target_min_unique = max(1, int(cfg.get("traj_poi_target_min_unique", 2)))
        self._poi_init_min_dist = float(cfg.get("traj_poi_init_min_dist", 1.0))
        self._nofly_num = max(0, int(cfg.get("traj_nofly_num", 0)))
        self._nofly_radius_min_ratio = float(max(0.0, cfg.get("traj_nofly_radius_min_ratio", 0.5)))
        self._nofly_radius_max_ratio = float(max(self._nofly_radius_min_ratio, cfg.get("traj_nofly_radius_max_ratio", 1.0)))
        self._nofly_penalty = float(max(0.0, cfg.get("traj_nofly_penalty", 3.0)))
        self._source_compatible_rng = bool(cfg.get("traj_source_compatible_rng", True))
        self._obs_vector_mode = str(cfg.get("traj_obs_vector_mode", "compact")).lower()
        self._privileged_stats_in_obs = bool(cfg.get("traj_privileged_stats_in_obs", False))
        self._curve_repr = str(cfg.get("traj_curve_repr", "fft")).lower()
        self._fft_k = max(1, min(int(cfg.get("traj_fft_k", 12)), self._num_angles // 2))
        self._stack_len = max(1, int(cfg.get("traj_stack_len", 1)))
        self._obs_noise_std = max(0.0, float(cfg.get("traj_obs_noise_std", 0.0)))
        self._obs_dropout_prob = min(max(float(cfg.get("traj_obs_dropout_prob", 0.0)), 0.0), 1.0)
        self._obs_source = str(cfg.get("traj_obs_source", "trajectory")).lower()
        if self._obs_source not in {"trajectory", "uav"}:
            raise ValueError(f"traj_obs_source must be 'trajectory' or 'uav', got {self._obs_source!r}")
        self._uav_num = max(1, int(cfg.get("traj_uav_num", 4)))
        self._uav_dt = max(0.0, float(cfg.get("traj_uav_dt", 0.1)))
        self._uav_speed = max(0.0, float(cfg.get("traj_uav_speed", 1.8)))
        self._uav_omega_max = max(0.0, float(cfg.get("traj_uav_omega_max", 2.4)))
        self._uav_phase_speed = float(cfg.get("traj_uav_phase_speed", 0.10))
        self._uav_init_radius_ratio = max(0.0, float(cfg.get("traj_uav_init_radius_ratio", 0.2)))
        self._uav_init_noise_ratio = max(0.0, float(cfg.get("traj_uav_init_noise_ratio", 0.03)))
        self._hide_hotspots = bool(cfg.get("traj_hide_hotspots", False))
        self._curve_dim = self._num_angles if self._curve_repr == "raw" else 2 * self._fft_k + 1
        self._single_state_dim = self._curve_dim + 2 + 2 * self._poi_max_obs + 3
        self._state_dim = self._single_state_dim * self._stack_len
        self._theta_cpu = torch.linspace(0.0, 2.0 * math.pi, self._num_angles + 1)[:-1]
        self._subsample_t_cpu = torch.arange(self._traj_subsamples, dtype=torch.float32) / float(self._traj_subsamples)
        self._reward_t_theta_weight = float(cfg.get("traj_reward_t_theta_weight", 0.30))
        self._kernel_cutoff = float(cfg.get("traj_kernel_cutoff", 2.0))
        if self._privileged_stats_in_obs or self._obs_mode_task != "partial":
            raise ValueError("This task requires partial observations and visible-only statistics.")
        if self._obs_vector_mode != "compact":
            raise ValueError("This task uses compact state observations.")

    def _action_dim(self):
        return 3

    def _make_single_action_space(self):
        dim = int(self._action_dim())
        return gym.spaces.Box(
            low=-np.ones(dim, dtype=np.float32),
            high=np.ones(dim, dtype=np.float32),
            dtype=np.float32,
        )

    def _ensure_storage(self):
        n = int(self.num_envs)
        d = self.device
        if hasattr(self, "_u") and self._u.shape[0] == n:
            return
        self._theta = self._theta_cpu.to(d)
        self._cos_theta = torch.cos(self._theta)
        self._sin_theta = torch.sin(self._theta)
        self._subsample_t = self._subsample_t_cpu.to(d)
        self._u = torch.zeros((n, self._num_angles), device=d)
        self._center_offset = torch.zeros((n, 2), device=d)
        self._poi_num = torch.zeros((n,), dtype=torch.long, device=d)
        self._poi_mask = torch.zeros((n, self._poi_max_obs), dtype=torch.bool, device=d)
        self._poi_pos = torch.zeros((n, self._poi_max_obs, 2), device=d)
        self._poi_hotspots = torch.zeros((n, self._poi_hotspot_num, 2), device=d)
        self._poi_target_hotspot = torch.zeros((n, self._poi_max_obs), dtype=torch.long, device=d)
        self._poi_v_rand = torch.zeros((n, self._poi_max_obs, 2), device=d)
        self._poi_retarget_timer = torch.zeros((n,), dtype=torch.long, device=d)
        self._poi_points_obs = torch.zeros((n, self._poi_max_obs, 2), device=d)
        self._poi_mask_obs = torch.zeros((n, self._poi_max_obs, 1), dtype=torch.float32, device=d)
        self._uav_phase = torch.zeros((n, self._uav_num), device=d)
        self._uav_pos = torch.zeros((n, self._uav_num, 2), device=d)
        self._uav_heading = torch.zeros((n, self._uav_num), device=d)
        self._uav_ever_covered = torch.zeros((n, self._poi_max_obs), dtype=torch.bool, device=d)
        self._state_history = torch.zeros((n, self._stack_len, self._single_state_dim), device=d)
        self._stacked_state = torch.zeros((n, self._state_dim), device=d)
        self._prev_mean_poi_dist = torch.zeros((n,), device=d)
        self._latest_reward = torch.zeros((n,), device=d)
        self._latest_metrics = {}
        self._ensure_source_rng_state()

    def _ensure_source_rng_state(self):
        n = int(getattr(self, "num_envs", 1))
        seed = int(getattr(self, "_source_seed", self._traj_cfg.get("seed", 0) if hasattr(self, "_traj_cfg") else 0))
        if not hasattr(self, "_source_np_rngs") or len(self._source_np_rngs) != n:
            self._source_seed = seed
            self._source_np_rngs = [np.random.RandomState(seed + i) for i in range(n)]
        if not hasattr(self, "_prev_observed_poi_ids") or len(self._prev_observed_poi_ids) != n:
            self._prev_observed_poi_ids = [set() for _ in range(n)]
            self._ever_observed_poi_ids = [set() for _ in range(n)]

    def _set_source_seed(self, seed: int):
        seed = int(seed)
        self._source_seed = seed
        n = int(getattr(self, "num_envs", 1))
        self._source_np_rngs = [np.random.RandomState(seed + i) for i in range(n)]
        self._prev_observed_poi_ids = [set() for _ in range(n)]
        self._ever_observed_poi_ids = [set() for _ in range(n)]

    def _source_env_ids(self, env_idx: torch.Tensor):
        return [int(x) for x in env_idx.detach().cpu().numpy().reshape(-1).tolist()]

    def _project_point_to_feasible_np(self, point: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        p = np.asarray(point, dtype=np.float32).reshape(2).copy()
        R = float(self._domain_radius)
        for dim in range(2):
            if p[dim] > R:
                p[dim] = 2.0 * R - p[dim]
            if p[dim] < -R:
                p[dim] = -2.0 * R - p[dim]
        p = np.clip(p, -R, R).astype(np.float32)
        centers = getattr(self, "_nofly_centers_np", None)
        radii = getattr(self, "_nofly_radii_np", None)
        if centers is None or radii is None or int(self._nofly_num) <= 0:
            return p
        eps = float(1e-3 * self._r_soft)
        for c, r in zip(centers, radii):
            d = p - c
            norm = float(np.linalg.norm(d))
            if norm < float(r):
                if norm < 1e-6:
                    d = rng.randn(2).astype(np.float32)
                    norm = float(np.linalg.norm(d))
                p = c + d / float(max(norm, 1e-6)) * float(r + eps)
        return np.clip(p, -R, R).astype(np.float32)

    def _inside_any_nofly_np(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        centers = getattr(self, "_nofly_centers_np", None)
        radii = getattr(self, "_nofly_radii_np", None)
        if centers is None or radii is None or int(self._nofly_num) <= 0:
            return np.zeros((pts.shape[0],), dtype=bool)
        diff = pts[:, None, :] - centers[None, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        return np.any(dist_sq <= (radii[None, :] ** 2), axis=1)

    def _sample_hotspots_np(self, rng: np.random.RandomState, num_hotspots: int) -> np.ndarray:
        R = float(self._domain_radius)
        num_hotspots = int(max(1, num_hotspots))
        d_min = float(max(self._poi_hotspot_min_dist, 0.5 * self._r_soft))
        r_min = float(np.clip(self._poi_hotspot_radius_min_ratio * R, 0.0, R))
        r_max = float(np.clip(max(self._poi_hotspot_radius_max_ratio * R, r_min), 0.0, R))
        hotspots = []
        for _ in range(num_hotspots):
            cand = None
            for _try in range(500):
                theta = float(rng.uniform(0.0, 2.0 * math.pi))
                radius = float(rng.uniform(r_min, r_max))
                p = np.array([radius * math.cos(theta), radius * math.sin(theta)], dtype=np.float32)
                p = np.clip(p + rng.randn(2).astype(np.float32) * float(0.05 * self._r_soft), -R, R).astype(np.float32)
                if bool(self._inside_any_nofly_np(p[None, :])[0]):
                    continue
                if all(float(np.linalg.norm(p - q)) >= d_min for q in hotspots):
                    cand = p
                    break
            if cand is None:
                phase = float(rng.uniform(0.0, 2.0 * math.pi))
                idx = len(hotspots)
                theta = phase + 2.0 * math.pi * float(idx) / float(max(num_hotspots, 1))
                radius = float(0.5 * (r_min + r_max))
                cand = self._project_point_to_feasible_np(
                    np.array([radius * math.cos(theta), radius * math.sin(theta)], dtype=np.float32),
                    rng,
                )
            hotspots.append(cand)
        return np.stack(hotspots, axis=0).astype(np.float32)

    def _retarget_poi_targets_np(
        self,
        rng: np.random.RandomState,
        current_targets: np.ndarray,
        poi_num: int,
        force_change: bool = True,
    ) -> np.ndarray:
        K = int(self._poi_hotspot_num)
        N = int(poi_num)
        targets = np.asarray(current_targets, dtype=np.int64).reshape(-1)[: max(N, 0)].copy()
        if N <= 0 or K <= 0:
            return targets
        if K == 1:
            return np.zeros((N,), dtype=np.int64)
        min_unique = int(min(max(self._poi_target_min_unique, 1), K, N))
        new_targets = np.full((N,), -1, dtype=np.int64)
        counts = np.zeros((K,), dtype=np.int64)
        order = rng.permutation(N) if N > 1 else np.arange(N, dtype=np.int64)
        for i in order:
            cur = int(targets[int(i)]) if int(i) < targets.shape[0] else 0
            candidates = list(range(K))
            if force_change and K > 1:
                candidates = [c for c in candidates if c != cur]
            cand_counts = np.array([counts[c] for c in candidates], dtype=np.int64)
            min_count = int(cand_counts.min()) if cand_counts.size > 0 else 0
            pool = [c for c in candidates if int(counts[c]) == min_count]
            chosen = int(rng.choice(pool)) if len(pool) > 0 else int(cur)
            new_targets[int(i)] = chosen
            counts[chosen] += 1
        uniq = set(int(x) for x in new_targets.tolist() if int(x) >= 0)
        if len(uniq) < min_unique:
            missing = [k for k in range(K) if k not in uniq]
            for k in missing:
                src = int(np.argmax(counts))
                idxs = np.where(new_targets == src)[0]
                if idxs.size == 0:
                    continue
                pick = int(rng.choice(idxs))
                if force_change and int(targets[pick]) == k and K > 1:
                    alt = [j for j in idxs if int(targets[int(j)]) != k]
                    if len(alt) > 0:
                        pick = int(rng.choice(alt))
                new_targets[pick] = int(k)
                counts[src] -= 1
                counts[k] += 1
        return new_targets.astype(np.int64)

    def _apply_scripted_poi_initial_positions(self, env_idx: torch.Tensor) -> None:
        if self._poi_motion_mode != "scripted_endpoints":
            return
        count = int(self._poi_script_start_xy.shape[0])
        start = torch.as_tensor(self._poi_script_start_xy, dtype=torch.float32, device=self.device)
        self._poi_num[env_idx] = count
        arange_p = torch.arange(self._poi_max_obs, device=self.device).view(1, -1)
        mask = arange_p < count
        self._poi_mask[env_idx] = mask.expand(len(env_idx), -1)
        self._poi_pos[env_idx] = 0.0
        self._poi_pos[env_idx, :count] = start.unsqueeze(0).expand(len(env_idx), -1, -1)
        self._poi_v_rand[env_idx] = 0.0

    def _advance_scripted_poi_dynamics(self) -> None:
        progress = (self._elapsed_steps.float() / float(self._poi_script_duration_steps)).clamp(0.0, 1.0)
        if self._poi_script_easing == "smoothstep":
            progress = progress * progress * (3.0 - 2.0 * progress)
        start = torch.as_tensor(self._poi_script_start_xy, dtype=torch.float32, device=self.device)
        end = torch.as_tensor(self._poi_script_end_xy, dtype=torch.float32, device=self.device)
        pos = start.unsqueeze(0) + progress.view(-1, 1, 1) * (end - start).unsqueeze(0)
        self._poi_pos[:, : start.shape[0]] = pos

    def _initialize_episode_source_compatible(self, env_idx: torch.Tensor, options: dict):
        self._ensure_source_rng_state()
        ids = self._source_env_ids(env_idx)
        base_radius = float(np.clip(self._kernel_sigma_phys, self._radius_min, self._radius_max))
        arange_p = torch.arange(self._poi_max_obs, device=self.device)
        for env_i in ids:
            rng = self._source_np_rngs[env_i]
            self._u[env_i] = math.log(base_radius)
            self._center_offset[env_i] = 0.0
            if self._poi_num_min == self._poi_num_max:
                poi_num_i = int(max(0, self._poi_num_min))
            elif self._poi_num_max <= 0:
                poi_num_i = 0
            else:
                poi_num_i = int(rng.randint(self._poi_num_min, self._poi_num_max + 1))
            self._poi_num[env_i] = poi_num_i
            mask_i = arange_p < int(poi_num_i)
            self._poi_mask[env_i] = mask_i
            self._poi_pos[env_i] = 0.0
            self._poi_hotspots[env_i] = 0.0
            self._poi_target_hotspot[env_i] = 0
            self._poi_v_rand[env_i] = 0.0
            if poi_num_i > 0:
                hotspots = self._sample_hotspots_np(rng, self._poi_hotspot_num)
                current_targets = np.zeros((poi_num_i,), dtype=np.int64)
                targets = self._retarget_poi_targets_np(rng, current_targets, poi_num_i, force_change=False)
                pos = np.zeros((self._poi_max_obs, 2), dtype=np.float32)
                for poi_i in range(poi_num_i):
                    base = hotspots[int(targets[poi_i])]
                    pos_i = None
                    for _try in range(200):
                        noise = rng.randn(2).astype(np.float32) * float(0.35 * self._r_soft)
                        cand = self._project_point_to_feasible_np(base + noise, rng)
                        if poi_i == 0 or self._poi_init_min_dist <= 0.0:
                            pos_i = cand
                            break
                        dist_min = float(np.min(np.linalg.norm(pos[:poi_i] - cand[None, :], axis=1)))
                        if dist_min >= float(self._poi_init_min_dist):
                            pos_i = cand
                            break
                    if pos_i is None:
                        pos_i = self._project_point_to_feasible_np(base, rng)
                    pos[poi_i] = pos_i
                self._poi_hotspots[env_i] = torch.as_tensor(hotspots, dtype=torch.float32, device=self.device)
                self._poi_target_hotspot[env_i, :poi_num_i] = torch.as_tensor(targets, dtype=torch.long, device=self.device)
                self._poi_pos[env_i] = torch.as_tensor(pos, dtype=torch.float32, device=self.device)
            self._poi_retarget_timer[env_i] = 0
            self._prev_observed_poi_ids[env_i] = set()
            self._ever_observed_poi_ids[env_i] = set()

        self._apply_scripted_poi_initial_positions(env_idx)
        self._reset_uav_observers(env_idx)
        self._uav_ever_covered[env_idx] = False
        self._sample_poi_observation_source(env_idx, force=True, use_trajectory_observer=True)
        single = self._single_state(env_idx)
        self._state_history[env_idx] = single.unsqueeze(1).expand(-1, self._stack_len, -1)
        self._refresh_stacked_state(env_idx)
        rho_bar, w_dir = self._directional_target(env_idx)
        traj = self._trajectory_points(env_idx)
        dist = self._poi_traj_distance(env_idx, traj)
        mean_dist = self._masked_mean(dist.min(dim=-1).values, self._poi_mask[env_idx], dim=1)
        self._prev_mean_poi_dist[env_idx] = mean_dist
        self._latest_reward[env_idx] = 0.0
        self._update_metrics(env_idx, mean_dist=mean_dist, reward=torch.zeros_like(mean_dist), rho_bar=rho_bar)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        self._ensure_storage()
        env_idx = env_idx.to(self.device)
        if self._source_compatible_rng:
            self._initialize_episode_source_compatible(env_idx, options)
            return
        b = len(env_idx)
        base_radius = float(np.clip(self._kernel_sigma_phys, self._radius_min, self._radius_max))
        self._u[env_idx] = math.log(base_radius)
        self._center_offset[env_idx] = 0.0
        if self._poi_num_min == self._poi_num_max:
            poi_num = torch.full((b,), self._poi_num_min, dtype=torch.long, device=self.device)
        else:
            poi_num = torch.randint(self._poi_num_min, self._poi_num_max + 1, (b,), device=self.device)
        self._poi_num[env_idx] = poi_num
        arange_p = torch.arange(self._poi_max_obs, device=self.device).view(1, -1)
        mask = arange_p < poi_num.view(-1, 1)
        self._poi_mask[env_idx] = mask
        self._sample_hotspots(env_idx)
        self._retarget_pois(env_idx)
        targets = self._poi_target_hotspot[env_idx].clamp(0, self._poi_hotspot_num - 1)
        hotspots = self._poi_hotspots[env_idx]
        base = torch.gather(hotspots, 1, targets.unsqueeze(-1).expand(-1, -1, 2))
        noise = torch.randn((b, self._poi_max_obs, 2), device=self.device) * (0.35 * self._r_soft)
        pos = (base + noise).clamp(-self._domain_radius, self._domain_radius)
        self._poi_pos[env_idx] = torch.where(mask.unsqueeze(-1), pos, torch.zeros_like(pos))
        self._apply_scripted_poi_initial_positions(env_idx)
        self._poi_v_rand[env_idx] = 0.0
        self._poi_retarget_timer[env_idx] = 0
        self._reset_uav_observers(env_idx)
        self._uav_ever_covered[env_idx] = False
        self._sample_poi_observation(env_idx, force=True)
        single = self._single_state(env_idx)
        self._state_history[env_idx] = single.unsqueeze(1).expand(-1, self._stack_len, -1)
        self._refresh_stacked_state(env_idx)
        rho_bar, w_dir = self._directional_target(env_idx)
        traj = self._trajectory_points(env_idx)
        dist = self._poi_traj_distance(env_idx, traj)
        mean_dist = self._masked_mean(dist.min(dim=-1).values, self._poi_mask[env_idx], dim=1)
        self._prev_mean_poi_dist[env_idx] = mean_dist
        self._latest_reward[env_idx] = 0.0
        self._update_metrics(env_idx, mean_dist=mean_dist, reward=torch.zeros_like(mean_dist), rho_bar=rho_bar)

    def _sample_hotspots(self, env_idx):
        b = len(env_idx)
        rmin = self._poi_hotspot_radius_min_ratio * self._domain_radius
        rmax = self._poi_hotspot_radius_max_ratio * self._domain_radius
        theta = 2.0 * math.pi * torch.rand((b, self._poi_hotspot_num), device=self.device)
        radius = rmin + (rmax - rmin) * torch.rand((b, self._poi_hotspot_num), device=self.device)
        pts = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=-1)
        pts = pts + 0.05 * self._r_soft * torch.randn_like(pts)
        self._poi_hotspots[env_idx] = pts.clamp(-self._domain_radius, self._domain_radius)

    def _retarget_pois(self, env_idx):
        b = len(env_idx)
        base = torch.arange(self._poi_max_obs, device=self.device).view(1, -1)
        offset = torch.randint(0, self._poi_hotspot_num, (b, 1), device=self.device)
        self._poi_target_hotspot[env_idx] = (base + offset) % self._poi_hotspot_num

    def get_obs(self, info=None, unflattened: bool = False):
        self._ensure_storage()
        if self.obs_mode == "none":
            return {}
        if self.obs_mode == "state_dict" or unflattened:
            return OrderedDict(state=self._stacked_state)
        return self._stacked_state

    def get_info(self):
        self._ensure_storage()
        info = dict(elapsed_steps=self._elapsed_steps.clone())
        for key, value in self._latest_metrics.items():
            info[key] = value.clone()
        return info

    def get_state_dict(self):
        return OrderedDict()

    def set_state_dict(self, state: dict, env_idx: torch.Tensor = None):
        return None

    def step(self, action):
        self._ensure_storage()
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[0] != self.num_envs:
            action = action.expand(self.num_envs, -1)
        reward_total = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        for _ in range(self._action_repeat):
            self._elapsed_steps += 1
            self._advance_dynamics(action)
            reward_total = reward_total + self._compute_reward()
            self._advance_uav_observers(torch.arange(self.num_envs, device=self.device))
            self._sample_poi_observation(torch.arange(self.num_envs, device=self.device), force=False)
        self._latest_reward = reward_total
        single = self._single_state(torch.arange(self.num_envs, device=self.device))
        self._state_history = torch.roll(self._state_history, shifts=-1, dims=1)
        self._state_history[:, -1] = single
        self._refresh_stacked_state(torch.arange(self.num_envs, device=self.device))
        obs = self.get_obs()
        terminated = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        truncated = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        info = self.get_info()
        self._last_obs = obs
        return obs, reward_total, terminated, truncated, info

    def _advance_poi_dynamics_source_compatible(self):
        self._ensure_source_rng_state()
        env_ids = range(int(self.num_envs))
        if self._poi_retarget_interval > 0:
            self._poi_retarget_timer += 1
        inertia = float(self._poi_inertia)
        noise_scale = math.sqrt(max(1.0 - inertia ** 2, 0.0))
        v0 = float(self._poi_hotspot_speed)
        eps_arrive = float(self._poi_hotspot_arrival_tol)
        for env_i in env_ids:
            poi_num_i = int(self._poi_num[env_i].detach().cpu().item())
            if poi_num_i <= 0:
                continue
            rng = self._source_np_rngs[env_i]
            if self._poi_retarget_interval > 0 and int(self._poi_retarget_timer[env_i].detach().cpu().item()) >= int(self._poi_retarget_interval):
                current = self._poi_target_hotspot[env_i, :poi_num_i].detach().cpu().numpy().astype(np.int64)
                targets = self._retarget_poi_targets_np(rng, current, poi_num_i, force_change=True)
                self._poi_target_hotspot[env_i, :poi_num_i] = torch.as_tensor(targets, dtype=torch.long, device=self.device)
                self._poi_retarget_timer[env_i] = 0
            pos_np = self._poi_pos[env_i, :poi_num_i].detach().cpu().numpy().astype(np.float32)
            v_np = self._poi_v_rand[env_i, :poi_num_i].detach().cpu().numpy().astype(np.float32)
            hotspots_np = self._poi_hotspots[env_i].detach().cpu().numpy().astype(np.float32)
            targets_np = self._poi_target_hotspot[env_i, :poi_num_i].detach().cpu().numpy().astype(np.int64)
            for poi_i in range(poi_num_i):
                z = int(np.clip(targets_np[poi_i], 0, self._poi_hotspot_num - 1))
                delta = (hotspots_np[z] - pos_np[poi_i]).astype(np.float32)
                dist = float(np.linalg.norm(delta))
                v_hop = np.zeros(2, dtype=np.float32)
                if dist > eps_arrive:
                    v_hop = (delta / max(dist, 1e-6)) * v0
                noise = rng.randn(2).astype(np.float32)
                v_np[poi_i] = inertia * v_np[poi_i] + noise_scale * float(self._poi_random_strength) * noise
                pos_np[poi_i] = self._project_point_to_feasible_np(
                    pos_np[poi_i] + (v_hop + v_np[poi_i]) * float(self._poi_dt),
                    rng,
                )
            self._poi_pos[env_i, :poi_num_i] = torch.as_tensor(pos_np, dtype=torch.float32, device=self.device)
            self._poi_v_rand[env_i, :poi_num_i] = torch.as_tensor(v_np, dtype=torch.float32, device=self.device)

    def _advance_dynamics(self, action):
        self._advance_orbit_action(action)
        self._advance_poi_dynamics()

    def _advance_orbit_action(self, action):
        phi = action[:, 0] * self._phi_max
        alpha = action[:, 1] * self._alpha_max
        lam = self._lambda_min + 0.5 * (action[:, 2] + 1.0) * (self._lambda_max - self._lambda_min)
        delta = (self._theta.view(1, -1) - phi.view(-1, 1) + math.pi) % (2.0 * math.pi) - math.pi
        weight = torch.exp(-0.5 * (delta / lam.clamp_min(1e-3).view(-1, 1)) ** 2)
        inc = torch.log((1.0 + alpha.view(-1, 1) * weight).clamp_min(1e-6))
        radius = torch.exp(self._u + inc).clamp(self._radius_min, self._radius_max)
        self._u = torch.log(radius)

    def _advance_poi_dynamics(self):
        if not self._poi_dynamic_enabled:
            return
        if self._poi_motion_mode == "scripted_endpoints":
            self._advance_scripted_poi_dynamics()
            return
        if self._source_compatible_rng:
            self._advance_poi_dynamics_source_compatible()
            return
        if self._poi_retarget_interval > 0:
            self._poi_retarget_timer += 1
            due = torch.nonzero(self._poi_retarget_timer >= self._poi_retarget_interval, as_tuple=False).flatten()
            if due.numel() > 0:
                self._retarget_pois(due)
                self._poi_retarget_timer[due] = 0
        target = torch.gather(
            self._poi_hotspots,
            1,
            self._poi_target_hotspot.clamp(0, self._poi_hotspot_num - 1).unsqueeze(-1).expand(-1, -1, 2),
        )
        diff = target - self._poi_pos
        dist = torch.linalg.norm(diff, dim=-1, keepdim=True)
        drift = torch.where(dist > self._poi_hotspot_arrival_tol, diff / dist.clamp_min(1e-6) * self._poi_hotspot_speed, torch.zeros_like(diff))
        inertia = self._poi_inertia
        noise_scale = math.sqrt(max(1.0 - inertia * inertia, 0.0))
        self._poi_v_rand = inertia * self._poi_v_rand + noise_scale * self._poi_random_strength * torch.randn_like(self._poi_v_rand)
        pos = self._poi_pos + (drift + self._poi_v_rand) * self._poi_dt
        pos = pos.clamp(-self._domain_radius, self._domain_radius)
        self._poi_pos = torch.where(self._poi_mask.unsqueeze(-1), pos, torch.zeros_like(pos))

    def _trajectory_points(self, env_idx):
        radius = torch.exp(self._u[env_idx])
        base = torch.stack(
            [
                radius * self._cos_theta.view(1, -1),
                radius * self._sin_theta.view(1, -1),
            ],
            dim=-1,
        )
        base = base + self._center_offset[env_idx].view(-1, 1, 2)
        nxt = torch.roll(base, shifts=-1, dims=1)
        t = self._subsample_t.view(1, 1, -1, 1)
        pts = (1.0 - t) * base.unsqueeze(2) + t * nxt.unsqueeze(2)
        return pts.reshape(len(env_idx), self._num_angles * self._traj_subsamples, 2)

    def _poi_traj_distance(self, env_idx, traj):
        pts = self._poi_pos[env_idx]
        return torch.linalg.norm(pts.unsqueeze(2) - traj.unsqueeze(1), dim=-1)

    def _reset_uav_observers(self, env_idx):
        if self._obs_source != "uav":
            return
        b = len(env_idx)
        if b <= 0:
            return
        base_phase = torch.linspace(
            0.0,
            2.0 * math.pi,
            self._uav_num + 1,
            device=self.device,
        )[:-1].view(1, -1)
        offset = 2.0 * math.pi * torch.rand((b, 1), device=self.device)
        phase = (base_phase + offset) % (2.0 * math.pi)
        radius = max(self._uav_init_radius_ratio * self._domain_radius, 0.1)
        pos = radius * torch.stack([torch.cos(phase), torch.sin(phase)], dim=-1)
        if self._uav_init_noise_ratio > 0.0:
            pos = pos + self._uav_init_noise_ratio * self._domain_radius * torch.randn_like(pos)
        self._uav_phase[env_idx] = phase
        self._uav_pos[env_idx] = pos.clamp(-self._domain_radius, self._domain_radius)
        self._uav_heading[env_idx] = (phase + 0.5 * math.pi) % (2.0 * math.pi)

    def _advance_uav_observers(self, env_idx):
        if self._obs_source != "uav" or env_idx.numel() == 0:
            return
        phase = (self._uav_phase[env_idx] + self._uav_phase_speed) % (2.0 * math.pi)
        radius_all = torch.exp(self._u[env_idx])
        angle_pos = phase / (2.0 * math.pi) * float(self._num_angles)
        idx0 = torch.floor(angle_pos).long() % self._num_angles
        idx1 = (idx0 + 1) % self._num_angles
        frac = (angle_pos - torch.floor(angle_pos)).unsqueeze(-1)
        r0 = torch.gather(radius_all, 1, idx0)
        r1 = torch.gather(radius_all, 1, idx1)
        radius = (1.0 - frac[..., 0]) * r0 + frac[..., 0] * r1
        target = torch.stack([radius * torch.cos(phase), radius * torch.sin(phase)], dim=-1)
        target = target + self._center_offset[env_idx].unsqueeze(1)
        delta = target - self._uav_pos[env_idx]
        dist = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
        max_step = float(self._uav_speed * self._uav_dt)
        scale = torch.clamp(max_step / dist, max=1.0)
        self._uav_pos[env_idx] = (self._uav_pos[env_idx] + delta * scale).clamp(
            -self._domain_radius,
            self._domain_radius,
        )
        self._uav_phase[env_idx] = phase
        self._uav_heading[env_idx] = torch.atan2(delta[..., 1], delta[..., 0])

    def _sample_poi_observation_source(
        self,
        env_idx,
        force=False,
        observer_points=None,
        reset_observation_bookkeeping=False,
        use_trajectory_observer=False,
    ):
        if env_idx.numel() == 0:
            return
        self._ensure_source_rng_state()
        if not force:
            refresh = (self._elapsed_steps[env_idx] % self._poi_sample_interval) == 0
            skipped = env_idx[~refresh]
            if skipped.numel() > 0:
                self._set_metric(skipped, "newly_observed_pois", torch.zeros((len(skipped),), device=self.device))
            env_idx = env_idx[refresh]
            if env_idx.numel() == 0:
                return
        if reset_observation_bookkeeping:
            for env_i in self._source_env_ids(env_idx):
                self._prev_observed_poi_ids[env_i] = set()
                self._ever_observed_poi_ids[env_i] = set()
        observer_np = None
        if observer_points is not None:
            observer_np = np.asarray(observer_points, dtype=np.float32).reshape(-1, 2)
        for env_i in self._source_env_ids(env_idx):
            rng = self._source_np_rngs[env_i]
            poi_num_i = int(self._poi_num[env_i].detach().cpu().item())
            self._poi_points_obs[env_i] = 0.0
            self._poi_mask_obs[env_i, :, 0] = 0.0
            if poi_num_i <= 0:
                self._set_metric(torch.tensor([env_i], device=self.device), "num_observed_pois", torch.zeros((1,), device=self.device))
                self._set_metric(torch.tensor([env_i], device=self.device), "observation_coverage", torch.zeros((1,), device=self.device))
                self._set_metric(torch.tensor([env_i], device=self.device), "num_discovered_pois", torch.zeros((1,), device=self.device))
                self._set_metric(torch.tensor([env_i], device=self.device), "newly_observed_pois", torch.zeros((1,), device=self.device))
                self._set_metric(torch.tensor([env_i], device=self.device), "mean_obs_noise", torch.zeros((1,), device=self.device))
                continue
            pts_true = self._poi_pos[env_i, :poi_num_i].detach().cpu().numpy().astype(np.float32)
            if self._obs_mode_task == "full":
                idx = np.arange(pts_true.shape[0], dtype=np.int64)
            else:
                if observer_np is not None:
                    dist = np.sqrt(np.sum((pts_true[:, None, :] - observer_np[None, :, :]) ** 2, axis=-1)).astype(np.float32)
                elif self._obs_source == "uav" and not use_trajectory_observer:
                    obs_pts = self._uav_pos[env_i].detach().cpu().numpy().astype(np.float32)
                    dist = np.sqrt(np.sum((pts_true[:, None, :] - obs_pts[None, :, :]) ** 2, axis=-1)).astype(np.float32)
                else:
                    idx_t = torch.tensor([env_i], dtype=torch.long, device=self.device)
                    traj = self._trajectory_points(idx_t)[0].detach().cpu().numpy().astype(np.float32)
                    dist = np.sqrt(np.sum((pts_true[:, None, :] - traj[None, :, :]) ** 2, axis=-1)).astype(np.float32)
                dists = np.min(dist, axis=1).astype(np.float32) if dist.ndim == 2 and dist.shape[1] > 0 else np.full((pts_true.shape[0],), np.inf, dtype=np.float32)
                idx = np.where(dists < float(self._r_soft))[0].astype(np.int64)
            if idx.size > 1:
                idx = rng.permutation(idx)
            if idx.size > 0 and self._obs_dropout_prob > 0.0:
                idx = idx[rng.rand(idx.size) > self._obs_dropout_prob]
            n = int(min(idx.size, self._poi_max_obs))
            selected_idx = idx[:n] if n > 0 else np.zeros((0,), dtype=np.int64)
            mean_obs_noise = 0.0
            if n > 0:
                obs_pts_true = pts_true[selected_idx]
                obs_pts = obs_pts_true.copy()
                if self._obs_noise_std > 0.0:
                    noise = rng.randn(n, 2).astype(np.float32) * float(self._obs_noise_std)
                    obs_pts = np.clip(obs_pts + noise, -self._domain_radius, self._domain_radius).astype(np.float32)
                    mean_obs_noise = float(np.mean(np.linalg.norm(obs_pts - obs_pts_true, axis=1)))
                self._poi_points_obs[env_i, :n] = torch.as_tensor(obs_pts, dtype=torch.float32, device=self.device)
                self._poi_mask_obs[env_i, :n, 0] = 1.0
            current_ids = set(int(x) for x in selected_idx.tolist())
            newly = len(current_ids.difference(self._prev_observed_poi_ids[env_i]))
            self._prev_observed_poi_ids[env_i] = current_ids
            self._ever_observed_poi_ids[env_i].update(current_ids)
            idx_tensor = torch.tensor([env_i], dtype=torch.long, device=self.device)
            self._set_metric(idx_tensor, "num_observed_pois", torch.tensor([float(len(current_ids))], device=self.device))
            self._set_metric(idx_tensor, "observation_coverage", torch.tensor([float(len(current_ids)) / float(max(poi_num_i, 1))], device=self.device))
            self._set_metric(idx_tensor, "num_discovered_pois", torch.tensor([float(len(self._ever_observed_poi_ids[env_i]))], device=self.device))
            self._set_metric(idx_tensor, "newly_observed_pois", torch.tensor([float(newly)], device=self.device))
            self._set_metric(idx_tensor, "mean_obs_noise", torch.tensor([float(mean_obs_noise)], device=self.device))

    def _sample_poi_observation(self, env_idx, force=False, observer_points=None, reset_observation_bookkeeping=False):
        if self._source_compatible_rng:
            self._sample_poi_observation_source(
                env_idx,
                force=force,
                observer_points=observer_points,
                reset_observation_bookkeeping=reset_observation_bookkeeping,
            )
            return
        if env_idx.numel() == 0:
            return
        if not force:
            refresh = (self._elapsed_steps[env_idx] % self._poi_sample_interval) == 0
            env_idx = env_idx[refresh]
            if env_idx.numel() == 0:
                return
        b = len(env_idx)
        mask = self._poi_mask[env_idx]
        visible = mask.clone()
        if self._obs_mode_task == "partial":
            if self._obs_source == "uav":
                observer = self._uav_pos[env_idx]
                dist = torch.linalg.norm(
                    self._poi_pos[env_idx].unsqueeze(2) - observer.unsqueeze(1),
                    dim=-1,
                ).min(dim=-1).values
            else:
                traj = self._trajectory_points(env_idx)
                dist = self._poi_traj_distance(env_idx, traj).min(dim=-1).values
            visible = visible & (dist < self._r_soft)
        if self._obs_dropout_prob > 0.0:
            visible = visible & (torch.rand((b, self._poi_max_obs), device=self.device) > self._obs_dropout_prob)
        order_score = torch.rand((b, self._poi_max_obs), device=self.device).masked_fill(~visible, 2.0)
        order = torch.argsort(order_score, dim=1)
        pts = torch.gather(self._poi_pos[env_idx], 1, order.unsqueeze(-1).expand(-1, -1, 2))
        vis_sorted = torch.gather(visible, 1, order)
        pts = torch.where(vis_sorted.unsqueeze(-1), pts, torch.zeros_like(pts))
        if self._obs_noise_std > 0.0:
            noise = torch.randn_like(pts) * self._obs_noise_std
            pts = torch.where(vis_sorted.unsqueeze(-1), (pts + noise).clamp(-self._domain_radius, self._domain_radius), pts)
        self._poi_points_obs[env_idx] = pts
        self._poi_mask_obs[env_idx, :, 0] = vis_sorted.float()
        observed = vis_sorted.sum(dim=1).float()
        denom = self._poi_num[env_idx].clamp_min(1).float()
        self._set_metric(env_idx, "num_observed_pois", observed)
        self._set_metric(env_idx, "observation_coverage", observed / denom)
        self._set_metric(env_idx, "num_discovered_pois", observed)
        self._set_metric(env_idx, "newly_observed_pois", observed)
        self._set_metric(env_idx, "mean_obs_noise", torch.full((b,), self._obs_noise_std, device=self.device))

    def _single_state(self, env_idx):
        radius = torch.exp(self._u[env_idx])
        if self._curve_repr == "raw":
            curve = ((radius - self._radius_min) / max(self._radius_max - self._radius_min, 1e-6)).clamp(0.0, 1.0)
        else:
            mid = 0.5 * (self._radius_max + self._radius_min)
            half = max(0.5 * (self._radius_max - self._radius_min), 1e-6)
            r_norm = ((radius - mid) / half).clamp(-1.0, 1.0)
            spectrum = torch.fft.rfft(r_norm, dim=-1) / float(self._num_angles)
            parts = [spectrum[:, :1].real]
            coeffs = spectrum[:, 1 : self._fft_k + 1]
            interleaved = torch.stack([coeffs.real, coeffs.imag], dim=-1).reshape(len(env_idx), -1)
            parts.append(interleaved)
            curve = torch.cat(parts, dim=-1).clamp(-1.0, 1.0)

        center = (self._center_offset[env_idx] / self._domain_radius).clamp(-1.0, 1.0)
        poi = (self._poi_points_obs[env_idx] / self._domain_radius).clamp(-1.0, 1.0).reshape(len(env_idx), -1)
        count_ratio = self._poi_mask_obs[env_idx, :, 0].mean(dim=1, keepdim=True)

        mean_dist, min_cov = self._observed_poi_traj_stats(env_idx)
        mean_dist = (mean_dist / self._domain_radius).clamp(0.0, 4.0).unsqueeze(-1)
        min_cov = min_cov.clamp(0.0, 1.0).unsqueeze(-1)

        return torch.cat([curve, center, poi, count_ratio, mean_dist, min_cov], dim=-1).float()

    def _refresh_stacked_state(self, env_idx):
        self._stacked_state[env_idx] = self._state_history[env_idx].reshape(len(env_idx), -1)

    def _directional_target(self, env_idx):
        radius = torch.exp(self._u[env_idx])
        r_mean = radius.mean(dim=1, keepdim=True)
        rel = self._poi_pos[env_idx] - self._center_offset[env_idx].unsqueeze(1)
        rho = torch.linalg.norm(rel, dim=-1).clamp(self._radius_min, self._radius_max)
        angle = torch.atan2(rel[..., 1], rel[..., 0])
        dtheta = (self._theta.view(1, 1, -1) - angle.unsqueeze(-1) + math.pi) % (2.0 * math.pi) - math.pi
        weights = torch.exp(-0.5 * (dtheta / max(self._directional_target_sigma, 1e-6)) ** 2)
        weights = weights * self._poi_mask[env_idx].unsqueeze(-1).float()
        w_dir = weights.sum(dim=1)
        rho_bar = (weights * rho.unsqueeze(-1)).sum(dim=1) / w_dir.clamp_min(self._psi_eps)
        rho_bar = torch.where(w_dir > self._psi_eps, rho_bar, r_mean.expand_as(rho_bar))
        return rho_bar.clamp(self._radius_min, self._radius_max), w_dir


    def _compute_reward(self):
        env_idx = torch.arange(self.num_envs, device=self.device)
        traj = self._trajectory_points(env_idx)
        dist = self._poi_traj_distance(env_idx, traj)
        min_dist = dist.min(dim=-1).values
        mean_dist = self._masked_mean(min_dist, self._poi_mask, dim=1)
        reward_abs = -self._reward_abs_weight * mean_dist

        t_theta = self._compute_t_theta_from_dist(dist, self._poi_mask)
        t_theta_error = 1.0 - t_theta
        reward_t_theta = -self._reward_t_theta_weight * t_theta_error
        reward = reward_abs + reward_t_theta

        covered = (dist < self._min_poi_coverage_radius).float().mean(dim=-1)
        min_cov_ratio = self._masked_min(covered, self._poi_mask, dim=1)
        rho_bar, _ = self._directional_target(env_idx)

        self._prev_mean_poi_dist = mean_dist
        self._update_metrics(
            env_idx,
            mean_dist=mean_dist,
            reward=reward,
            reward_abs=reward_abs,
            reward_t_theta=reward_t_theta,
            t_theta=t_theta,
            T_theta=t_theta,
            t_theta_error=t_theta_error,
            min_cov_ratio=min_cov_ratio,
            rho_bar=rho_bar,
        )
        return reward.float()

    def _update_metrics(self, env_idx, **kwargs):
        n = self.num_envs
        for key, value in kwargs.items():
            if key == "rho_bar":
                radius = torch.exp(self._u[env_idx])
                radial_rmse = torch.sqrt(((radius - value) ** 2).mean(dim=1))
                self._set_metric(env_idx, "radial_rmse", radial_rmse)
                self._set_metric(env_idx, "chamfer_dist", 2.0 * radial_rmse)
                continue
            metric_key = "mean_poi_dist" if key == "mean_dist" else key
            self._set_metric(env_idx, metric_key, value)
        defaults = [
            "reward_total",
            "reward_poi_dist",
            "reward_poi_abs",
            "mean_poi_dist",
            "min_poi_coverage",
            "min_poi_coverage_ratio",
            "penalty_nofly",
            "nofly_overlap_ratio",
            "num_observed_pois",
            "num_discovered_pois",
            "observation_coverage",
            "newly_observed_pois",
            "mean_obs_noise",
            "radial_rmse",
            "chamfer_dist",
        ]
        for key in defaults:
            if key not in self._latest_metrics:
                self._latest_metrics[key] = torch.zeros((n,), device=self.device)
        if "reward" in kwargs:
            self._latest_metrics["reward_total"][env_idx] = kwargs["reward"].detach()
            self._latest_metrics["reward_poi_dist"][env_idx] = kwargs.get("reward_abs", torch.zeros_like(kwargs["reward"])).detach()
            self._latest_metrics["reward_poi_abs"][env_idx] = kwargs.get("reward_abs", torch.zeros_like(kwargs["reward"])).detach()
            self._latest_metrics["min_poi_coverage"][env_idx] = kwargs.get("min_cov_ratio", torch.zeros_like(kwargs["reward"])).detach()

    def _set_metric(self, env_idx, key, value):
        if key not in self._latest_metrics:
            self._latest_metrics[key] = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.device)
        self._latest_metrics[key][env_idx] = value.detach().float()

    @staticmethod
    def _masked_mean(values, mask, dim):
        values = values * mask.float()
        return values.sum(dim=dim) / mask.float().sum(dim=dim).clamp_min(1.0)

    @staticmethod
    def _masked_min(values, mask, dim):
        inf = torch.full_like(values, float("inf"))
        out = torch.where(mask, values, inf).min(dim=dim).values
        return torch.where(torch.isfinite(out), out, torch.zeros_like(out))

    @staticmethod
    def _masked_mean_points(points, mask):
        weights = mask.float().unsqueeze(-1)
        return (points * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self._latest_reward

    def compute_normalized_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return self.compute_dense_reward(obs, action, info)

    def render(self):
        self._ensure_storage()
        env_idx = torch.arange(self.num_envs, device=self.device)
        traj = self._trajectory_points(env_idx)
        frames = torch.zeros((self.num_envs, self._height, self._width, 3), dtype=torch.float32, device=self.device)
        poi = self._poi_points_obs
        poi_mask = self._poi_mask_obs[..., 0] > 0.5
        self._scatter_points(frames[..., 0], poi, poi_mask)
        self._scatter_points(frames[..., 1], traj, torch.ones(traj.shape[:2], dtype=torch.bool, device=self.device))
        true_mask = self._poi_mask
        self._scatter_points(frames[..., 2], self._poi_pos, true_mask)
        return (frames.clamp(0.0, 1.0) * 255.0).to(torch.uint8)

    def _scatter_points(self, canvas, points, mask):
        x = ((points[..., 0] + self._domain_radius) / (2.0 * self._domain_radius) * (self._width - 1)).round().long().clamp(0, self._width - 1)
        y = ((points[..., 1] + self._domain_radius) / (2.0 * self._domain_radius) * (self._height - 1)).round().long().clamp(0, self._height - 1)
        env = torch.arange(points.shape[0], device=self.device).view(-1, 1).expand_as(x)
        canvas[env[mask], y[mask], x[mask]] = 1.0

    def _observed_poi_traj_stats(self, env_idx):
        b = len(env_idx)
        visible = self._poi_mask_obs[env_idx, :, 0] > 0.5
        if b == 0:
            return (
                torch.zeros((0,), device=self.device),
                torch.zeros((0,), device=self.device),
            )

        traj = self._trajectory_points(env_idx)
        pts = self._poi_points_obs[env_idx]
        dist = torch.linalg.norm(pts.unsqueeze(2) - traj.unsqueeze(1), dim=-1)
        min_dist = dist.min(dim=-1).values
        mean_dist = self._masked_mean(min_dist, visible, dim=1)

        cov_radius = max(float(self._min_poi_coverage_radius), 1e-6)
        cov_ratio = torch.exp(-0.5 * (min_dist / cov_radius).square())
        min_cov = self._masked_min(cov_ratio, visible, dim=1)
        has_visible = visible.any(dim=1)
        mean_dist = torch.where(has_visible, mean_dist, torch.zeros_like(mean_dist))
        min_cov = torch.where(has_visible, min_cov, torch.zeros_like(min_cov))
        return mean_dist, min_cov

    def _compute_t_theta_from_dist(self, dist, mask):
        sigma = max(float(self._kernel_sigma_phys), 1e-6)
        heat = torch.exp(-0.5 * (dist / sigma).square())
        cutoff = max(float(self._kernel_cutoff), 0.0) * sigma
        if cutoff > 0.0:
            heat = torch.where(dist <= cutoff, heat, torch.zeros_like(heat))
        poi_heat = heat.mean(dim=-1)
        scale = float(self._resource_eta) * float(self._num_agents) / max(float(self._heat_lambda), 1e-6)
        poi_heat_sat = (scale * poi_heat / max(float(self._phi_sat), 1e-6)).clamp(0.0, 1.0)
        return self._masked_mean(poi_heat_sat, mask, dim=1).clamp(0.0, 1.0)


gym.register(
    id="TrajectoryShapePOMDPV2CoreReward-v1",
    entry_point="mani_skill.envs.tasks.trajectory_shape:TrajectoryShapePOMDPV2CoreRewardEnv",
    disable_env_checker=True,
    order_enforce=False,
)
