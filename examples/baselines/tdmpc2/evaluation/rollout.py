"""Default-policy joint rollout with a fixed LOS executor and UAV observations."""
from __future__ import annotations
from types import SimpleNamespace
from typing import Any, Sequence, Dict
import numpy as np
import torch

def _joint_refresh_observation(task: Any, tracking_state: Any, *, reset: bool) -> Any:
    """
    Refresh visibility from the executed UAV positions and update frame history.
    Reset duplicates the current frame; subsequent calls append one new frame.
    """
    env_idx = torch.arange(int(task.num_envs), device=task.device)
    # Force sensing from executed positions after each joint evaluation step.
    task._sample_poi_observation(
        env_idx,
        force=True,
        observer_points=np.asarray(tracking_state.positions, dtype=np.float32),
        reset_observation_bookkeeping=bool(reset),
    )
    single = task._single_state(env_idx)
    # Reset has no earlier frames, so repeat the first partial observation.
    if reset:
        task._state_history[env_idx] = single.unsqueeze(1).expand(-1, int(task._stack_len), -1)
    else:
        task._state_history = torch.roll(task._state_history, shifts=-1, dims=1)
        task._state_history[:, -1] = single
    task._refresh_stacked_state(env_idx)
    obs = task.get_obs()
    task._last_obs = obs
    return obs

def _joint_apply_action(task: Any, action: Any) -> torch.Tensor:
    """
    Apply the shaping action and PoI motion before the external UAV step.
    Return the core reward without advancing native observers or frame history.
    """
    action_t = torch.as_tensor(action, dtype=torch.float32, device=task.device)
    if action_t.shape == (3,):
        action_t = action_t.unsqueeze(0)
    if action_t.shape[0] != int(task.num_envs):
        action_t = action_t.expand(int(task.num_envs), -1)
    reward_total = torch.zeros((int(task.num_envs),), dtype=torch.float32, device=task.device)
    for _ in range(int(getattr(task, "_action_repeat", 1))):
        task._elapsed_steps += 1
        task._advance_dynamics(action_t)
        reward_total = reward_total + task._compute_reward()
    task._latest_reward = reward_total
    return reward_total

def _initial_uav_state(
    tracking: Any,
    task: Any,
    rng: np.random.RandomState,
    initial_position_items: Sequence[str] | None = None,
):
    """
    Initialize phase-separated UAV positions and tangential headings.
    Optional explicit positions must match the team size and world bounds.
    """
    num = max(int(getattr(task, "_uav_num", 4)), 1)
    domain_radius = float(getattr(task, "_domain_radius", 1.0))
    items = [str(item).strip() for item in (initial_position_items or []) if str(item).strip()]
    if items:
        try:
            positions = np.asarray([[float(value) for value in item.split(",")] for item in items], dtype=np.float64)
        except ValueError as exc:
            raise ValueError("each --initial-uav-position must use x,y format") from exc
        if positions.shape != (num, 2):
            raise ValueError(f"expected {num} --initial-uav-position entries, got shape={positions.shape}")
        if not np.isfinite(positions).all():
            raise ValueError("initial UAV positions must be finite")
        if np.abs(positions).max() > domain_radius:
            raise ValueError(f"initial UAV positions must remain inside domain radius {domain_radius}")
        center = task._center_offset[0].detach().cpu().numpy().astype(np.float64).reshape(2)
        angles = np.arctan2(positions[:, 1] - center[1], positions[:, 0] - center[0])
        headings = np.asarray(
            [tracking.wrap_angle(float(theta + 0.5 * np.pi)) for theta in angles],
            dtype=np.float64,
        )
        return tracking.UnicycleSwarmState(positions=positions, headings=headings)
    radius = float(max(0.2 * domain_radius, 0.1))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    angles = np.linspace(0.0, 2.0 * np.pi, num, endpoint=False, dtype=np.float64) + phase
    positions = np.stack([radius * np.cos(angles), radius * np.sin(angles)], axis=-1).astype(np.float64)
    noise = rng.randn(num, 2).astype(np.float64) * float(0.03 * max(domain_radius, 1e-6))
    positions = np.clip(positions + noise, -domain_radius, domain_radius)
    headings = np.asarray([tracking.wrap_angle(float(theta + 0.5 * np.pi)) for theta in angles], dtype=np.float64)
    return tracking.UnicycleSwarmState(positions=positions, headings=headings)

def _step_uav_tracking(tracking: Any, task: Any, state: Any, curve: np.ndarray):
    """
    Execute one fixed-profile LOS/phase tracking step on the shaped path.
    Returns the next unicycle state, controls [N, 2], and mean projection error.
    """
    profile = SimpleNamespace(dt=0.1)
    # Resample the radial path into the executor's fixed-resolution closed polyline.
    polyline, _ = tracking.curve_to_polyline(np.asarray(curve, dtype=np.float64), num_points=256, smoothing_window=5)
    ref = tracking.ReferenceTrajectory(polyline)
    v_min, v_max = 0.0, 1.8
    controller = tracking.LOSPhaseController(
        ref_traj=ref,
        num_agents=int(state.positions.shape[0]),
        dt=float(profile.dt),
        v0=0.5 * (v_min + v_max),
        v_min=float(v_min),
        v_max=float(v_max),
        k_psi=2.5,
        k_c=0.15,
        lookahead=float(max(0.4, 0.4 * getattr(task, "_r_soft", 1.0))),
        omega_max=1.5,
        sort_by_phase=True,
        use_spillover=False,
        use_avoid=False,
    )
    v_cmd, w_cmd, dbg = controller.step(state, return_debug=True)
    dt = float(profile.dt)
    # Integrate heading first, then translate using the updated heading.
    headings = np.asarray([tracking.wrap_angle(float(psi + dt * omega)) for psi, omega in zip(state.headings, w_cmd)], dtype=np.float64)
    direction = np.stack([np.cos(headings), np.sin(headings)], axis=-1).astype(np.float64)
    positions = state.positions + direction * np.asarray(v_cmd, dtype=np.float64).reshape(-1, 1) * dt
    domain_radius = float(getattr(task, "_domain_radius", 1.0))
    positions = np.clip(positions, -domain_radius, domain_radius).astype(np.float64)
    next_state = tracking.UnicycleSwarmState(positions=positions, headings=headings)
    action = np.stack([np.asarray(v_cmd, dtype=np.float32), np.asarray(w_cmd, dtype=np.float32)], axis=-1)
    e_perp = np.asarray((dbg or {}).get("e_perp", []), dtype=np.float64).reshape(-1)
    e_perp = e_perp[np.isfinite(e_perp)]
    track_error = float(np.mean(np.abs(e_perp))) if e_perp.size else float("nan")
    return next_state, action, track_error
