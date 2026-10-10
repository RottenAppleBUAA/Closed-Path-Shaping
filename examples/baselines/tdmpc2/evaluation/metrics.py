"""Reporting-only trajectory/UAV metrics; never used by the default planner."""
from __future__ import annotations
from typing import Any, Dict
import numpy as np
import torch
UAV_TRACKING_V_MAX = 1.8
UAV_TRACKING_OMEGA_MAX = 1.5
UAV_ENERGY_LAMBDA_V = 1.0
UAV_ENERGY_LAMBDA_A = 1.0

def _curve_np(task: Any) -> np.ndarray:
    """Export the first environment path as world-coordinate points [K, 2]."""
    env_idx = torch.arange(1, device=task.device)
    return task._trajectory_points(env_idx)[0].detach().cpu().numpy().astype(np.float32)

def _poi_np(task: Any) -> np.ndarray:
    """Export valid scene PoIs [P, 2] for evaluation reporting."""
    poi_num = int(task._poi_num[0].detach().cpu().item()) if torch.is_tensor(task._poi_num) else int(task._poi_num)
    return task._poi_pos[0, :poi_num].detach().cpu().numpy().astype(np.float32)

def _grid_cache(task: Any) -> Dict[str, Any]:
    """
    Cache grid coordinates and the truncated Gaussian footprint kernel.
    The cache key includes geometry, kernel parameters, and tensor device.
    """
    height = int(getattr(task, "_height", 64))
    width = int(getattr(task, "_width", 64))
    domain_radius = float(getattr(task, "_domain_radius", 10.0))
    sigma_phys = float(getattr(task, "_kernel_sigma_phys", 1.5))
    cutoff_phys = float(getattr(task, "_kernel_cutoff_phys", getattr(task, "_kernel_cutoff", 2.0)))
    device = getattr(task, "device", torch.device("cpu"))
    key = (height, width, domain_radius, sigma_phys, cutoff_phys, str(device))
    cache = getattr(task, "_eval_grid_cache", None)
    if isinstance(cache, dict) and cache.get("key") == key:
        return cache

    # Use the same world bounds for interest maps and executed footprints.
    xs = np.linspace(-domain_radius, domain_radius, width, dtype=np.float32)
    ys = np.linspace(-domain_radius, domain_radius, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    dx_ref = max(float((2.0 * domain_radius) / max(width - 1, 1)), 1e-6)
    sigma_pix = max(sigma_phys / dx_ref, 1e-6)
    cutoff_pix = max(cutoff_phys / dx_ref, 1.0)
    radius_pix = int(max(cutoff_pix, 1.0))
    k_range = torch.arange(-radius_pix, radius_pix + 1, device=device, dtype=torch.float32)
    try:
        ky, kx = torch.meshgrid(k_range, k_range, indexing="ij")
    except TypeError:
        ky, kx = torch.meshgrid(k_range, k_range)
    dist_sq = kx.square() + ky.square()
    kernel = torch.exp(-dist_sq / (2.0 * float(sigma_pix**2)))
    kernel = torch.where(dist_sq <= float(cutoff_pix**2), kernel, torch.zeros_like(kernel))
    cache = {
        "key": key,
        "height": height,
        "width": width,
        "domain_radius": domain_radius,
        "grid_x": grid_x,
        "grid_y": grid_y,
        "kernel": kernel.view(1, 1, kernel.shape[0], kernel.shape[1]),
        "kernel_radius_pix": radius_pix,
    }
    setattr(task, "_eval_grid_cache", cache)
    return cache

def _points_to_grid_indices(task: Any, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map world-coordinate points to clipped integer grid indexes."""
    cache = _grid_cache(task)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    radius = float(cache["domain_radius"])
    x_norm = (pts[:, 0] + radius) / (2.0 * radius)
    y_norm = (pts[:, 1] + radius) / (2.0 * radius)
    ix = np.clip(np.rint(x_norm * float(int(cache["width"]) - 1)), 0, int(cache["width"]) - 1).astype(np.int64)
    iy = np.clip(np.rint(y_norm * float(int(cache["height"]) - 1)), 0, int(cache["height"]) - 1).astype(np.int64)
    return ix, iy

def _points_to_interest_map(task: Any, points: np.ndarray) -> np.ndarray:
    """
    Build a truncated Gaussian interest map from evaluation PoIs.
    This scene-level map is used for scoring and never for planner input.
    """
    cache = _grid_cache(task)
    height = int(cache["height"])
    width = int(cache["width"])
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.zeros((height, width), dtype=np.float32)
    sigma = float(getattr(task, "_interest_sigma", getattr(task, "_r_soft", getattr(task, "_kernel_sigma_phys", 1.5))))
    sigma_sq = float(max(sigma, 1e-6) ** 2)
    grid_x = np.asarray(cache["grid_x"], dtype=np.float32)
    grid_y = np.asarray(cache["grid_y"], dtype=np.float32)
    dx = grid_x[None, :, :] - pts[:, 0:1, None]
    dy = grid_y[None, :, :] - pts[:, 1:2, None]
    dist_sq = dx * dx + dy * dy
    contrib = np.exp(-dist_sq / (2.0 * sigma_sq)).astype(np.float32)
    cutoff_sq = float(max(getattr(task, "_r_soft", sigma), 1e-6) ** 2)
    contrib[dist_sq > cutoff_sq] = 0.0
    return np.clip(np.mean(contrib, axis=0), 0.0, 1.0).astype(np.float32)

def _compute_points_coverage(task: Any, points: np.ndarray) -> np.ndarray:
    """Rasterize sensing points and convolve their normalized occupancy footprint."""
    cache = _grid_cache(task)
    height = int(cache["height"])
    width = int(cache["width"])
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] == 0:
        return np.zeros((height, width), dtype=np.float32)
    ix_np, iy_np = _points_to_grid_indices(task, pts)
    if ix_np.size == 0:
        return np.zeros((height, width), dtype=np.float32)
    device = getattr(task, "device", torch.device("cpu"))
    occ = torch.zeros((1, 1, height, width), device=device, dtype=torch.float32)
    ix = torch.as_tensor(ix_np, device=device, dtype=torch.long)
    iy = torch.as_tensor(iy_np, device=device, dtype=torch.long)
    flat = occ.view(-1)
    indices = (iy * int(width) + ix).long()
    # Accumulate coincident sensing points before spatial convolution.
    flat.index_add_(0, indices, torch.ones_like(indices, dtype=flat.dtype))
    occ = flat.view_as(occ)
    conv = torch.nn.functional.conv2d(occ, cache["kernel"], padding=int(cache["kernel_radius_pix"]))
    conv = conv / float(max(int(pts.shape[0]), 1))
    return conv[0, 0].detach().cpu().numpy().astype(np.float32)

def _saturate_heat(task: Any, heat: np.ndarray) -> np.ndarray:
    """Normalize and clip heat to [0, 1] when a positive saturation level is set."""
    phi_sat = float(getattr(task, "_phi_sat", 1.0))
    # Saturation is applied when scoring, not when storing the accumulated heat.
    heat_arr = np.asarray(heat, dtype=np.float32)
    if phi_sat <= 0.0:
        return heat_arr.astype(np.float32)
    return np.clip(heat_arr / phi_sat, 0.0, 1.0).astype(np.float32)

def _advance_uav_heat(task: Any, heat: np.ndarray, positions: np.ndarray, dt: float) -> np.ndarray:
    """Decay the UAV heat field and accumulate the current sensing footprint."""
    lam = float(max(getattr(task, "_heat_lambda", 1.0), 0.0))
    # Keep continuous-time decay consistent with the execution interval.
    decay = float(np.exp(-lam * float(dt)))
    footprint = _compute_points_coverage(task, positions)
    num_agents = float(max(getattr(task, "_num_agents", footprint.shape[0] if footprint.ndim > 0 else 1.0), 1e-6))
    resource = float(getattr(task, "_resource_eta", 1.0))
    return (decay * np.asarray(heat, dtype=np.float32) + resource * num_agents * footprint * float(dt)).astype(np.float32)

def _weighted_overlap(coverage_sat: np.ndarray, weight_map: np.ndarray, eps: float) -> float:
    """Compute normalized overlap between saturated heat and the interest map."""
    h_val = np.asarray(coverage_sat, dtype=np.float64)
    w_val = np.asarray(weight_map, dtype=np.float64)
    if h_val.size == 0 or w_val.size == 0 or h_val.shape != w_val.shape:
        return 0.0
    den = float(np.sum(w_val, dtype=np.float64))
    if den <= 0.0:
        return 0.0
    return float(np.sum(h_val * w_val, dtype=np.float64) / (den + float(max(eps, 1e-12))))

def _uav_grid_t_theta_and_point_metrics(task: Any, positions: np.ndarray, uav_coverage_sat: np.ndarray) -> Dict[str, float]:
    """
    Report executed grid coverage and instantaneous UAV-to-PoI geometry.
    Ground-truth scene PoIs are used only for these evaluation metrics.
    """
    pos = torch.as_tensor(np.asarray(positions, dtype=np.float32).reshape(1, -1, 2), device=task.device)
    env_idx = torch.arange(1, device=task.device)
    poi = task._poi_pos[env_idx]
    mask = task._poi_mask[env_idx]
    dist = torch.linalg.norm(poi.unsqueeze(2) - pos.unsqueeze(1), dim=-1)
    weight_map = _points_to_interest_map(task, _poi_np(task))
    uav_t = _weighted_overlap(uav_coverage_sat, weight_map, float(getattr(task, "_psi_eps", 1e-6)))
    min_dist = dist.min(dim=-1).values
    mean_dist = task._masked_mean(min_dist, mask, dim=1)
    covered = (dist < float(getattr(task, "_min_poi_coverage_radius", 1.8))).float().mean(dim=-1)
    min_cov = task._masked_min(covered, mask, dim=1)
    return {
        "uav_T_theta": float(uav_t),
        "uav_mean_poi_dist": float(mean_dist[0].detach().cpu().item()),
        "uav_min_poi_coverage_ratio": float(min_cov[0].detach().cpu().item()),
        "uav_min_poi_coverage": float((covered * mask.float()).sum(dim=1)[0].detach().cpu().item()),
    }

def _uav_execution_metrics(task: Any, positions: np.ndarray, action: np.ndarray | None, sweep_state: Dict[str, Any]) -> Dict[str, float]:
    """
    Update PoI sweep counts, path length, and normalized control effort.
    The supplied sweep_state retains cumulative episode statistics; effort
    is a unicycle surrogate rather than a battery-energy estimate.
    """
    dt = float(getattr(task, "_uav_dt", 0.1))
    dt = max(dt, 1e-9)
    sweep_state["step_count"] = int(sweep_state.get("step_count", 0)) + 1
    elapsed = float(sweep_state["step_count"]) * dt
    poi = _poi_np(task)
    pos = np.asarray(positions, dtype=np.float32).reshape(-1, 2)
    if poi.size == 0 or pos.size == 0:
        covered_ids: set[int] = set()
    else:
        dist = np.linalg.norm(poi[:, None, :] - pos[None, :, :], axis=-1)
        radius = float(getattr(task, "_min_poi_coverage_radius", getattr(task, "_r_soft", 1.8)))
        covered_ids = set(int(i) for i in np.where(np.min(dist, axis=1) < radius)[0])
    # Track distinct PoI IDs over the episode, separately from per-step sweeps.
    seen: set[int] = sweep_state.setdefault("covered_poi_ids", set())
    new_ids = covered_ids - seen
    seen.update(covered_ids)
    sweep_count = float(len(covered_ids))
    new_count = float(len(new_ids))
    unique_count = float(len(seen))
    sweep_state["sweep_count_total"] = float(sweep_state.get("sweep_count_total", 0.0)) + sweep_count

    if action is None:
        action_arr = np.zeros((pos.shape[0], 2), dtype=np.float32)
    else:
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1, 2)
    if action_arr.size:
        speed = np.abs(action_arr[:, 0].astype(np.float64))
        omega = np.abs(action_arr[:, 1].astype(np.float64))
        v_max = max(float(getattr(task, "_uav_speed", UAV_TRACKING_V_MAX)), 1e-9)
        omega_max = max(float(getattr(task, "_uav_omega_max", UAV_TRACKING_OMEGA_MAX)), 1e-9)
        a_max = max(v_max * omega_max, 1e-9)
        # Normalize speed and lateral-acceleration proxies by fixed control scales.
        v_term = (speed / v_max) ** 2
        a_term = ((speed * omega) / a_max) ** 2
        energy_step = float(dt * np.sum(UAV_ENERGY_LAMBDA_V * v_term + UAV_ENERGY_LAMBDA_A * a_term, dtype=np.float64))
        path_step = float(dt * np.sum(speed, dtype=np.float64))
        mean_speed = float(np.mean(speed))
        mean_abs_omega = float(np.mean(omega))
    else:
        energy_step = 0.0
        path_step = 0.0
        mean_speed = float("nan")
        mean_abs_omega = float("nan")
    sweep_state["control_energy_total"] = float(sweep_state.get("control_energy_total", 0.0)) + energy_step
    sweep_state["path_length_total"] = float(sweep_state.get("path_length_total", 0.0)) + path_step
    energy_total = float(sweep_state["control_energy_total"])
    sweep_total = float(sweep_state.get("sweep_count_total", 0.0))
    poi_num = int(poi.shape[0])
    return {
        "uav_poi_sweep_count": sweep_count,
        "uav_new_poi_swept_count": new_count,
        "uav_unique_poi_swept": unique_count,
        "uav_unique_poi_swept_ratio": unique_count / float(poi_num) if poi_num > 0 else float("nan"),
        "uav_poi_sweep_rate": sweep_count / dt,
        "uav_new_poi_sweep_rate": new_count / dt,
        "uav_unique_poi_sweep_rate": unique_count / elapsed if elapsed > 0.0 else float("nan"),
        "uav_control_energy_step": energy_step,
        "uav_control_energy_cumulative": energy_total,
        "uav_path_length_step": path_step,
        "uav_path_length_cumulative": float(sweep_state["path_length_total"]),
        "uav_mean_speed": mean_speed,
        "uav_mean_abs_omega": mean_abs_omega,
        "uav_energy_per_sweep": energy_total / sweep_total if sweep_total > 0.0 else float("nan"),
        "uav_energy_per_unique_poi": energy_total / unique_count if unique_count > 0.0 else float("nan"),
    }
