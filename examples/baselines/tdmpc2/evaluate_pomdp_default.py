"""Evaluate only the default TD-MPC2 method with same-rollout UAV execution metrics."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from common.parser import parse_cfg
from common.seed import set_seed
from envs import make_envs
from tdmpc2 import TDMPC2
from evaluation import tracking
from evaluation.metrics import (
    _advance_uav_heat, _curve_np, _grid_cache, _saturate_heat,
    _uav_execution_metrics, _uav_grid_t_theta_and_point_metrics,
)
from evaluation.rollout import (
    _initial_uav_state, _joint_apply_action, _joint_refresh_observation, _step_uav_tracking,
)


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Saved training config; auto-detected next to models/.")
    parser.add_argument("--seeds", default="20260518,20260519,20260520,20260521,20260522")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/default_eval"))
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        parser.error("At least one seed is required")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu for a verification rollout")
    config_path = args.config or checkpoint.parent.parent / "config.yaml"
    if args.config is not None and not config_path.is_file():
        parser.error(f"Training configuration not found: {config_path}")
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parent), version_base=None):
        cfg = compose(config_name="config_traj_shape_pomdp_default")
    if config_path.is_file():
        saved = OmegaConf.to_container(OmegaConf.load(config_path), resolve=False)
        # Preserve task overrides while ignoring retired robotics/video settings.
        saved = {key: value for key, value in saved.items()
                 if key in cfg or str(key).startswith("traj_")}
        with open_dict(cfg):
            cfg = OmegaConf.merge(cfg, saved)
    cfg.env_type = "gpu" if args.device == "cuda" else "cpu"
    cfg.num_envs = cfg.num_eval_envs = 1
    cfg.time_limit = args.steps
    cfg.wandb = False
    # The only task supported by this entry point has a strict policy boundary.
    if cfg.env_id != "TrajectoryShapePOMDPV2CoreReward-v1":
        parser.error("Checkpoint configuration must use the final POMDP-v2 task")
    cfg = parse_cfg(cfg)
    set_seed(seeds[0])
    env = make_envs(cfg, 1, is_eval=True)
    agent = TDMPC2(cfg)
    agent.load(checkpoint)
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    summaries = []
    for seed in seeds:
        set_seed(seed)
        env.reset(seed=seed)
        task = env._task
        state = _initial_uav_state(tracking, task, np.random.RandomState(seed + 1000003))
        obs = _joint_refresh_observation(task, state, reset=True)
        set_seed(seed)
        grid = _grid_cache(task)
        heat = np.zeros((grid["height"], grid["width"]), dtype=np.float32)
        sweep = {}
        rows = []
        for step in range(args.steps):
            action = agent.act(obs, t0=step == 0, eval_mode=True)
            reward = _joint_apply_action(task, action)
            curve = _curve_np(task)
            state, controls, track_error = _step_uav_tracking(tracking, task, state, curve)
            heat = _advance_uav_heat(task, heat, state.positions, 0.1)
            uav = _uav_grid_t_theta_and_point_metrics(task, state.positions, _saturate_heat(task, heat))
            uav.update(_uav_execution_metrics(task, state.positions, controls, sweep))
            # These GT metrics are reporting-only and never reach act()/plan().
            info = task.get_info()
            t_theta = float(info["T_theta"][0])
            row = dict(seed=seed, step=step + 1, reward=float(reward[0]),
                       T_theta=t_theta, mean_poi_dist=float(info["mean_poi_dist"][0]),
                       min_poi_coverage_ratio=float(info["min_poi_coverage_ratio"][0]),
                       uav_track_error_mean=track_error,
                       phi=float(action[0, 0]), alpha=float(action[0, 1]), width=float(action[0, 2]))
            row.update(uav)
            row["t_uav_theta_gap"] = t_theta - uav["uav_T_theta"]
            rows.append(row)
            obs = _joint_refresh_observation(task, state, reset=False)
        _write_csv(out / f"steps_seed_{seed}.csv", rows)
        summary = dict(seed=seed, episode_steps=len(rows),
                       episode_return=sum(row["reward"] for row in rows))
        for name in ["T_theta", "uav_T_theta", "mean_poi_dist", "uav_mean_poi_dist",
                     "min_poi_coverage_ratio", "uav_min_poi_coverage_ratio", "uav_track_error_mean"]:
            summary[name + "_auc"] = float(np.mean([row[name] for row in rows]))
            summary[name + "_final"] = rows[-1][name]
        summary["t_uav_theta_auc_gap"] = summary["T_theta_auc"] - summary["uav_T_theta_auc"]
        for name in ["uav_unique_poi_swept", "uav_unique_poi_swept_ratio", "uav_control_energy_cumulative"]:
            summary[name + "_final"] = rows[-1][name]
        integral = sum(row["uav_T_theta"] for row in rows)
        summary["uav_T_theta_integral"] = integral
        summary["uav_energy_per_t_theta_integral"] = rows[-1]["uav_control_energy_cumulative"] / max(integral, 1e-6)
        summary["uav_coverage_integral_zero"] = int(integral <= 1e-6)
        summaries.append(summary)
        print(json.dumps(summary))
    _write_csv(out / "episodes.csv", summaries)
    aggregate = {key: {"mean": float(np.mean([row[key] for row in summaries])),
                       "std": float(np.std([row[key] for row in summaries]))}
                 for key in summaries[0] if key not in {"seed", "episode_steps"}}
    manifest = dict(protocol="ManiSkill native joint UAV evaluation (standalone default extraction)",
                    method="default TD-MPC2", environment=cfg.env_id, seeds=seeds, steps=args.steps,
                    checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    checkpoint_selection="explicit_checkpoint", model_size=int(cfg.model_size),
                    planner_budget=dict(horizon=int(cfg.horizon), iterations=int(cfg.iterations),
                                        num_samples=int(cfg.num_samples), num_elites=int(cfg.num_elites),
                                        num_pi_trajs=int(cfg.num_pi_trajs)),
                    num_uavs=int(env._task._uav_num),
                    device=args.device, uav_tracking_mode="joint", uav_dt=0.1,
                    outer_metric="Core reward PoI point-sampling T_theta",
                    inner_metric="Grid overlap UAV_T_theta with cumulative decayed UAV heat",
                    auc_definition="Arithmetic mean over rollout steps",
                    integral_definition="Sum over rollout steps; energy/T denominator clamped at 1e-6, matching source native eval",
                    energy="Normalized unicycle surrogate energy; not battery energy",
                    planner_uses_full_poi=False, planner_uses_eval_metrics=False,
                    observation_source="joint UAV positions; refresh every outer step",
                    aggregate=aggregate)
    (out / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    env.close()
    print(f"Saved evaluation: {out}")


if __name__ == "__main__":
    main()
