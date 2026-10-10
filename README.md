# Persistent Multi-UAV Coverage through Closed-Path Shaping

English | [简体中文](README_zh.md)

This project studies **persistent coverage of moving points of interest (PoIs) by multiple UAVs under partial observation**. A compact three-dimensional macro action locally deforms a shared closed patrol path. Standard TD-MPC2 performs online planning over path deformations, and a deterministic follower converts the shaped path into UAV motion.

The method centers on a closed-path representation and an interpretable shaping interface whose action dimension is independent of team size. This repository provides the POMDP environment, default TD-MPC2 training, and evaluation of planned and executed coverage in the same rollout.

**Paper:** [Persistent Multi-UAV Coverage through Closed-Path Shaping with Model-Based Reinforcement Learning](<Persistent Multi-UAV Coverage through Closed-Path Shaping with Model-Based Reinforcement Learning.pdf>).

## Method

### Shared closed patrol path

The patrol path is represented by a positive, periodic radius function:

$$
\Gamma_t(\theta)=c_t+r_t(\theta)[\cos\theta,\sin\theta]^\top,
\qquad \ell_t(\theta)=\log r_t(\theta).
$$

The periodic angular coordinate preserves closure, while radius bounds constrain the path geometry. The default representation uses a fixed center, 128 angular samples, and low-frequency FFT coefficients for the policy input.

### Three-dimensional shaping action

The normalized action `(phi, alpha, lambda)` controls the location, signed magnitude, and width of a local deformation:

| Component | Meaning | Default mapping |
| --- | --- | --- |
| `phi` | Angular location | `[-1, 1] → [-π, π]` |
| `alpha` | Radial amplitude: expansion or contraction | `[-1, 1] → [-0.3, 0.3]` |
| `lambda` | Angular width | `[-1, 1] → [0.2, 1.0]` |

The code's width parameter `lambda` corresponds to `sigma` in the paper's shaping equations. A Gaussian weight `w(theta)` is computed from the shortest angular distance on the circle, and the radius is updated as

$$
r_{t+1}(\theta)=\operatorname{clip}_{[r_{\min},r_{\max}]}
\left(r_t(\theta)[1+\alpha_t w_t(\theta)]\right).
$$

The implementation performs this update in log-radius space. The default radius bounds are `[1.0, 7.5]`, and the action dimension remains three for any UAV team size.

### Model-based planning

TD-MPC2 encodes a four-frame partial-observation history into a latent state. Its world model learns latent dynamics, rewards, a Q ensemble, and a policy prior. Policy-prior-assisted MPPI evaluates candidate shaping-action sequences, updates its sampling distribution, and executes the first macro action.

The planner follows the default TD-MPC2 formulation; the task-specific planning interface is the three-dimensional closed-path shaping action.

### Deterministic UAV execution

Joint UAV evaluation uses a line-of-sight (LOS) and phase-spacing controller. Each UAV is projected onto the path, a look-ahead target regulates heading, and cyclic phase gaps regulate speed. Bounded unicycle motion produces the next UAV positions and local PoI observations.

```mermaid
flowchart LR
    A[Four-frame partial observations] --> B[TD-MPC2 world model and MPPI]
    B --> C[Three-dimensional closed-path shaping]
    C --> D[LOS and phase-spacing control]
    D --> E[UAV motion and local sensing]
    E --> A
```

## Environment and metrics

The environment ID is `TrajectoryShapePOMDPV2CoreReward-v1`. A default observation frame contains 25 curve FFT coefficients, 2 path-center coordinates, 16 visible-PoI coordinates, and 3 visible-PoI statistics. Stacking four frames yields a **184-dimensional input**. Hidden PoIs do not enter the policy input.

The implemented training reward is

$$
r_t=-0.45\,\overline d(\Gamma_t,P_t)-0.30(1-C_t^{\mathrm{plan}}),
$$

where `C_plan` is a PoI point-sampling approximation of planned coverage and the distance term measures proximity between the patrol path and moving PoIs.

| Protocol | Observation and execution | Purpose |
| --- | --- | --- |
| Training and training-time evaluation | Tensor environment; trajectory-neighborhood visibility with sampling every 3 steps | Learning and training-return monitoring |
| Joint UAV evaluation | 256-point closed polyline; fixed LOS follower; observations refreshed from actual UAV positions every step | Planned and executed coverage in the same rollout |

The joint follower uses `dt=0.1`, `v0=0.9`, `v_max=1.8`, `omega_max=1.5`, `k_psi=2.5`, `k_c=0.15`, and look-ahead distance `0.6`. The control-effort surrogate is normalized by `v_n=1.8` and `omega_n=2.4`.

Reported metrics include planned coverage `T_theta`, executed coverage `uav_T_theta`, the planning–execution gap, UAV–PoI distance, swept PoIs, tracking error, and effort per coverage. AUC denotes the step mean; `integral` denotes the step sum. Ground-truth PoIs and global coverage metrics are used for reward/reporting, and are not passed to the planner. The effort surrogate does not model battery energy.

## Installation

Use **Python 3.9** on Linux or WSL2. GPU training requires a CUDA-capable PyTorch installation. CPU mode is available for evaluation.

Clone this repository and create an isolated environment:

```bash
git clone https://github.com/RottenAppleBUAA/Closed-Path-Shaping.git
cd Closed-Path-Shaping
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) specifies PyTorch `2.5.1`, TorchRL `0.6.0`, TensorDict `0.6.2`, numerical utilities, configuration tools, and logging dependencies. It also installs the task package included in this repository with `-e .`. Choose a CPU or CUDA PyTorch wheel using the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v251) before installing the remaining requirements if needed.

The environment runs directly on PyTorch tensors and uses Gymnasium spaces. No robot assets, SAPIEN installation, or rigid-body simulator are needed. Training and evaluation use the code shipped in this repository; no sibling repositories, external checkpoints, or machine-specific environments are required.

## Training

```bash
cd examples/baselines/tdmpc2
python train.py --config-name config_traj_shape_pomdp_default \
  seed=123 exp_name=default-pomdp-1m
```

Defaults: 256 GPU tensor environments, 1M environment steps, `model_size=5`, MPPI horizon 3, 512 samples, 64 elites, 24 policy-prior trajectories, and 6 planning iterations. The update settings are `steps_per_update=16` and `pretrain_updates=1280`. W&B logging is disabled by default.

Training creates the following files under the baseline working directory:

```text
logs/TrajectoryShapePOMDPV2CoreReward-v1/123/default-pomdp-1m/
  config.yaml
  models/final.pt
  eval.csv          # Created when training-time evaluation runs
```

The saved configuration records the model architecture, observation settings, and planning budget. The checkpoint used below is generated by this training command; pretrained weights are not bundled.

## Joint UAV evaluation

From `examples/baselines/tdmpc2`, evaluate the trained checkpoint:

```bash
python evaluate_pomdp_default.py \
  --checkpoint logs/TrajectoryShapePOMDPV2CoreReward-v1/123/default-pomdp-1m/models/final.pt \
  --seeds 20260518,20260519,20260520,20260521,20260522 \
  --steps 512 --device cuda \
  --output-dir outputs/default-pomdp-1m-eval
```

The output directory must not already exist. The evaluator reads the training configuration next to the checkpoint, or accepts an explicit `--config` path. Use `--device cpu` for CPU evaluation.

Outputs are `episodes.csv`, `steps_seed_<seed>.csv`, `summary.json`, and the effective configuration. They record the checkpoint SHA256, scene seeds, planning budget, per-step trajectory/UAV metrics, and aggregate statistics.

## Code structure

| Module | Entry point |
| --- | --- |
| POMDP environment and closed-path action | [trajectory_shape.py](mani_skill/envs/tasks/trajectory_shape/trajectory_shape.py) |
| Tensor vector environment | [trajectory_shape_vector.py](examples/baselines/tdmpc2/envs/trajectory_shape_vector.py) |
| World model and default planner | [world_model.py](examples/baselines/tdmpc2/common/world_model.py), [tdmpc2.py](examples/baselines/tdmpc2/tdmpc2.py) |
| Training and configuration | [train.py](examples/baselines/tdmpc2/train.py), [config_traj_shape_pomdp_default.yaml](examples/baselines/tdmpc2/config_traj_shape_pomdp_default.yaml) |
| UAV follower and reporting metrics | [evaluation/](examples/baselines/tdmpc2/evaluation/) |
| Joint evaluation | [evaluate_pomdp_default.py](examples/baselines/tdmpc2/evaluate_pomdp_default.py) |

## Background and license

The task interface and vectorized TD-MPC2 implementation are derived from ManiSkill 3.0.1. This repository retains the tensor task and planner required for closed-path shaping. The LOS follower is included as a self-contained execution module.

See [LICENSE](LICENSE) and [LICENSE-3RD-PARTY](LICENSE-3RD-PARTY) and [LICENSE-TD-MPC2](LICENSE-TD-MPC2) for the upstream licensing notices. The extracted reference follower's notice is preserved in [evaluation/LICENSE_REFERENCE](examples/baselines/tdmpc2/evaluation/LICENSE_REFERENCE).
