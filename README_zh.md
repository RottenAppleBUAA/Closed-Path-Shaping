# Persistent Multi-UAV Coverage through Closed-Path Shaping

[English](README.md) | 简体中文

本项目研究**部分可观测条件下，多 UAV 对动态兴趣点的持续覆盖**。我们用一个三维宏动作局部编辑共享闭合航线，由标准 TD-MPC2 进行在线规划，再通过确定性跟踪器把航线转化为 UAV 运动。方法贡献落在闭合航线表示与三维塑形接口，任务接口与向量化实现源自 ManiSkill。

参考论文：[Persistent Multi-UAV Coverage through Closed-Path Shaping with Model-Based Reinforcement Learning](<Persistent Multi-UAV Coverage through Closed-Path Shaping with Model-Based Reinforcement Learning.pdf>)。

**本仓库提供三维闭合航线塑形的 default TD-MPC2 实现、POMDP-v2 环境，以及训练和 joint UAV 评测入口。**

## 方法

### 1. 共享闭合航线

共享巡逻航线用极径函数表示：

$$
\Gamma_t(\theta)=c_t+r_t(\theta)[\cos\theta,\sin\theta]^\top,
\qquad \ell_t(\theta)=\log r_t(\theta).
$$

周期角坐标维持航线闭合，正半径和半径裁剪限制形状范围。当前默认使用固定中心、128 个角向半径采样；策略输入使用低频 FFT 编码。

### 2. 三维局部塑形动作

策略输出归一化动作 `(phi, alpha, lambda)`：

| 分量 | 含义 | 默认映射 |
| --- | --- | --- |
| `phi` | 形变发生的角向位置 | `[-1, 1] → [-π, π]` |
| `alpha` | 有符号径向强度，控制扩张或收缩 | `[-1, 1] → [-0.3, 0.3]` |
| `lambda` | 形变的角向宽度 | `[-1, 1] → [0.2, 1.0]` |

代码中的宽度 `lambda` 对应论文式 (12)–(14) 的 `sigma`，与热图衰减率的符号分开理解。用圆周最短角距离构造 Gaussian 权重 `w(theta)`，然后更新：

$$
r_{t+1}(\theta)=\operatorname{clip}_{[r_{\min},r_{\max}]}
\left(r_t(\theta)[1+\alpha_t w_t(\theta)]\right).
$$

代码在 log-radius 空间实现该更新。动作维度固定为 3，与 UAV 数量无关；默认半径范围为 `[1.0, 7.5]`。

### 3. 标准 TD-MPC2 在线规划

四帧部分观测堆叠输入状态编码器，世界模型学习 latent dynamics、reward、Q ensemble 和 policy prior。MPPI 在 latent 空间评估候选塑形动作序列，迭代更新采样分布，并执行首个宏动作。

高层规划器沿用标准 TD-MPC2 算法，任务专用的规划接口是三维闭合航线塑形动作。

### 4. 确定性 UAV 执行

joint UAV 评测采用 LOS/phase controller：投影 UAV 到航线，用前视目标控制航向，用循环相位间距调节速度，再推进有界 unicycle 运动。执行后的 UAV 位置决定下一次可见 PoI 观测。

```mermaid
flowchart LR
    A[四帧部分观测] --> B[TD-MPC2 世界模型与 MPPI]
    B --> C[三维闭合航线塑形]
    C --> D[LOS 与相位间距控制]
    D --> E[UAV 运动与局部感知]
    E --> A
```

图示展示高层塑形规划与确定性 UAV 执行组成的在线闭环。

## 环境与指标

环境 ID 为 `TrajectoryShapePOMDPV2CoreReward-v1`。默认单帧观测由 25 维曲线 FFT、2 维航线中心、16 维可见 PoI 坐标以及 3 维可见点统计组成；四帧堆叠后为 **184 维**。隐藏 PoI 不进入策略输入，尾部几何统计也只使用当前可见点。

当前训练奖励为：

$$
r_t=-0.45\,\overline d(\Gamma_t,P_t)-0.30(1-C_t^{\mathrm{plan}}).
$$

其中 `C_plan` 是基于 PoI 点采样的计划覆盖近似，距离项衡量航线与动态兴趣点的接近程度。

| 协议 | 观测与执行 | 用途 |
| --- | --- | --- |
| 训练及训练期 eval | 默认 `traj_obs_source=trajectory`，轨迹邻域可见性，每 3 步采样；tensor 环境推进 | 学习和训练回报检查 |
| joint UAV 评测 | 256 点闭合折线、固定 LOS 执行器、每步按实际 UAV 位置刷新观测 | 同回合报告计划与执行覆盖 |

joint 执行器默认参数为 `dt=0.1`、`v0=0.9`、`v_max=1.8`、`omega_max=1.5`、`k_psi=2.5`、`k_c=0.15`、前视距离 `0.6`。能耗按 `v_n=1.8`、`omega_n=2.4` 归一化，表示控制努力 surrogate，不代表电池能耗。

评测输出外层计划覆盖 `T_theta`、实际覆盖 `uav_T_theta`、计划执行 gap、UAV–PoI 距离、扫掠 PoI 数、跟踪误差及 effort/coverage。AUC 是 step 均值，`integral` 是 step 求和。真实 PoI 和全局覆盖指标只用于 reward/报告计算，不传给 planner。

## 安装

在 Linux 或 WSL2 下使用 **Python 3.9**。GPU 训练需要支持 CUDA 的 PyTorch；CPU 模式可用于评测。

克隆本仓库，并创建独立环境：

```bash
git clone https://github.com/RottenAppleBUAA/Closed-Path-Shaping.git
cd Closed-Path-Shaping
python3.9 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

[requirements.txt](requirements.txt) 指定 PyTorch `2.5.1`、TorchRL `0.6.0`、TensorDict `0.6.2`，以及数值计算、配置和日志依赖，并通过 `-e .` 安装仓库内附带的任务包。如需指定 CPU 或 CUDA wheel，可先按 [PyTorch 安装说明](https://pytorch.org/get-started/previous-versions/#v251) 安装 PyTorch，再安装其余依赖。

环境直接使用 PyTorch tensor 和 Gymnasium 空间运行，不需要机器人资产、SAPIEN 或刚体物理仿真器。训练和评测使用本仓库提供的代码，不依赖同级项目、外部 checkpoint 或机器专用环境。

## 训练

```bash
cd examples/baselines/tdmpc2
python train.py --config-name config_traj_shape_pomdp_default \
  seed=123 exp_name=default-pomdp-1m
```

默认：256 个 GPU tensor 环境、1M env steps、`model_size=5`、MPPI horizon 3、512 samples、64 elites、24 policy-prior trajectories、6 次迭代；`steps_per_update=16`、`pretrain_updates=1280`。W&B 日志默认关闭。

训练在 baseline 工作目录下生成：

```text
logs/TrajectoryShapePOMDPV2CoreReward-v1/123/default-pomdp-1m/
  config.yaml
  models/final.pt
  eval.csv          # 发生训练期 eval 时生成
```

保存的配置记录模型结构、观测设置和规划预算。下方评测命令中的 checkpoint 由这一步训练生成；仓库不附带预训练权重。

## joint UAV 评测

在 `examples/baselines/tdmpc2` 中评测训练得到的 checkpoint：

```bash
python evaluate_pomdp_default.py \
  --checkpoint logs/TrajectoryShapePOMDPV2CoreReward-v1/123/default-pomdp-1m/models/final.pt \
  --seeds 20260518,20260519,20260520,20260521,20260522 \
  --steps 512 --device cuda \
  --output-dir outputs/default-pomdp-1m-eval
```

输出目录必须不存在。评测自动读取 checkpoint 附近的训练配置，也可通过 `--config` 指定；CPU 评测可使用 `--device cpu`。产物为 `episodes.csv`、`steps_seed_<seed>.csv`、`summary.json` 和有效配置，记录 checkpoint SHA256、seed、规划预算、外层/UAV 指标以及跨 seed 汇总。

## 代码地图

| 模块 | 入口 |
| --- | --- |
| POMDP-v2 环境和闭合航线动作 | [trajectory_shape.py](mani_skill/envs/tasks/trajectory_shape/trajectory_shape.py) |
| Tensor 向量环境 | [trajectory_shape_vector.py](examples/baselines/tdmpc2/envs/trajectory_shape_vector.py) |
| default 世界模型与规划器 | [world_model.py](examples/baselines/tdmpc2/common/world_model.py)、[tdmpc2.py](examples/baselines/tdmpc2/tdmpc2.py) |
| 训练和默认配置 | [train.py](examples/baselines/tdmpc2/train.py)、[config_traj_shape_pomdp_default.yaml](examples/baselines/tdmpc2/config_traj_shape_pomdp_default.yaml) |
| 固定 UAV 跟踪与报告指标 | [evaluation/](examples/baselines/tdmpc2/evaluation/) |
| joint 评测入口 | [evaluate_pomdp_default.py](examples/baselines/tdmpc2/evaluate_pomdp_default.py) |

## 背景与许可

任务接口和向量化 TD-MPC2 实现源自 ManiSkill 3.0.1，本仓库仅保留闭合航线塑形所需的 tensor 任务和规划器。LOS 跟踪器作为独立执行模块随仓库提供。

原始许可声明见 [LICENSE](LICENSE) 和 [LICENSE-3RD-PARTY](LICENSE-3RD-PARTY)、[LICENSE-TD-MPC2](LICENSE-TD-MPC2)，提取的参考执行器声明保留在 [evaluation/LICENSE_REFERENCE](examples/baselines/tdmpc2/evaluation/LICENSE_REFERENCE)。
