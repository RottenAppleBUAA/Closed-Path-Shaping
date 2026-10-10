"""Self-contained LOS/phase tracker extracted from the task's reference executor."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple
import numpy as np

def wrap_angle(rad: float) -> float:
    """
    逻辑说明：将角度归一到 (-π, π] 以消除航向误差的 2π 周期不连续性。
    数学原理：利用取模运算 rad↦(rad+π) mod 2π - π，使误差始终选择最短旋转方向。
    设计目的：保证航向误差 e_ψ 的符号与幅值连续，避免控制律在 π 附近抖振或反向跳变。
    """

    return float((rad + math.pi) % (2.0 * math.pi) - math.pi)

def clamp(x: float, lo: float, hi: float) -> float:
    """
    逻辑说明：对标量执行区间裁剪，保证速度/偏置等控制量处于安全边界内。
    数学原理：实现 x↦min(max(x,lo),hi)，对应对可行域的投影算子 Π_[lo,hi](x)。
    设计目的：在不改变控制律结构的前提下约束控制输入，提升数值稳定性与工程可落地性。
    """

    return float(min(max(x, lo), hi))

def rotate90_ccw(v: np.ndarray) -> np.ndarray:
    """
    逻辑说明：将二维向量逆时针旋转 90°，由切向量生成左法向量。
    数学原理：若 v=[v_x,v_y]^T，则 R_{90}v=[-v_y,v_x]^T，对应二维正交变换。
    设计目的：在路径跟踪中构造局部 Frenet 标架 (t,n)，用于定义有符号横向误差 e_⊥。
    """

    v = np.asarray(v, dtype=np.float64).reshape(2)
    return np.array([-v[1], v[0]], dtype=np.float64)

def _build_closed_arclength(polyline: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, float]:
    """
    逻辑说明：为闭合折线构造顶点弧长累积表 s_grid，并返回总弧长 L。
    数学原理：对每段长度 ℓ_k=||p_{k+1}-p_k|| 求累和得到 s_k=∑_{j<k}ℓ_j，闭合段使用 p_K≡p_0。
    设计目的：支持 O(logK) 的按弧长插值 point_at_s，以及在投影后快速计算相位 s_i。
    """

    pts = np.asarray(polyline, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("polyline 必须为形状 (K,2) 的二维点序列")
    if pts.shape[0] < 2:
        raise ValueError("polyline 至少需要 2 个点")
    # 逻辑说明：若首尾重复点导致零长度段，会干扰投影与切向计算，这里做一次去重。
    # 数学原理：当 ||p_0-p_{K-1}||≈0 时移除末端点，保持闭合仍由最后一段隐式连接实现。
    # 设计目的：避免 seg_len≈0 造成除零或随机法向方向。
    if np.linalg.norm(pts[0] - pts[-1]) <= eps and pts.shape[0] >= 3:
        pts = pts[:-1]
    diffs = np.roll(pts, shift=-1, axis=0) - pts
    seg_lens = np.linalg.norm(diffs, axis=1)
    seg_lens = np.where(seg_lens <= eps, 0.0, seg_lens)
    s_grid = np.concatenate([[0.0], np.cumsum(seg_lens)], axis=0)
    L = float(s_grid[-1])
    if L <= eps:
        raise ValueError("polyline 总弧长过小（可能所有点重合）")
    return s_grid, L

def curve_to_polyline(
    curve: np.ndarray,
    *,
    num_points: int = 256,
    smoothing_window: int = 7,
    eps: float = 1e-9,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    逻辑说明：将 TrajectoryShapeEnv 的解析离散曲线（物理坐标点序列）重采样为等弧长闭合折线，并可选执行环形平滑。
    数学原理：先计算闭合弧长参数 s_k 与总弧长 L，再在 s∈[0,L) 上等间隔取样并做段内线性插值；平滑采用周期滑动平均（低通滤波）。
    设计目的：避免从像素环反求半径带来的量化误差/中心漂移，使“世界模型生成轨迹”效果与 trajectory_shape_eval_viz.py 的 env._discretize_curve() 更一致。
    """

    pts = np.asarray(curve, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("curve 必须为形状 (K,2) 的二维点序列")
    if pts.shape[0] < 2:
        raise ValueError("curve 至少需要 2 个点")

    # 逻辑说明：对闭合曲线构造弧长累计表，用于等弧长重采样；若首尾重复则在表构造中自动去除。
    # 数学原理：s_grid[k+1]=s_grid[k]+||p_{k+1}-p_k||，并以 p_K≡p_0 构造闭合段。
    # 设计目的：使输出折线在弧长意义上均匀离散，提升下游投影/切向估计的数值稳定性。
    s_grid, L = _build_closed_arclength(pts, eps=float(eps))
    K = int(pts.shape[0])
    diffs = np.roll(pts, shift=-1, axis=0) - pts
    seg_len = np.linalg.norm(diffs, axis=1)

    M = int(num_points)
    if M <= 2:
        raise ValueError("num_points 必须≥3")
    s_targets = np.linspace(0.0, float(L), M, endpoint=False, dtype=np.float64)
    out = np.zeros((M, 2), dtype=np.float64)
    for i, s in enumerate(s_targets):
        idx = int(np.searchsorted(s_grid, s, side="right") - 1)
        idx = int(np.clip(idx, 0, K - 1))
        den = float(seg_len[idx])
        if den <= float(eps):
            out[i] = pts[idx]
            continue
        tau = float((s - float(s_grid[idx])) / den)
        tau = clamp(tau, 0.0, 1.0)
        out[i] = pts[idx] + tau * diffs[idx]

    # 逻辑说明：对等弧长点序列做环形滑动平均，抑制局部折线尖角与高频噪声。
    # 数学原理：p_k ← (1/W)∑_{j=-h}^{h} p_{(k+j) mod M}，等价于对周期序列做有限长卷积低通滤波。
    # 设计目的：提升 LOS 前视点方向与法向估计的连续性，减少跟踪环的航向抖动。
    win = int(smoothing_window)
    if win > 1 and M > 2:
        if win % 2 == 0:
            win += 1
        half = win // 2
        if half > 0:
            smooth = np.zeros_like(out)
            for k in range(M):
                acc = np.zeros(2, dtype=np.float64)
                for j in range(-half, half + 1):
                    acc += out[(k + j) % M]
                smooth[k] = acc / float(win)
            out = smooth

    debug: Dict[str, np.ndarray] = {
        "s_grid": np.asarray(s_grid, dtype=np.float64),
        "length": np.asarray([float(L)], dtype=np.float64),
    }
    return out.astype(np.float64), debug

class ReferenceTrajectory:
    """
    逻辑说明：以闭合折线（polyline）表示参考轨迹，并提供“投影到轨迹”和“按弧长取点”的几何查询接口。
    数学原理：将连续轨迹用分段线性曲线近似，对每段执行最近点投影；弧长相位 s∈[0,L) 通过段内线性插值实现。
    设计目的：为 LOS 导引与相位均匀化层提供统一的局部几何量 (p,t,n,e_⊥,s)，并保持依赖轻量、可替换。
    """

    def __init__(self, polyline: np.ndarray, eps: float = 1e-9) -> None:
        # polyline：闭合折线顶点序列 (K,2)，默认首尾不必重复
        self._poly = np.asarray(polyline, dtype=np.float64)
        self._eps = float(eps)
        self._s_grid, self._L = _build_closed_arclength(self._poly, eps=self._eps)
        # 预计算段向量与段长度，避免在高频控制循环中重复开销
        self._seg = np.roll(self._poly, shift=-1, axis=0) - self._poly
        self._seg_len = np.linalg.norm(self._seg, axis=1)
        self._seg_len2 = np.sum(self._seg * self._seg, axis=1)

    @property
    def length(self) -> float:
        """闭合轨迹总弧长 L。"""

        return float(self._L)

    @property
    def polyline(self) -> np.ndarray:
        """轨迹折线顶点（只读视图）。"""

        return self._poly

    def project(self, x: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        逻辑说明：将点 x 投影到闭合折线上，返回弧长相位 s、最近点 p、单位切向 t、单位法向 n 与有符号横向误差 e_⊥。
        数学原理：对每段 [p_k,p_{k+1}] 求最近点 p_k+τ·(p_{k+1}-p_k)，其中 τ=clip(((x-p_k)·d)/||d||²,0,1)，
                  取距离最小的段；切向 t=d/||d||，法向 n=R_{90}t，横向误差 e_⊥=(x-p)·n。
        设计目的：在 O(K) 的轻量计算下提供 Frenet 几何量，支撑相位控制与 LOS 航向控制的核心反馈信号。
        """

        x = np.asarray(x, dtype=np.float64).reshape(2)
        best_i = -1
        best_tau = 0.0
        best_dist2 = float("inf")
        for i in range(self._poly.shape[0]):
            len2 = float(self._seg_len2[i])
            if len2 <= self._eps:
                continue
            p0 = self._poly[i]
            d = self._seg[i]
            tau = float(np.dot(x - p0, d) / len2)
            tau = clamp(tau, 0.0, 1.0)
            proj = p0 + tau * d
            dist2 = float(np.dot(x - proj, x - proj))
            if dist2 < best_dist2:
                best_dist2 = dist2
                best_i = i
                best_tau = tau
        if best_i < 0:
            # 退化保护：若所有段长度过小，则退回到第 0 个点
            p = self._poly[0].copy()
            t = np.array([1.0, 0.0], dtype=np.float64)
            n = rotate90_ccw(t)
            e_perp = float(np.dot(x - p, n))
            return 0.0, p, t, n, e_perp

        p0 = self._poly[best_i]
        d = self._seg[best_i]
        seg_len = float(self._seg_len[best_i])
        if seg_len <= self._eps:
            t = np.array([1.0, 0.0], dtype=np.float64)
        else:
            t = d / seg_len
        n = rotate90_ccw(t)
        p = p0 + best_tau * d
        s = float(self._s_grid[best_i] + best_tau * seg_len) % float(self._L)
        e_perp = float(np.dot(x - p, n))
        return s, p, t, n, e_perp

    def point_at_s(self, s: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        逻辑说明：按弧长相位 s∈[0,L) 在线性段上插值，返回轨迹点 p(s)、单位切向 t(s) 与单位法向 n(s)。
        数学原理：用 searchsorted 找到满足 s_k≤s<s_{k+1} 的段 k，并取 τ=(s-s_k)/ℓ_k，令 p=p_k+τ·d_k；
                  切向 t=d_k/ℓ_k，法向 n=R_{90}t。
        设计目的：为 LOS 前视点计算提供“按弧长前推”的连续接口，避免依赖角度参数化或外部样条库。
        """

        s_mod = float(s) % float(self._L)
        # s_grid 长度为 K+1，最后一项为 L
        idx = int(np.searchsorted(self._s_grid, s_mod, side="right") - 1)
        idx = int(np.clip(idx, 0, self._poly.shape[0] - 1))
        seg_len = float(self._seg_len[idx])
        if seg_len <= self._eps:
            p = self._poly[idx].copy()
            t = np.array([1.0, 0.0], dtype=np.float64)
            n = rotate90_ccw(t)
            return p, t, n
        tau = (s_mod - float(self._s_grid[idx])) / seg_len
        tau = clamp(float(tau), 0.0, 1.0)
        p0 = self._poly[idx]
        d = self._seg[idx]
        p = p0 + tau * d
        t = d / seg_len
        n = rotate90_ccw(t)
        return p, t, n

@dataclass
class UnicycleSwarmState:
    """
    逻辑说明：以最小状态集描述多智能体单轨迹（unicycle）运动学：位置与航向。
    数学原理：每个体满足 ẋ=v cosψ, ẏ=v sinψ, ψ̇=ω；控制器输出 (v,ω) 用于离散积分推进。
    设计目的：与 SimpleCoverageSimEnv 的状态语义对齐，使该基线可直接作为“名义控制器”接入残差 RL 或仿真评估。
    """

    positions: np.ndarray  # positions：形状 (N,2) 的二维位置
    headings: np.ndarray  # headings：形状 (N,) 的航向角 ψ（弧度）

    def __post_init__(self) -> None:
        pos = np.asarray(self.positions)
        hdg = np.asarray(self.headings)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError("positions 必须为 (N,2)")
        if hdg.ndim != 1 or hdg.shape[0] != pos.shape[0]:
            raise ValueError("headings 必须为 (N,) 且与 positions 的 N 一致")
        self.positions = pos.astype(np.float64, copy=False)
        self.headings = hdg.astype(np.float64, copy=False)

class LOSPhaseController:
    """
    逻辑说明：实现“投影→相位均匀化→LOS 航向”三段式名义控制律，输出每个智能体的 (v_nom, ω_nom)。
    数学原理：
      1) 投影层：x_i↦(s_i,p_i,t_i,n_i,e_{⊥,i})；
      2) 相位层：v_{∥,i}=v0+k_c(Δs_{i→i+1}-Δs_{i-1→i})，在环上驱动间距趋于 L/N；
      3) LOS 层：ψ_ref=atan2(p_la-x_i)，ω=k_ψ·wrap(ψ_ref-ψ_i)。
    设计目的：作为内环基线控制器，提供可解释的等间距巡航与贴轨能力，并保留“补漏偏置 b_i”与“轻量避碰”可选模块接口。
    """

    def __init__(
        self,
        ref_traj: ReferenceTrajectory,
        num_agents: int,
        dt: float,
        v0: float,
        v_min: float,
        v_max: float,
        k_psi: float,
        k_c: float,
        lookahead: float,
        omega_max: Optional[float] = None,
        sort_by_phase: bool = True,
        # 溢出补漏偏置（可选）：ḃ=-k_r b + k_d sat(g·n)，前视点沿法向平移 p_la←p_la+b·n
        use_spillover: bool = False,
        k_r: float = 0.0,
        k_d: float = 0.0,
        b_max: float = 0.0,
        demand_normal_estimator: Optional[
            Callable[[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], int], float]
        ] = None,
        # 轻量避碰（可选）
        use_avoid: bool = False,
        avoid_d0: float = 0.0,
        avoid_k_psi: float = 0.0,
        eps: float = 1e-9,
    ) -> None:
        self.path = ref_traj
        self.N = int(num_agents)
        self.dt = float(dt)
        self.v0 = float(v0)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.k_psi = float(k_psi)
        self.k_c = float(k_c)
        self.lookahead = float(lookahead)
        self.omega_max = None if omega_max is None else float(omega_max)
        self.sort_by_phase = bool(sort_by_phase)

        self.use_spillover = bool(use_spillover)
        self.k_r = float(k_r)
        self.k_d = float(k_d)
        self.b_max = float(b_max)
        self._demand_normal_estimator = demand_normal_estimator
        self.b = np.zeros(self.N, dtype=np.float64)

        self.use_avoid = bool(use_avoid)
        self.avoid_d0 = float(avoid_d0)
        self.avoid_k_psi = float(avoid_k_psi)
        self.eps = float(eps)

        if self.N <= 0:
            raise ValueError("num_agents 必须为正")
        if self.dt <= 0.0:
            raise ValueError("dt 必须为正")
        if not (self.v_min <= self.v0 <= self.v_max):
            raise ValueError("需满足 v_min ≤ v0 ≤ v_max")
        if self.v_max < self.v_min:
            raise ValueError("需满足 v_max ≥ v_min")
        if self.lookahead <= 0.0:
            raise ValueError("lookahead 必须为正（弧长单位）")
        if self.use_spillover and self._demand_normal_estimator is None:
            raise ValueError("启用 use_spillover 时必须提供 demand_normal_estimator 回调")

    def reset(self) -> None:
        """
        逻辑说明：重置补漏偏置状态 b_i，确保每个 episode/实验的初始条件可控。
        数学原理：令 b_i(0)=0，相当于从名义轨迹本身开始，不引入额外法向偏置。
        设计目的：满足可复现实验要求，并避免跨 episode 的隐状态泄漏影响统计对照。
        """

        self.b = np.zeros(self.N, dtype=np.float64)

    def step(
        self,
        state: UnicycleSwarmState,
        heatfield: Optional[np.ndarray] = None,
        weightfield: Optional[np.ndarray] = None,
        return_debug: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        逻辑说明：基于当前群体状态计算名义控制 (v_nom, ω_nom)，可选返回用于验收的中间几何量。
        数学原理：先投影得到相位 s_i，再按环一致性计算 v_{∥,i}，最后用 LOS 前视点生成航向参考并做比例控制。
        设计目的：作为可插拔的内环基线，为外环（如世界模型规划或 RL 残差）提供稳定、可解释的名义动作。
        """

        pos = np.asarray(state.positions, dtype=np.float64)
        psi = np.asarray(state.headings, dtype=np.float64)
        if pos.shape[0] != self.N:
            raise ValueError(f"状态智能体数量不匹配：controller.N={self.N}，state.N={pos.shape[0]}")

        # 1) 投影到参考轨迹：得到 s_i 与局部几何量
        s = np.zeros(self.N, dtype=np.float64)
        p = np.zeros((self.N, 2), dtype=np.float64)
        t = np.zeros((self.N, 2), dtype=np.float64)
        n = np.zeros((self.N, 2), dtype=np.float64)
        e_perp = np.zeros(self.N, dtype=np.float64)
        for i in range(self.N):
            s[i], p[i], t[i], n[i], e_perp[i] = self.path.project(pos[i])

        # 2) 相位均匀化：在环上调节切向推进速度，使弧长间距趋于 L/N
        L = float(self.path.length)
        v_par = np.zeros(self.N, dtype=np.float64)
        gap_fwd = np.zeros(self.N, dtype=np.float64)
        gap_bwd = np.zeros(self.N, dtype=np.float64)
        if self.sort_by_phase:
            order = np.argsort(s)
            s_sorted = s[order]
            for k in range(self.N):
                s_k = float(s_sorted[k])
                s_next = float(s_sorted[(k + 1) % self.N])
                s_prev = float(s_sorted[(k - 1) % self.N])
                gap_f = (s_next - s_k) % L
                gap_b = (s_k - s_prev) % L
                idx_agent = int(order[k])
                gap_fwd[idx_agent] = gap_f
                gap_bwd[idx_agent] = gap_b
                v = self.v0 + self.k_c * (gap_f - gap_b)
                v_par[idx_agent] = clamp(v, self.v_min, self.v_max)
        else:
            for i in range(self.N):
                ip = (i + 1) % self.N
                im = (i - 1) % self.N
                gap_f = (float(s[ip]) - float(s[i])) % L
                gap_b = (float(s[i]) - float(s[im])) % L
                gap_fwd[i] = gap_f
                gap_bwd[i] = gap_b
                v = self.v0 + self.k_c * (gap_f - gap_b)
                v_par[i] = clamp(v, self.v_min, self.v_max)

        # 3) 溢出补漏偏置（可选）：用一阶偏置状态 b_i 平移前视点，实现“短暂离轨补漏”
        if self.use_spillover:
            for i in range(self.N):
                g_dot_n = float(self._demand_normal_estimator(pos[i], n[i], heatfield, weightfield, i))
                # 逻辑说明：用 tanh 将需求信号压到 [-1,1]，避免偏置动力学被异常梯度放大。
                # 数学原理：sat(z)=tanh(z) 为光滑饱和函数，可视为对控制输入的软投影。
                # 设计目的：提升数值稳定性，并保证 b_i 的变化率有界。
                sat = math.tanh(g_dot_n)
                b_dot = -self.k_r * float(self.b[i]) + self.k_d * float(sat)
                self.b[i] = clamp(float(self.b[i] + self.dt * b_dot), -self.b_max, self.b_max)
        else:
            self.b[:] = 0.0

        # 4) LOS 航向控制：将航向拉向前视点方向
        v_nom = np.zeros(self.N, dtype=np.float64)
        w_nom = np.zeros(self.N, dtype=np.float64)
        for i in range(self.N):
            s_la = (float(s[i]) + self.lookahead) % L
            p_la, _, n_la = self.path.point_at_s(s_la)
            p_la = p_la + float(self.b[i]) * n_la
            vec = p_la - pos[i]
            if float(np.dot(vec, vec)) <= self.eps:
                psi_ref = float(psi[i])
            else:
                psi_ref = float(math.atan2(vec[1], vec[0]))
            e_psi = wrap_angle(psi_ref - float(psi[i]))
            omega = self.k_psi * float(e_psi)
            if self.omega_max is not None:
                omega = clamp(omega, -self.omega_max, self.omega_max)
            v = float(v_par[i])

            # 5) 轻量避碰（可选）：在近距离时对航向施加远离邻居的附加 P 反馈，并轻度降速
            if self.use_avoid and self.avoid_d0 > 0.0 and self.avoid_k_psi > 0.0:
                rep = np.zeros(2, dtype=np.float64)
                for j in range(self.N):
                    if j == i:
                        continue
                    d = pos[i] - pos[j]
                    dist = float(np.linalg.norm(d))
                    if dist <= self.eps or dist >= self.avoid_d0:
                        continue
                    # 逻辑说明：采用 1/r 型斥力方向，距离越近斥力越强。
                    # 数学原理：rep += d/(r^2+ε)，相当于势函数 U=∑ 1/(r+ε) 的负梯度方向近似。
                    # 设计目的：提供极简避碰趋势，不引入复杂约束求解器。
                    rep += d / (dist * dist + self.eps)
                if float(np.dot(rep, rep)) > self.eps:
                    psi_rep = float(math.atan2(rep[1], rep[0]))
                    e_rep = wrap_angle(psi_rep - float(psi[i]))
                    omega += self.avoid_k_psi * float(e_rep)
                    # 近距离降速：将 v 按最小距离比例缩放到 [v_min,v]
                    d_min = float("inf")
                    for j in range(self.N):
                        if j == i:
                            continue
                        d_ij = float(np.linalg.norm(pos[i] - pos[j]))
                        d_min = min(d_min, d_ij)
                    if np.isfinite(d_min) and d_min < self.avoid_d0:
                        ratio = clamp(d_min / self.avoid_d0, 0.0, 1.0)
                        v = self.v_min + ratio * (v - self.v_min)

            v_nom[i] = clamp(v, self.v_min, self.v_max)
            w_nom[i] = float(omega)

        debug = None
        if return_debug:
            debug = {
                "s": s,
                "e_perp": e_perp,
                "gap_fwd": gap_fwd,
                "gap_bwd": gap_bwd,
                "b": self.b.copy(),
            }
        return v_nom, w_nom, debug
