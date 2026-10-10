"""Self-contained LOS/phase tracker extracted from the task's reference executor."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple
import numpy as np

def wrap_angle(rad: float) -> float:
    """Wrap a heading error to [-pi, pi) for shortest-turn feedback."""

    return float((rad + math.pi) % (2.0 * math.pi) - math.pi)

def clamp(x: float, lo: float, hi: float) -> float:
    """Clip a scalar control or offset to its admissible interval."""

    return float(min(max(x, lo), hi))

def rotate90_ccw(v: np.ndarray) -> np.ndarray:
    """Rotate a planar tangent by 90 degrees to obtain its left normal."""

    v = np.asarray(v, dtype=np.float64).reshape(2)
    return np.array([-v[1], v[0]], dtype=np.float64)

def _build_closed_arclength(polyline: np.ndarray, eps: float = 1e-9) -> Tuple[np.ndarray, float]:
    """
    Build cumulative segment lengths [K + 1] and the closed-path length.
    The last segment connects back to the first point; a repeated endpoint
    is removed before measuring the segments.
    """

    pts = np.asarray(polyline, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("polyline 必须为形状 (K,2) 的二维点序列")
    if pts.shape[0] < 2:
        raise ValueError("polyline 至少需要 2 个点")
    # Remove a duplicate endpoint; closure is represented by the final segment.
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
    Resample world-coordinate path points into a closed polyline [M, 2].
    Interpolate at uniform arc-length phases, then optionally apply a periodic
    moving average to smooth the executor reference.
    """

    pts = np.asarray(curve, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("curve 必须为形状 (K,2) 的二维点序列")
    if pts.shape[0] < 2:
        raise ValueError("curve 至少需要 2 个点")

    # Build the arc-length grid used to resample the closed curve.
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

    # Smooth across the periodic boundary to reduce sharp reference turns.
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
    Closed-polyline geometry for projection and arc-length lookup.
    Queries return local tangent/normal frames for phase and LOS feedback.
    """

    def __init__(self, polyline: np.ndarray, eps: float = 1e-9) -> None:
        """Cache closed segments and lengths for repeated controller queries."""
        # Closed-path vertices [K, 2]; an explicit duplicate endpoint is not required.
        self._poly = np.asarray(polyline, dtype=np.float64)
        self._eps = float(eps)
        self._s_grid, self._L = _build_closed_arclength(self._poly, eps=self._eps)
        # Cache segment geometry for repeated projection in the control loop.
        self._seg = np.roll(self._poly, shift=-1, axis=0) - self._poly
        self._seg_len = np.linalg.norm(self._seg, axis=1)
        self._seg_len2 = np.sum(self._seg * self._seg, axis=1)

    @property
    def length(self) -> float:
        """Return the total arc length of the closed reference path."""

        return float(self._L)

    @property
    def polyline(self) -> np.ndarray:
        """Return the stored reference vertices [K, 2] without copying."""

        return self._poly

    def project(self, x: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
        """
        Project a point onto the nearest nondegenerate closed-path segment.
        Returns arc-length phase, nearest point, unit tangent, left normal,
        and signed cross-track error (positive on the left).
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
            # Fall back to the first vertex when no usable segment remains.
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
        Interpolate a point and tangent/normal frame at arc-length phase s.
        Phases wrap modulo the closed-path length.
        """

        s_mod = float(s) % float(self._L)
        # The arc-length grid has K + 1 entries, ending at the total length.
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
    Planar UAV execution state: positions [N, 2] and headings [N].
    The rollout integrates speed and turn-rate controls using unicycle kinematics.
    """

    positions: np.ndarray  # World-coordinate positions [N, 2].
    headings: np.ndarray  # Heading angles [N], in radians.

    def __post_init__(self) -> None:
        """Validate team dimensions and store execution state as float64 arrays."""
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
    Track a closed path using phase spacing and line-of-sight heading feedback.
    Speed responds to forward/backward arc-length gaps; turn rate follows
    a lookahead point. Optional offset and avoidance branches are disabled
    by the default joint-evaluation profile.
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
        # Optional normal offset: shift the lookahead point by b along its normal.
        use_spillover: bool = False,
        k_r: float = 0.0,
        k_d: float = 0.0,
        b_max: float = 0.0,
        demand_normal_estimator: Optional[
            Callable[[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], int], float]
        ] = None,
        # Optional short-range avoidance feedback.
        use_avoid: bool = False,
        avoid_d0: float = 0.0,
        avoid_k_psi: float = 0.0,
        eps: float = 1e-9,
    ) -> None:
        """Store executor gains, actuator bounds, and optional feedback settings."""
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
        """Clear the per-agent normal-offset state at the start of an episode."""

        self.b = np.zeros(self.N, dtype=np.float64)

    def step(
        self,
        state: UnicycleSwarmState,
        heatfield: Optional[np.ndarray] = None,
        weightfield: Optional[np.ndarray] = None,
        return_debug: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[Dict[str, np.ndarray]]]:
        """
        Compute per-agent speeds and turn rates from the current execution state.
        Project positions, balance cyclic spacing, and steer toward lookahead
        points. Optional debug output contains phase, gaps, and cross-track errors.
        """

        pos = np.asarray(state.positions, dtype=np.float64)
        psi = np.asarray(state.headings, dtype=np.float64)
        if pos.shape[0] != self.N:
            raise ValueError(f"状态智能体数量不匹配：controller.N={self.N}，state.N={pos.shape[0]}")

        # Project positions onto the reference to recover phase and local geometry.
        s = np.zeros(self.N, dtype=np.float64)
        p = np.zeros((self.N, 2), dtype=np.float64)
        t = np.zeros((self.N, 2), dtype=np.float64)
        n = np.zeros((self.N, 2), dtype=np.float64)
        e_perp = np.zeros(self.N, dtype=np.float64)
        for i in range(self.N):
            s[i], p[i], t[i], n[i], e_perp[i] = self.path.project(pos[i])

        # Balance cyclic arc-length gaps by adjusting forward speed.
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

        # Update the optional normal offset before constructing lookahead points.
        if self.use_spillover:
            for i in range(self.N):
                g_dot_n = float(self._demand_normal_estimator(pos[i], n[i], heatfield, weightfield, i))
                # Bound the demand signal before integrating and clipping the offset.
                sat = math.tanh(g_dot_n)
                b_dot = -self.k_r * float(self.b[i]) + self.k_d * float(sat)
                self.b[i] = clamp(float(self.b[i] + self.dt * b_dot), -self.b_max, self.b_max)
        else:
            self.b[:] = 0.0

        # Steer toward the lookahead point with wrapped heading-error feedback.
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

            # Optionally turn away from close neighbors and reduce forward speed.
            if self.use_avoid and self.avoid_d0 > 0.0 and self.avoid_k_psi > 0.0:
                rep = np.zeros(2, dtype=np.float64)
                for j in range(self.N):
                    if j == i:
                        continue
                    d = pos[i] - pos[j]
                    dist = float(np.linalg.norm(d))
                    if dist <= self.eps or dist >= self.avoid_d0:
                        continue
                    # Weight repulsion more strongly for nearby agents.
                    rep += d / (dist * dist + self.eps)
                if float(np.dot(rep, rep)) > self.eps:
                    psi_rep = float(math.atan2(rep[1], rep[0]))
                    e_rep = wrap_angle(psi_rep - float(psi[i]))
                    omega += self.avoid_k_psi * float(e_rep)
                    # Interpolate speed toward v_min as the nearest neighbor approaches.
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
