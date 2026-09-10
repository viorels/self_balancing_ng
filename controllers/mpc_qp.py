"""
Sparse linear-MPC quadratic programme for the tribot, solved with OSQP.

Decision vector (in this order):
    z_0 .. z_N        state trajectory        (NX each)
    u_0 .. u_{N-1}    input trajectory        (NU each)
    e_1 .. e_N        pitch-limit slack, >= 0 (scalar each)

Cost:
    sum_k  (z_k - zr_k)' Q (z_k - zr_k) + (u_k - ur)' R (u_k - ur)
         + (u_k - u_{k-1})' Rd (u_k - u_{k-1})
         + (z_N - zr_N)' P_N (z_N - zr_N)
         + sum_k  rho1 e_k + rho2 e_k^2

Constraints:
    z_0 = z_meas
    z_{k+1} = Ad z_k + Bd u_k + cd
    u_min <= u_k <= u_max
    |theta_k| <= theta_max + e_k         (soft)
    |u_t,k + kappa u_d,k| <= tip_max     (4WD tipping; +inf otherwise)
    e_k >= 0

The sparsity pattern is fixed at construction so that every subsequent
solve only updates the numeric values (OSQP `update`), which keeps the
per-tick cost to a warm-started solve.
"""

from __future__ import annotations

import time

import numpy as np
import osqp
import scipy.sparse as sp

from .mpc_plant import NX, NU


class MPCQP:
    """Builds and re-solves the MPC QP with a fixed sparsity pattern."""

    THETA_IDX = 2      # index of theta in the state vector

    def __init__(self, N: int, Q: np.ndarray, R: np.ndarray, Rd: np.ndarray,
                 theta_max: float, rho1: float, rho2: float,
                 max_iter: int = 400, eps: float = 1e-4):
        self.N = N
        self.Q = np.asarray(Q, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.Rd = np.asarray(Rd, dtype=float)
        self.theta_max = theta_max
        self.rho1 = rho1
        self.rho2 = rho2

        n_z = NX * (N + 1)
        n_u = NU * N
        n_e = N
        self.n = n_z + n_u + n_e
        self.iz = lambda k: NX * k
        self.iu = lambda k: n_z + NU * k
        self.ie = lambda k: n_z + n_u + (k - 1)

        # Constraint row offsets
        self.r_init = 0
        self.r_dyn = NX
        self.r_ubox = self.r_dyn + NX * N
        self.r_pitch = self.r_ubox + NU * N
        self.r_tip = self.r_pitch + 2 * N
        self.r_slack = self.r_tip + N
        self.m = self.r_slack + N

        self._build_pattern()
        self._P_terminal = np.zeros((NX, NX))
        self._Pval = self._cost_values(self.Q, self._P_terminal)
        self._q = np.zeros(self.n)
        self._l = np.zeros(self.m)
        self._u = np.zeros(self.m)

        # Placeholder model to set up the solver
        Ad = np.eye(NX)
        Bd = np.zeros((NX, NU))
        cd = np.zeros(NX)
        self._Aval = self._constraint_values(Ad, Bd, 0.0)
        self._fill_bounds(np.zeros(NX), cd, np.full(NU, -1.0), np.full(NU, 1.0),
                          np.inf)

        self.prob = osqp.OSQP()
        self.prob.setup(P=self._P_csc(self._Pval), q=self._q,
                        A=self._A_csc(self._Aval), l=self._l, u=self._u,
                        verbose=False, warm_starting=True,
                        max_iter=max_iter, eps_abs=eps, eps_rel=eps,
                        polishing=False, adaptive_rho=True,
                        check_termination=10)

        self.last_solve_ms = 0.0
        self.last_status = ''
        self.last_iter = 0
        self.solve_count = 0
        self.fail_count = 0
        self._x_prev = None

    # ------------------------------------------------------------------
    # Sparsity patterns
    # ------------------------------------------------------------------

    def _build_pattern(self):
        """Enumerate every structural entry of P and A once."""
        N = self.N
        # --- Cost matrix P (upper triangle only) ---
        rows, cols = [], []
        for k in range(N + 1):
            base = self.iz(k)
            for i in range(NX):
                for j in range(i, NX):
                    rows.append(base + i)
                    cols.append(base + j)
        for k in range(N):
            base = self.iu(k)
            for i in range(NU):
                for j in range(i, NU):
                    rows.append(base + i)
                    cols.append(base + j)
        for k in range(1, N):
            b0, b1 = self.iu(k - 1), self.iu(k)
            for i in range(NU):
                for j in range(NU):
                    rows.append(b0 + i)
                    cols.append(b1 + j)
        for k in range(1, N + 1):
            rows.append(self.ie(k))
            cols.append(self.ie(k))
        self._P_rows = np.array(rows)
        self._P_cols = np.array(cols)
        self._P_perm = self._csc_permutation(self._P_rows, self._P_cols,
                                             (self.n, self.n))

        # --- Constraint matrix A ---
        rows, cols = [], []
        # z_0 = z_meas
        for i in range(NX):
            rows.append(self.r_init + i)
            cols.append(self.iz(0) + i)
        # dynamics: -Ad z_k - Bd u_k + z_{k+1} = cd
        for k in range(N):
            r0 = self.r_dyn + NX * k
            for i in range(NX):
                for j in range(NX):
                    rows.append(r0 + i)
                    cols.append(self.iz(k) + j)
                for j in range(NU):
                    rows.append(r0 + i)
                    cols.append(self.iu(k) + j)
                rows.append(r0 + i)
                cols.append(self.iz(k + 1) + i)
        # input box
        for k in range(N):
            for j in range(NU):
                rows.append(self.r_ubox + NU * k + j)
                cols.append(self.iu(k) + j)
        # pitch soft limit: theta_k - e_k <= theta_max ; theta_k + e_k >= -theta_max
        for k in range(1, N + 1):
            r = self.r_pitch + 2 * (k - 1)
            rows += [r, r, r + 1, r + 1]
            cols += [self.iz(k) + self.THETA_IDX, self.ie(k),
                     self.iz(k) + self.THETA_IDX, self.ie(k)]
        # tipping: u_t + kappa u_d
        for k in range(N):
            r = self.r_tip + k
            rows += [r, r]
            cols += [self.iu(k) + 0, self.iu(k) + 1]
        # slack >= 0
        for k in range(1, N + 1):
            rows.append(self.r_slack + (k - 1))
            cols.append(self.ie(k))
        self._A_rows = np.array(rows)
        self._A_cols = np.array(cols)
        self._A_perm = self._csc_permutation(self._A_rows, self._A_cols,
                                             (self.m, self.n))

    @staticmethod
    def _csc_permutation(rows, cols, shape):
        """Permutation from entry order to CSC data order (no duplicates)."""
        idx = np.arange(len(rows), dtype=float) + 1.0
        M = sp.coo_matrix((idx, (rows, cols)), shape=shape).tocsc()
        return (M.data - 1.0).astype(int)

    def _P_csc(self, vals):
        return sp.csc_matrix((vals[self._P_perm],
                              (self._P_rows[self._P_perm],
                               self._P_cols[self._P_perm])),
                             shape=(self.n, self.n))

    def _A_csc(self, vals):
        return sp.csc_matrix((vals[self._A_perm],
                              (self._A_rows[self._A_perm],
                               self._A_cols[self._A_perm])),
                             shape=(self.m, self.n))

    # ------------------------------------------------------------------
    # Numeric values
    # ------------------------------------------------------------------

    def _cost_values(self, Q, P_terminal):
        """Values of the upper-triangular P (OSQP form 1/2 x'Px)."""
        N = self.N
        vals = []
        for k in range(N + 1):
            W = P_terminal if k == N else Q
            for i in range(NX):
                for j in range(i, NX):
                    vals.append(2.0 * W[i, j])
        for k in range(N):
            W = self.R + self.Rd * (2.0 if 0 < k < N - 1 else 1.0)
            if N == 1:
                W = self.R
            for i in range(NU):
                for j in range(i, NU):
                    vals.append(2.0 * W[i, j])
        for k in range(1, N):
            for i in range(NU):
                for j in range(NU):
                    vals.append(-2.0 * self.Rd[i, j])
        for k in range(1, N + 1):
            vals.append(2.0 * self.rho2)
        return np.array(vals)

    def _constraint_values(self, Ad, Bd, kappa):
        N = self.N
        vals = []
        vals += [1.0] * NX
        for k in range(N):
            for i in range(NX):
                vals += list(-Ad[i, :])
                vals += list(-Bd[i, :])
                vals.append(1.0)
        vals += [1.0] * (NU * N)
        for k in range(1, N + 1):
            vals += [1.0, -1.0, 1.0, 1.0]
        for k in range(N):
            vals += [kappa, 1.0]
        vals += [1.0] * N
        return np.array(vals)

    def _fill_bounds(self, z0, cd, u_min, u_max, tip_max):
        N = self.N
        l, u = self._l, self._u
        l[self.r_init:self.r_init + NX] = z0
        u[self.r_init:self.r_init + NX] = z0
        for k in range(N):
            r0 = self.r_dyn + NX * k
            l[r0:r0 + NX] = cd
            u[r0:r0 + NX] = cd
        for k in range(N):
            r0 = self.r_ubox + NU * k
            l[r0:r0 + NU] = u_min
            u[r0:r0 + NU] = u_max
        for k in range(1, N + 1):
            r = self.r_pitch + 2 * (k - 1)
            l[r], u[r] = -np.inf, self.theta_max
            l[r + 1], u[r + 1] = -self.theta_max, np.inf
        l[self.r_tip:self.r_tip + N] = -tip_max
        u[self.r_tip:self.r_tip + N] = tip_max
        l[self.r_slack:self.r_slack + N] = 0.0
        u[self.r_slack:self.r_slack + N] = np.inf

    def _fill_linear(self, z_ref, u_ref, P_terminal):
        """Linear cost term q from references."""
        N = self.N
        q = self._q
        q[:] = 0.0
        for k in range(N + 1):
            W = P_terminal if k == N else self.Q
            q[self.iz(k):self.iz(k) + NX] = -2.0 * (W @ z_ref[k])
        for k in range(N):
            q[self.iu(k):self.iu(k) + NU] = -2.0 * (self.R @ u_ref)
        for k in range(1, N + 1):
            q[self.ie(k)] = self.rho1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_weights(self, Q: np.ndarray, P_terminal: np.ndarray):
        """Update the state weights and terminal cost (same sparsity)."""
        self.Q = np.asarray(Q, dtype=float)
        self._P_terminal = np.asarray(P_terminal, dtype=float)
        self._Pval = self._cost_values(self.Q, self._P_terminal)
        self.prob.update(Px=self._Pval[self._P_perm])

    def solve(self, z0, Ad, Bd, cd, z_ref, u_ref, u_min, u_max,
              kappa=0.0, tip_max=np.inf):
        """
        Solve for the current state and model.

        z_ref: (N+1, NX) array of state references.
        Returns (u_traj (N, NU), z_traj (N+1, NX), ok).
        """
        self._Aval = self._constraint_values(Ad, Bd, kappa)
        self._fill_bounds(z0, cd, u_min, u_max, tip_max)
        self._fill_linear(z_ref, u_ref, self._P_terminal)
        self.prob.update(q=self._q, l=self._l, u=self._u,
                         Ax=self._Aval[self._A_perm])

        t0 = time.perf_counter()
        res = self.prob.solve()
        self.last_solve_ms = (time.perf_counter() - t0) * 1e3
        self.last_status = res.info.status
        self.last_iter = int(res.info.iter)
        self.solve_count += 1

        ok = res.info.status_val in (1, 2)   # solved / solved inaccurate
        x = res.x
        if not ok or x is None or not np.all(np.isfinite(x)):
            self.fail_count += 1
            if self._x_prev is None:
                return None, None, False
            x = self._x_prev
            ok = False
        else:
            self._x_prev = x.copy()

        n_z = NX * (self.N + 1)
        z_traj = x[:n_z].reshape(self.N + 1, NX)
        u_traj = x[n_z:n_z + NU * self.N].reshape(self.N, NU)
        return u_traj, z_traj, ok
