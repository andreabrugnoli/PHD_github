import numpy as np
from math import ceil, floor

from scipy import linalg as la
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as inv_sp


class StormerVerletGrad( object ):

    def __init__(self, Mp, Mq, Dp, Dq, Rp, Pp, Fp=0):
        self.Mp = Mp; self.Mq = Mq
        self.Dp = Dp; self.Dq = Dq
        self.Rp = Rp

        self.n_p = len(Mp)
        self.n_q = len(Mq)
        self.Pp = Pp
        self.Fp = Fp

    def compute_sol(self, ep0, eq0, t_f, t_0=0.0, dt=1e-6, n_ev=100):

        t_ev = np.linspace(t_0, t_f, n_ev)

        ep_sol = np.zeros((self.n_p, n_ev))
        eq_sol = np.zeros((self.n_q, n_ev))

        ep_sol[:, 0] = ep0
        eq_sol[:, 0] = eq0

        n_t = ceil((t_f - t_0) / dt)

        if n_ev > n_t:
            raise ValueError("Choose less evaluation points")

        Ap = self.Mp
        invAp = inv_sp(csc_matrix(Ap)).toarray()


        Ap_ctrl = self.Mp + dt / 2 * self.Pp @ self.Rp
        invAp_ctrl = inv_sp(csc_matrix(Ap_ctrl)).toarray()

        Aq = self.Mq
        invAq = inv_sp(csc_matrix(Aq)).toarray()

        ep_old = ep0
        eq_old = eq0

        t_stop = 0.25 * t_f

        k = 1
        for i in range(1, n_t + 1):
            t = t_0 + i * dt

            # Integration for p (n+1/2)
            bp = self.Mp @ ep_old + dt / 2 * self.Pp @ (self.Dq @ eq_old + self.Fp * (t < t_stop))

            if t < t_stop:
                ep_new = invAp @ bp
            else: ep_new = invAp_ctrl @ bp
            ep_old = ep_new

            # Integration of q (n+1)

            bq = self.Mq @ eq_old + dt * self.Dp @ ep_new
            eq_new = invAq @ bq
            eq_old = eq_new

            # Integration for p (n+1)
            bp = self.Mp @ ep_new + dt / 2 * self.Pp @ (self.Dq @ eq_old + self.Fp * (t < t_stop))

            if t < t_stop:
                ep_new = invAp @ bp
            else: ep_new = invAp_ctrl @ bp

            ep_old = ep_new

            #print(str(t), str(t_ev[k]), str(k))
            if k < n_ev and t >= t_ev[k]:
                ep_sol[:, k] = ep_new
                eq_sol[:, k] = eq_new
                print(str(int(k / (n_ev-1) * 100)), '% of the solution computed ')
                k = k + 1
            elif k == n_ev:
                break

        sol = Sol(t_ev, ep_sol, eq_sol)

        return sol


class Sol:
    def __init__(self, t_ev, ep_sol, eq_sol):
        self.t_ev = t_ev
        self.ep_sol = ep_sol
        self.eq_sol = eq_sol
