# General PHDAE
import numpy as np
import scipy.linalg as la


class SysPhode:
    def __init__(self, n, J=None, R=None, Q=None, B=None):
        """General phdae class. Only the size of the system is needed"""
        assert n > 0 and isinstance(n, int)

        self.n = n
        matr_shape = (n, n)

        if J is not None:
            assert J.shape == matr_shape
            assert check_skew_symmetry(J)
            self.J = J
        else:
            self.J = np.zeros(matr_shape)

        if Q is not None:
            assert Q.shape == matr_shape
            assert check_symmetry(Q)
            assert check_positive_matrix(Q)
            self.Q = Q
        else:
            self.Q = np.eye(n)

        if R is not None:
            assert R.shape == matr_shape
            assert check_symmetry(R)
            self.R = R
        else:
            self.R = np.zeros(matr_shape)

        if B is not None:
            assert len(B) == n
            self.B = B
        else:
            self.B = np.eye(n)

class SysPhdae:
    def __init__(self, n, n_lmb, E=None, J=None, R=None, Q=None, B=None):
        """General phdae class. Only the size of the system is needed"""
        assert n > 0 and isinstance(n, int)
        assert n_lmb >= 0 and isinstance(n_lmb, int)

        self.n = n
        self.n_lmb = n_lmb
        self.n_x = n - n_lmb
        matr_shape = (n, n)

        if E is not None:
            assert E.shape == matr_shape
            assert np.linalg.matrix_rank(E) == n - n_lmb
            self.E = E
        else:
            E = np.eye(n)
            E[-n_lmb:, -n_lmb] = 0.0
            self.E = E

        if J is not None:
            assert J.shape == matr_shape
            assert check_skew_symmetry(J)
            self.J = J
        else:
            self.J = np.zeros(matr_shape)

        if Q is not None:
            assert Q.shape == matr_shape
            assert check_symmetry(Q)
            assert check_positive_matrix(Q)
            self.Q = Q
        else:
            self.Q = np.eye(n)

        if R is not None:
            assert R.shape == matr_shape
            assert check_symmetry(R)
            self.R = R
        else:
            self.R = np.zeros(matr_shape)

        if B is not None:
            assert len(B) == n
            self.B = B
        else:
            self.B = np.eye(n)

    def transformer(self, sys2, ind1, ind2, C):
        """Transformer interconnection of pHDAE systems considering the following convection
        u1_int = - C^T u2_int
        y2_int =   C y1_int
        This kind of connection is of use for connecting bodies in a multibody system
        """

        assert isinstance(sys2, SysPhdae)
        assert C.shape == (len(ind1), len(ind2))

        J_int = la.block_diag(self.J, sys2.J)
        R_int = la.block_diag(self.R, sys2.R)
        Q_int = la.block_diag(self.Q, sys2.Q)

        m1 = len(self.B.T)
        m2 = len(sys2.B.T)

        m1int = len(ind1)
        m2int = len(ind2)

        m1ext = m1 - m1int
        m2ext = m2 - m2int

        n_lmb = m1int

        ind1_bol = np.array([(i in ind1) for i in range(m1)])
        ind2_bol = np.array([(i in ind2) for i in range(m2)])

        B1int = self.B[:, ind1_bol]
        B2int = sys2.B[:, ind2_bol]

        B1ext = self.B[:, ~ind1_bol]
        B2ext = sys2.B[:, ~ind2_bol]

        G_lmb = np.concatenate((-B1int @ C.T, B2int))

        n = len(J_int)
        n_aug = n + m1int

        J_aug = np.zeros((n_aug, n_aug))
        J_aug[:n, :n] = J_int
        J_aug[:n, n:] = G_lmb
        J_aug[n:, :n] = -G_lmb.T

        E_aug = la.block_diag(self.E, sys2.E, np.zeros((n_lmb, n_lmb)))
        R_aug = la.block_diag(R_int, np.zeros((n_lmb, n_lmb)))
        Q_aug = la.block_diag(Q_int, np.eye(n_lmb))

        Bext = la.block_diag(B1ext, B2ext)
        B_aug = np.concatenate((Bext, np.zeros((n_lmb, m1ext + m2ext))))

        n_lmb = self.n_lmb + sys2.n_lmb + m2int

        sys_int = SysPhdae(n_aug, n_lmb, E=E_aug, J=J_aug, R=R_aug, Q=Q_aug, B=B_aug)

        return sys_int




class SysPhdaeRig(SysPhdae):
    """Class for PHDAEs flexible rigid body. The system has structure
    [M_r  M_rf  0] d/dt [e_r]   = [0         0    G_r] [e_r] + [B_r]
    [M_fr  M_f  0]      [e_f]     [0       J_f    G_f] [e_f]   [B_f]
    [0      0   0]      [lmb]     [-G_r^  -G_f^T    0] [lmb]   [  0]

    The matrices have structure:
    J_f = [ 0    A
             -A.T  0];

    M_f = [Mp   0
            0  Mq];

    B_f = [Bp
          0];

    Variables for the rigid motion and constraints might not be present
    """

    def __init__(self, n, n_lmb, n_r, n_p, n_q, E, J, B, R=None, Q=None):
        SysPhdae.__init__(self, n, n_lmb, E=E, J=J, B=B, R=R, Q=Q)

        assert n_r >= 0 and isinstance(n_r, int)
        assert n_lmb >= 0 and isinstance(n_lmb, int)
        assert n_p >= 0 and isinstance(n_p, int)
        assert n_q >= 0 and isinstance(n_q, int)
        assert n_r + n_p + n_q + n_lmb == n

        n_e = n_p + n_q + n_r
        np_end = n_r + n_p

        M_r = E[:n_r, :n_r]
        M_fr = E[n_r:np_end, :n_r]
        M_rf = E[:n_r, n_r:np_end]
        if n_p > 0:
            assert M_fr.all() == M_rf.T.all()
        if n_r > 0:
            assert np.count_nonzero(J[:n_r, :n_e]) == 0 and  np.count_nonzero(J[:n_e, :n_r]) == 0
        if n_r > 0 and n_p > 0:
            assert np.count_nonzero(E[:, n_e:]) == 0 and np.count_nonzero(E[n_e:, :]) == 0

        self.n_r = n_r
        self.M_r = M_r
        self.M_fr = M_fr
        self.B_r = self.B[:n_r]
        self.G_r = J[:n_r, n_e:]

        self.n_p = n_p
        self.n_q = n_q
        self.M_f = self.E[n_r:n_e, n_r:n_e]
        self.J_f = self.J[n_r:n_e, n_r:n_e]
        self.R_f = self.R[n_r:n_e, n_r:n_e]
        self.Q_f = self.Q[n_r:n_e, n_r:n_e]
        self.B_f = self.B[n_r:n_e]
        self.G_f = self.J[n_r:n_e, n_e:]

        self.n_e = n_e
        self.G_e = np.concatenate((self.G_r, self.G_f))
        self.M_e = self.E[:n_e, :n_e]
        self.J_e = self.J[:n_e, :n_e]
        self.B_e = self.B[:n_e]
        self.Q_e = self.Q[:n_e, :n_e]
        self.R_e = self.R[:n_e, :n_e]

    def transformer_ordered(self, sys2, ind1, ind2, C):

        assert isinstance(sys2, SysPhdaeRig)
        sys_mixed = self.transformer(sys2, ind1, ind2, C)
        n_r1 = self.n_r
        n_r2 = sys2.n_r

        n_p1 = self.n_p
        n_p2 = sys2.n_p
        n_q1 = self.n_q
        n_q2 = sys2.n_q

        n1_end = self.n
        n2 = sys2.n
        np1_end = n_r1 + n_p1
        nq1_end = np1_end + n_q1

        nr2_end = n1_end + n_r2
        np2_end = nr2_end + n_p2
        nq2_end = np2_end + n_q2
        n2_end = n1_end + n2

        n_int = sys_mixed.n
        nlmb_int = sys_mixed.n_lmb

        nr_int = n_r1 + n_r2
        np_int = n_p1 + n_p2
        nq_int = n_q1 + n_q2

        nr_index = list(range(n_r1)) + list(range(n1_end, nr2_end))
        np_index = list(range(n_r1, np1_end)) + list(range(nr2_end, np2_end))
        nq_index = list(range(np1_end, nq1_end)) + list(range(np2_end, nq2_end))
        nlmb_index = list(range(nq1_end, n1_end)) + list(range(nq2_end, n2_end))
        nfin_index = list(range(n2_end, n_int))

        perm_index = nr_index + np_index + nq_index + nlmb_index + nfin_index

        E_perm = permute_rows_columns(sys_mixed.E, perm_index)
        J_perm = permute_rows_columns(sys_mixed.J, perm_index)
        R_perm = permute_rows_columns(sys_mixed.R, perm_index)
        Q_perm = permute_rows_columns(sys_mixed.Q, perm_index)
        B_perm = permute_rows(sys_mixed.B, perm_index)

        sys_ordered = SysPhdaeRig(n_int, nlmb_int, nr_int, np_int, nq_int,
                                      E=E_perm, J=J_perm, R=R_perm, Q=Q_perm, B=B_perm)

        return sys_ordered

    def dae_to_ode(self):
        G = self.G_e
        GannL = la.null_space(G.T).T

        T = np.concatenate((GannL, la.inv(G.T @ G) @ G.T))
        invT = la.inv(T)
        print(T)
        assert np.linalg.matrix_rank(T @ G) == self.n_lmb

        Mtil = T @ self.M_e @ T.T
        Jtil = T @ self.J_e @ T.T
        Rtil = T @ self.R_e @ T.T
        Qtil = invT.T @ self.Q_e @ invT
        Btil = T @ self.B_e

        n_z = self.n_e - self.n_lmb
        Qtil11 = Qtil[:n_z, :n_z]
        Qtil22 = Qtil[n_z:, n_z:]
        Qtil12 = Qtil[:n_z, n_z:]
        Qtil21 = Qtil[n_z:, :n_z]
        Qz1to2 =  -la.inv(Qtil22) @ Qtil21
        Q_shur = Qtil11 + Qtil12 @ Qz1to2

        Mtil11 = Mtil[:n_z, :n_z]
        Mtil22 = Mtil[n_z:, n_z:]
        Mtil12 = Mtil[:n_z, n_z:]
        Mtil21 = Mtil[n_z:, :n_z]
        M_Qshur = Mtil11 + Mtil12 @ Qz1to2

        J_ode = Jtil[:n_z, :n_z]
        R_ode = Rtil[:n_z, :n_z]
        Q_ode = Q_shur @ la.inv(M_Qshur)
        B_ode = Btil[:n_z]

        T_ode2dae = T.T @ np.concatenate((np.eye(n_z), Qz1to2)) @ la.inv(M_Qshur)

        sysOde = SysPhode(n_z, J=J_ode, R=R_ode, Q=Q_ode, B=B_ode)

        return sysOde, T_ode2dae






def permute_rows_columns(mat, ind_perm):
    assert len(ind_perm) == len(mat) and len(ind_perm) == len(mat.T)
    mat = mat[ind_perm, :]
    mat = mat[:, ind_perm]
    return mat


def permute_rows(mat, ind_perm):
    assert len(ind_perm) == len(mat)
    mat = mat[ind_perm, :]
    return mat


def check_positive_matrix(mat):
    try:
        np.linalg.cholesky(mat)
        return True
    except np.linalg.LinAlgError:
        return False


tol = 1e-14


def check_symmetry(mat):
    return np.linalg.norm(mat - mat.T) < tol


def check_skew_symmetry(mat):
    return np.linalg.norm(mat + mat.T) < tol
