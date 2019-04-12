# General PHDAE
import numpy as np
import scipy.linalg as la


class SysPhdae:
    def __init__(self, n, E=None, J=None, R=None, Q=None, B=None):
        """General phdae class. Only the size of the system is needed"""
        assert n > 0 and isinstance(n, int)

        self.n = n
        matr_shape = (n, n)

        if E is not None:
            assert E.shape == matr_shape
            self.E = E
        else:
            self.E = np.ones(matr_shape)

        if J is not None:
            assert J.shape == matr_shape
            assert check_skew_symmetry(J)
            self.J = J
        else:
            self.J = np.zeros(matr_shape)

        if Q is not None:
            assert Q.shape == matr_shape
            assert check_positive_matrix(Q)
            self.Q = Q
        else:
            self.Q = np.ones(matr_shape)

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
            self.B = np.eye(matr_shape)


class SysPhdaePfemRig(SysPhdae):
    """Class for PHDAEs flexible rigid body. The system has structure
    [M_r  M_rf  0] d/dt [e_r]   = [0         0   G_r] [e_r] + [B_r]
    [M_fr  M_f  0]      [e_f]     [0       J_f   G_f] [e_f]   [B_f]
    [0      0   0]      [lmb]     [-G_r^  -G_f^T   0] [lmb]   [  0]

    The matrices have structure:
    J_f = [0     D
          -D.T  0];

    M_f = [Mp   0
            0  Mq];

    B_f = [Bp
          0];

    """

    def __init__(self, n, n_r, n_lmb, n_p, n_q, E, J, B, R=None, Q=None):
        SysPhdae.__init__(n, E=E, J=J, B=B, R=R, Q=Q)

        assert n_r > 0 and isinstance(n_r, int)
        assert n_lmb >= 0 and isinstance(n_lmb, int)
        assert n_p > 0 and isinstance(n_p, int)
        assert n_q > 0 and isinstance(n_q, int)
        assert n_r + n_p + n_q == n

        nf_end = n_r + n_p + n_q

        M_r = E[:n_r, :n_r]
        M_fr = E[n_r:nf_end, :n_r]
        M_rf = E[:n_r, n_r:]

        assert M_fr.all() == M_rf.T.all()
        assert J[:n_r].all() == 0.0 and J[:, :n_r].all() == 0.0

        self.n_r = n_r

        self.M_r = M_r
        self.M_fr = M_fr
        self.B_r = B[:n_r]

        n_f = n - n_r
        E_f = E[n_r:, n_r:]
        J_f = J[n_r:, n_r:]
        R_f = R[n_r:, n_r:]
        Q_f = Q[n_r:, n_r:]
        B_f = B[n_r:]

        self.flex = SysPhdaePfemFl(n_f, n_p, n_q, n_lmb, E=E_f, J=J_f, B=B_f, R=R_f, Q=Q_f)



class SysPhdaePfemFl(SysPhdae):
    """Class for PFEM generated PHDAE with p first and q second. Only flexible motion is included.
     The matrices have structure:
    J = [0     D   G
         -D.T  0   0
         -G.T  0   0];

    E = [Mp   0   0
          0  Mq   0
          0   0   0];

    B = [Bp
          0
          0];

    Constraint might not be present
    """

    def __init__(self, n, n_p, n_q, n_lmb, E, J, B, R=None, Q=None):
        SysPhdae.__init__(n, E=E, J=J, B=B, R=R, Q=Q)

        assert n_p > 0 and isinstance(n_p, int)
        assert n_q > 0 and isinstance(n_q, int)
        assert n_lmb >= 0 and isinstance(n_lmb, int)
        assert n_p + n_q + n_lmb == n
        n_e = n_p + n_q

        M = E[:n_e, :n_e]
        Mp = M[:n_p, :n_p]
        Mq = M[n_p:n_e, n_p:n_e]
        M = la.block_diag(Mp, Mq)

        assert check_positive_matrix(Mp)
        assert check_positive_matrix(Mq)
        assert E[n_e:, :].all() == 0.0 and E[:, n_e:].all() == 0.0

        if n_lmb !=0 :
            G = J[:n_e, n_e:]
            assert G[n_p:].all() == 0.0
            assert np.linalg.matrix_rank(G) == n_lmb
            self.G = G

        assert B[n_p:].all == 0.0

        self.M = M
        self.J_f = J[:n_e, n_e]
        self.Mp = Mp
        self.Mq = Mq
        self.n_p = n_p
        self.n_q = n_q
        self.n_lmb = n_lmb



def check_positive_matrix(mat):
    try:
        np.linalg.cholesky(mat)
        return 1
    except np.linalg.linalg.LinAlgError as err:
        if 'Matrix is not positive definite' in err.message:
            return 0
        else:
            raise


def check_symmetry(mat):
    tol = 1e-15
    return np.linalg.norm(mat - mat.T) < tol


def check_skew_symmetry(mat):
    tol = 1e-15
    return np.linalg.norm(mat + mat.T) < tol
