# General PHDAE
import numpy as np
import scipy.linalg as la


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

    def transformer(sys1, sys2, ind1, ind2, C):
        """Transformer interconnection of pHDAE systems considering the following convection
        u1_int = - C^T u2_int
        y2_int =   C y1_int
        This kind of connection is of use for connecting bodies in a multibody system
        """

        assert isinstance(sys2, SysPhdae)
        assert C.shape == (len(ind1), len(ind2))

        J_int = la.block_diag(sys1.J, sys2.J)
        R_int = la.block_diag(sys1.R, sys2.R)
        Q_int = la.block_diag(sys1.Q, sys2.Q)

        m1 = len(sys1.B.T)
        m2 = len(sys2.B.T)

        m1int = len(ind1)
        m2int = len(ind2)

        m1ext = m1 - m1int
        m2ext = m2 - m2int

        n_lmb = m1int

        ind1_bol = np.array([(i in ind1) for i in range(m1)])
        ind2_bol = np.array([(i in ind2) for i in range(m2)])

        B1int = sys1.B[:, ind1_bol]
        B2int = sys2.B[:, ind2_bol]

        B1ext = sys1.B[:, ~ind1_bol]
        B2ext = sys2.B[:, ~ind2_bol]

        G_lmb = np.concatenate((-B1int @ C.T, B2int))

        n = len(J_int)
        n_aug = n + m1int

        J_aug = np.zeros((n_aug, n_aug))
        J_aug[:n, :n] = J_int
        J_aug[:n, n:] = G_lmb
        J_aug[n:, :n] = -G_lmb.T

        E_aug = la.block_diag(sys1.E, sys2.E, np.zeros((n_lmb, n_lmb)))
        R_aug = la.block_diag(R_int, np.zeros((n_lmb, n_lmb)))
        Q_aug = la.block_diag(Q_int, np.eye(n_lmb))

        Bext = la.block_diag(B1ext, B2ext)
        B_aug = np.concatenate((Bext, np.zeros((n_lmb, m1ext + m2ext))))

        n_lmb = sys1.n_lmb + sys2.n_lmb + m2int

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

        nf_end = n_r + n_p + n_q
        np_end = n_r + n_p

        M_r = E[:n_r, :n_r]
        M_fr = E[n_r:np_end, :n_r]
        M_rf = E[:n_r, n_r:np_end]

        if n_p > 0:
            assert M_fr.all() == M_rf.T.all()
        if n_r > 0:
            assert J[:n_r].all() == 0.0 and J[:, :n_r].all() == 0.0
        if n_r > 0 and n_p > 0:
            assert E[:n_r, np_end:].all() == 0.0 and E[np_end:, :n_r].all() == 0.0

        if self.n_lmb != 0:
            G_r = J[:n_r, nf_end:]
        else:
            G_r = np.empty((n_r, n_lmb))

        self.n_r = n_r

        self.M_r = M_r
        self.M_fr = M_fr
        self.B_r = B[:n_r]
        self.G_r = G_r

        n_fl = n_p + n_q + n_r
        self.n_p = n_p
        self.n_q = n_q
        self.M_f = self.E[n_r:n_fl, n_r:n_fl]
        self.J_f = self.J[n_r:n_fl, n_r:n_fl]
        self.R_f = self.R[n_r:n_fl, n_r:n_fl]
        self.Q_f = self.Q[n_r:n_fl, n_r:n_fl]
        self.B_f = self.B[n_r:n_fl]
        self.G_f = self.J[n_r:n_fl, n_fl:]

        self.G = np.concatenate((self.G_r, self.G_f))



    def transformer_ordered(sys1, sys2, ind1, ind2, C):

        assert isinstance(sys2, SysPhdaeRig)

        sys_mixed = sys1.transformer(sys2, ind1, ind2, C)

        n_r1 = sys1.n_r
        n_r2 = sys2.n_r

        n_p1 = sys1.n_p
        n_p2 = sys2.n_p

        n_q1 = sys1.n_q
        n_q2 = sys2.n_q

        n1_end = sys1.n
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

        perm_ind = nr_index + np_index + nq_index + nlmb_index + nfin_index
        print(perm_ind)

        E_perm = permute_rows_columns(sys_mixed.E, perm_ind)
        J_perm = permute_rows_columns(sys_mixed.J, perm_ind)
        R_perm = permute_rows_columns(sys_mixed.R, perm_ind)
        Q_perm = permute_rows_columns(sys_mixed.Q, perm_ind)
        B_perm = permute_rows(sys_mixed.B, perm_ind)

        sys_ordered = SysPhdaeRig(n_int, nlmb_int, nr_int, np_int, nq_int, \
                                      E=E_perm, J=J_perm, R=R_perm, Q=Q_perm, B=B_perm)

        return sys_ordered



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
