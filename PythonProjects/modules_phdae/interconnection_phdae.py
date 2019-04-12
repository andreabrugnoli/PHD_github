import numpy as np
import scipy.linalg as la
from classes_phdae import SysPhdae


def transformer(sys1, sys2, ind1, ind2, C):
    """Transformer interconnection of pHDAE systems considering the following convection
    u1_int = - C^T u2_int
    y2_int = C y1_int
    """

    assert isinstance(sys1, SysPhdae)
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

    sys_int = SysPhdae(n_aug, E=E_aug, J=J_aug, R=R_aug, Q=Q_aug, B=B_aug)

    return sys_int
