import numpy as np
import scipy.linalg as la


def transformer(sys1, sys2, ind1, ind2, C):

    assert C.shape == (len(ind1), len(ind2))

    J_int = la.block_diag(sys1.J, sys2.J)
    R_int = la.block_diag(sys1.R, sys2.R)

    m1 = len(sys1.B.T)
    m2 = len(sys2.B.T)

    B1int = sys1.B[:, ind1]
    B2int = sys2.B[:, ind1]

    ind1_bol = np.array([(i in ind1) for i in range(m1)])

    G_lmb =