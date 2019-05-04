import numpy as np
from scipy.linalg import solve_triangular, null_space, block_diag
import pygsvd


def proj_matrices(E, A, B, s0, L, npw, np_tot, nfl_tot, tol=1e-14):
    """Reduction of pHDAE based on the article
    On structure preserving model reduction for damped wave propagation in transport network
    Matrices are supposed to have the structure

    A = [-Dth    0  -Gw^T   -Nw^T
          0   -Dth -Gth^T  -Nth^T
          Gw  Gth       0       0
          Nw  Nth       0       0];

    E = [Mpw   0  0   0
          0  Mpth  0  0
          0   0  Mq   0
          0   0   0   0];

    B = [Bpw
         Bpth
          0
          Blmb];

    """

    Mpw = E[:npw, :npw]; Mpth= E[npw:np_tot, npw:np_tot]; Mq = E[np_tot:nfl_tot, np_tot:nfl_tot]

    Mp = block_diag(Mpw, Mpth)

    W, r = krylov(E, A, B, s0, L, tol)

    Wp = W[:np_tot, :]; Wq = W[np_tot:nfl_tot, :]

    Wp, Wq = splitting(Wp, Wq, Mp, Mq, tol)
    Wpw = Wp[:npw, :]; Wpth = Wp[npw:np_tot, :]

    Wpw, Wpth = splitting(Wpw, Wpth, Mpw, Mpth, tol)

    Gw = A[np_tot:nfl_tot, :npw]
    Gth = A[np_tot:nfl_tot, npw:np_tot]

    nullGw = null_space(Gw)
    nullGth = null_space(Gth)


    if s0 == 0.0:
        xpw_L = r[:npw, :]
        xpth_L = r[npw:np_tot, :]
        Vpw, Vpth, Vq = modifyAt0(Wpw, Wpth, Wq, Mpw, Mpth, Mq, nullGw, nullGth, xpw_L, xpth_L, tol)
    else:
        Vpw, Vpth, Vq = modify(Wpw, Wpth, Wq, Mpw, Mpth, Mq, nullGw, nullGth, tol)

    nw_red = Vpw.shape[1]
    nth_red = Vpth.shape[1]
    return Vpw, Vpth, Vq, nw_red, nth_red


def krylov(E, A, B, s0, L, tol):
    n = E.shape[0]
    if len(B.shape) == 1:
        B = B.reshape((-1, 1))
    m = B.shape[1]
    W = np.zeros((n, m * L))
    r = np.linalg.solve((s0 * E - A), B)
    r = ortho(r, np.zeros((0, 0)), E, tol)
    W[:, :m] = r

    for l in range(1, L):
        r = np.linalg.solve((s0 * E - A), E @ r)
        r = ortho(r, W[:, :l*m], E, tol)
        W[:, l * m:(l + 1) * m] = r

    if s0 == 0:
        r = np.linalg.solve((s0 * E - A), E @ r)
        r = ortho(r, W, E, tol)

    return W, r


def ortho(V, W, E, tol):

    if len(V.shape) == 1:
        V = V.reshape((-1, 1))
    if len(W.shape) == 1:
        W = W.reshape((-1, 1))

    m = V.shape[1]
    n = W.shape[1]

    d = np.zeros((m,))
    for k in range(m):
        # orthogonalize with respect to W
        for r in range(2):  # rehorthogonalization
            for j in range(n):
                hk1j = W[:, j].T @ E @ V[:, k]
                V[:, k] = V[:, k] - W[:, j] * hk1j

        # orthogonalize with respect to V
        for r in range(2):
            for j in range(k):
                d_j = np.sqrt(V[:, j].T @ E @ V[:, j])
                if d_j < tol:
                    continue

                hk1j = V[:, j].T @ E @ V[:, k]
                V[:, k] = V[:, k] - V[:, j] * hk1j

            # normalize
            d[k] = np.sqrt(V[:, k].T @ E @ V[:, k])
            if d[k] > tol:
                V[:, k] = V[:, k] / d[k]
    # Only keep relevant vector
    V = V[:, d > tol]
    return V


def splitting(W1, W2, M1, M2, tol):
    L1 = np.linalg.cholesky(M1)
    L2 = np.linalg.cholesky(M2)

    R1 = L1.T
    R2 = L2.T

    C, S, X, U1, U2 = pygsvd.gsvd(R1 @ W1, R2 @ W2, extras='uv')

    kc = np.logical_and(C > tol, S < 1 - tol)
    ks = np.logical_and(S > tol, C < 1 - tol)

    # kc = np.square(C) > tol
    # ks = np.square(S) > tol

    W1 = solve_triangular(R1, U1)
    W2 = solve_triangular(R2, U2)

    W1 = W1[:, kc]
    W2 = W2[:, ks]

    return W1, W2


def modify(W1, W2, W3, M1, M2, M3, nullG1, nullG2, tol):
    V1 = ortho(np.concatenate((W1, nullG1), axis=1), np.zeros((0, 0)), M1, tol)
    V2 = ortho(np.concatenate((W2, nullG2), axis=1), np.zeros((0, 0)), M2, tol)
    V3 = ortho(W3, np.zeros((0, 0)), M3, tol)

    return V1, V2, V3


def modifyAt0(W1, W2, W3, M1, M2, M3, nullG1, nullG2, x1_L, x2_L, tol):
    V1 = ortho(np.concatenate((W1, x1_L, nullG1), axis=1), np.zeros((0, 0)), M1, tol)
    V2 = ortho(np.concatenate((W2, x2_L, nullG2), axis=1), np.zeros((0, 0)), M2, tol)
    V3 = ortho(W3, np.zeros((0, 0)), M3, tol)

    return V1, V2, V3
