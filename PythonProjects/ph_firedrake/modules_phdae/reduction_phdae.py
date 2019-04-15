import numpy as np
from scipy.linalg import solve_triangular
from scipy.linalg import null_space
import pygsvd


def proj_matrices(E, A, B, s0, L, n1, n2, tol):
    """Reduction of pHDAE based on the article
    On structure preserving model reduction for damped wave propagation in transport network
    Matrices are supposed to have the structure

    A = [-D -G^T -N^T
          G  0    0
          N  0    0];

    E = [M1   0  0
          0  M2  0
          0   0  0];

    B = [B1
          0
          0];

    """

    M1 = E[:n1, :n1]; M2 = E[n1:n2, n1:n2]
    W, r = krylov(E, A, B, s0, L, tol)

    W1 = W[:n1, :]; W2 = W[n1:n2, :]

    W1, W2 = splitting(W1, W2, M1, M2, tol)

    G = A[n1:n2, :n1]
    nullG = null_space(G)


    if s0 == 0.0:
        x1_L = r[:n1, :]
        V1, V2 = modifyAt0(W1, W2, M1, M2, nullG, x1_L, tol)
    else:
        V1, V2 = modify(W1, W2, M1, M2, nullG, tol)

    return V1, V2



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
    W2 = solve_triangular(R1, U2)

    W1 = W1[:, kc]
    W2 = W2[:, ks]

    return W1, W2


def modify(W1, W2, M1, M2, nullG, tol):
    V1 = ortho(np.concatenate((W1, nullG), axis=1), np.zeros((0, 0)), M1, tol)
    V2 = ortho(W2, np.zeros((0, 0)), M2, tol)

    return V1, V2


def modifyAt0(W1, W2, M1, M2, nullG, x1_L, tol):
    V1 = ortho(np.concatenate((W1, x1_L, nullG), axis=1), np.zeros((0, 0)), M1, tol)
    V2 = ortho(W2, np.zeros((0, 0)), M2, tol)

    return V1, V2
