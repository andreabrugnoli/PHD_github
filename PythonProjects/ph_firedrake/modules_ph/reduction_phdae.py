import numpy as np
from scipy.linalg import solve_triangular
from scipy.linalg import null_space
import pygsvd


def proj_matrices_phdae(E, A, B, s0, L, n1, n2, oper="grad", tol=1e-14):
    """Reduction of pHDAE based on the article
    On structure preserving model reduction for damped wave propagation in transport network
    Matrices are supposed to have the structure

    if grad
    A = [-D1 -G^T -N^T
          G  -D2    0
          N    0    0];

    or if div
    A = [-D1    G   0
         -G^T  -D2 -N^T
          0     N   0];

    E = [M1   0  0
          0  M2  0
          0   0  0];

    B = [B1
          0
          0];

    """
    assert (oper == "grad" or oper == "div")
    M1 = E[:n1, :n1]; M2 = E[n1:n2, n1:n2]
    W, r = krylov(E, A, B, s0, L, tol)

    W1 = W[:n1, :]; W2 = W[n1:n2, :]

    W1, W2 = splitting(W1, W2, M1, M2, tol)

    if oper == "grad":
        G = A[n1:n2, :n1]
        N = A[n2:, :n1]
        x_L = r[:n1, :]
    else:
        G = A[:n1, n1:n2]
        N = A[n2:, n1:n2]
        x_L = r[n1:n2, :]

    if s0 == 0.0:
        V1, V2 = modifyAt0(W1, W2, M1, M2, G, N, x_L, oper, tol)
    else:
        V1, V2 = modify(W1, W2, M1, M2, G, N, oper, tol)

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
    W2 = solve_triangular(R2, U2)

    W1 = W1[:, kc]
    W2 = W2[:, ks]

    return W1, W2


def modify(W1, W2, M1, M2, G, N, oper, tol):
    GN = np.concatenate((G, N))
    nullG = null_space(G)

    if oper=="grad":
        # V1 = ortho(np.concatenate((W1, nullG), axis=1), np.zeros((0, 0)), M1, tol)
        # # V1 = ortho(W1, np.zeros((0, 0)), M1, tol)
        # V2 = ortho(W2, np.zeros((0, 0)), M2, tol)

        V2 = ortho(W2, np.zeros((0, 0)), M2, tol)
        H = GN @ np.linalg.solve(M1, GN.T)
        F = GN.T @ np.linalg.solve(H, np.concatenate((M2 @ V2, np.zeros((N.shape[0], V2.shape[1])))))
        V1 = np.linalg.solve(M1, F)
        V1 = ortho(np.concatenate((V1, nullG), axis=1), np.zeros((0, 0)), M1, tol)
    else:
        # V1 = ortho(W1, np.zeros((0, 0)), M1, tol)
        # # V2 = ortho(W2, np.zeros((0, 0)), M2, tol)
        # V2 = ortho(np.concatenate((W2, nullG), axis=1), np.zeros((0, 0)), M2, tol)

        V1 = ortho(W1, np.zeros((0, 0)), M1, tol)
        H = GN @ np.linalg.solve(M2, GN.T)
        F = GN.T @ np.linalg.solve(H, np.concatenate((M1 @ V1, np.zeros((N.shape[0], V1.shape[1])))))
        V2 = np.linalg.solve(M2, F)
        V2 = ortho(np.concatenate((V2, nullG), axis=1), np.zeros((0, 0)), M2, tol)

    return V1, V2


def modifyAt0(W1, W2, M1, M2, G, N, x_L, oper, tol):

    GN = np.concatenate((G, N))
    nullG = null_space(G)

    if oper=="grad":
        # V1 = ortho(np.concatenate((W1, x_L, nullG), axis=1), np.zeros((0, 0)), M1, tol)
        # V1 = ortho(W1, np.zeros((0, 0)), M1, tol)
        # V2 = ortho(W2, np.zeros((0, 0)), M2, tol)

        V2 = ortho(W2, np.zeros((0, 0)), M2, tol)
        H = GN @ np.linalg.solve(M1, GN.T)
        F = GN.T @ np.linalg.solve(H, np.concatenate((M2 @ V2, np.zeros((N.shape[0], V2.shape[1])))))
        V1 = np.linalg.solve(M1, F)
        V1 = ortho(np.concatenate((V1, x_L, nullG), axis=1), np.zeros((0, 0)), M1, tol)
    else:
        # V1 = ortho(W1, np.zeros((0, 0)), M1, tol)
        # V2 = ortho(np.concatenate((W2, x_L, nullG), axis=1), np.zeros((0, 0)), M2, tol)

        V1 = ortho(W1, np.zeros((0, 0)), M1, tol)
        H = GN @ np.linalg.solve(M2, GN.T)
        F = GN.T @ np.linalg.solve(H, np.concatenate((M1 @ V1, np.zeros((N.shape[0], V1.shape[1])))))
        V2 = np.linalg.solve(M2, F)
        V2 = ortho(np.concatenate((V2, x_L, nullG), axis=1), np.zeros((0, 0)), M2, tol)

    return V1, V2
