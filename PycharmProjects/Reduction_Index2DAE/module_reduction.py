import numpy as np


def krylov(E, A, B, s0, L, tol):

    n = E.shape[0]
    m = B.shape[1]

    W = np.zeros((n, m*L))
    r = np.solve((s0*E + A), B)
    r = ortho(r, [], E, tol)
    W[:, :m] = r

    for l in range(1,L-1):
        r = np.solve((s0*E + A), E @ r)
        r = ortho(r, W, E, tol)
        W[:, l*m:(l+1)*m] = r

    if s0 == 0:
        r = np.solve((s0*E + A), E @ r)
        r = ortho(r, W, E, tol)

    return W, r

def ortho(V, W, E, tol):

    m = V.shape[1]
    n = W.shape[1]
    d = np.zeros((m, ))
    for k in range(m):
        # orthogonalize with respect to W
        for r in range(2):
            for j in range(n):
                hk1j = W[:, j].T @ E @ V[:, k]
                V[:, k] = V[:, k] - W[:, j] * hk1j

        # orthogonalize with respect to V
        for r in range(2):
            for j in range(k-1):
                d_j = np.sqrt(V[:, j].T @ E @ V[:, j])
                if d_j < tol:
                    continue

                hk1j = W[:, j].T @ E @ V[:, k]
                V[:, k] = V[:, k] - W[:, j] * hk1j

            # normalize
            d[k] = np.sqrt(V[:, k].T @ E @ V[:, k])
            if d[k] >= tol:
                V[:, k] = V[:, k]/d[k]

    # Only keep relevant vector
    V = V[:, d > tol]
    return V



