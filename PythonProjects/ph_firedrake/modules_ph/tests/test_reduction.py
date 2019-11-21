import numpy as np
from reduction_phdae import ortho
from reduction_phdae import splitting
from scipy.linalg import block_diag

n=8
m=3
k=3

M1 = 3*np.eye(m)
M2 = 5*np.eye(k)

E = block_diag(M1, M2, np.zeros((n-m-k, n-m-k)))

V = np.random.rand(n, n)
W = np.zeros((0,0))

tol = 1e-10

V = ortho(V, np.zeros((0,0)), E, tol)

W1 = V[:m, :m]; W2 = V[m:m+k, m:m+k]

W1, W2 = splitting(W1, W2, M1, M2, tol)

V1 = ortho(W1, np.zeros((0, 0)), M1, tol)
V2 = ortho(W2, np.zeros((0, 0)), M2, tol)


