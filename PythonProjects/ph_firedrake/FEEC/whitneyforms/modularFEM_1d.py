from firedrake import *

import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

tol = 1e-10

n_el = int(input("Enter number of element : "))
L = n_el*1

deg = 1

mesh = IntervalMesh(n_el, L)

P_0 = FiniteElement("CG", interval, deg)
P_1 = FiniteElement("DG", interval, deg-1)

V_0 = FunctionSpace(mesh, P_0)
V_1 = FunctionSpace(mesh, P_1)

v_0 = TestFunction(V_0)
u_0 = TrialFunction(V_0)

v_1 = TestFunction(V_1)
u_1 = TrialFunction(V_1)

# Construction of the D_0 co-incidence matrix
m_1 = v_1* u_1 * dx
d_10 = v_1 * u_0.dx(0) * dx

M1_petsc = assemble(m_1, mat_type='aij' )
D10_petsc = assemble(d_10, mat_type='aij' )

M1_mat = csr_matrix(M1_petsc.M.handle.getValuesCSR()[::-1])
D10_mat = csr_matrix(D10_petsc.M.handle.getValuesCSR()[::-1])

M1_mat.tocsc()
D10_mat.tocsc()

D_0 = spsolve(M1_mat, D10_mat)
# D_0.tolil()

D_0_dense = csr_matrix.todense(D_0)

print(D_0_dense)

# D_0[abs(D_0) < tol] = 0.0
# print("D_0")
# print(D_0)
# print(D_0.shape)
# print(D_0[abs(D_0)>tol])
