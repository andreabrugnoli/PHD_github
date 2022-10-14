from firedrake import *
import numpy as np
from scipy.sparse import csr_matrix

L = 1
n_el = 5
deg = 1

mesh = RectangleMesh(n_el, n_el, L, L)

P_0 = FiniteElement("CG", triangle, deg)
V_0 = FunctionSpace(mesh, P_0)

v_0 = TestFunction(V_0)
u_0 = TrialFunction(V_0)

# Mass matrix
m_0 = dot(v_0, u_0) * dx

M0_petsc = assemble(m_0, mat_type='aij').M.handle    # Petsc format of the matrix
M0_scipy = csr_matrix(M0_petsc.getValuesCSR()[::-1])  # Scipy CSR format
M0_dense = np.array(M0_petsc.convert("dense").getDenseArray())  # Standard numpy dense format


print(M0_petsc)
print(M0_scipy)
print(M0_dense)
