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


# Primal variables
v1_pr = TestFunction(V_1)

f1_pr = TrialFunction(V_1)
e0_pr = TrialFunction(V_0)
# Dual variables
v0_dl = TestFunction(V_0)

f0_dl = TrialFunction(V_0)
e1_dl = TrialFunction(V_1)

# Construction of the D_0 co-incidence matrix
m_1 = v1_pr * f1_pr * dx
d_10 = v1_pr * e0_pr.dx(0) * dx

M1_petsc = assemble(m_1, mat_type='aij' )
D10_petsc = assemble(d_10, mat_type='aij' )

M1_mat = csr_matrix(M1_petsc.M.handle.getValuesCSR()[::-1])
D10_mat = csr_matrix(D10_petsc.M.handle.getValuesCSR()[::-1])

M1_dense = csr_matrix.todense(M1_mat)
D10_dense = csr_matrix.todense(D10_mat)

# Print M0 matrix
print("M1 matrix : ")
print(M1_dense)
# Adjoint D10 matrix
print("D10 matrix : ")
print(D10_dense)

M1_mat.tocsc()
D10_mat.tocsc()

D_0 = spsolve(M1_mat, D10_mat)
# D_0.tolil()
D_0_dense = csr_matrix.todense(D_0)

# Coincidence matrix
print("d0 coincidence matrix ")
print(D_0_dense)

# D_0[abs(D_0) < tol] = 0.0
# print("D_0")
# print(D_0)
# print(D_0.shape)
# print(D_0[abs(D_0)>tol])


m_0 = v0_dl * f0_dl * dx
adj_d_10 = - v0_dl.dx(0) * e1_dl * dx

bd_form_00 = v0_dl * e0_pr * ds

M0_petsc = assemble(m_0, mat_type='aij' )
adj_D01_petsc = assemble(adj_d_10, mat_type='aij' )
B_00_petsc = assemble(bd_form_00, mat_type='aij' )

M0_mat = csr_matrix(M0_petsc.M.handle.getValuesCSR()[::-1])
adj_D10_mat = csr_matrix(adj_D01_petsc.M.handle.getValuesCSR()[::-1])
B_00_mat = csr_matrix(B_00_petsc.M.handle.getValuesCSR()[::-1])

# M0_mat.tocsc()
# adj_D10_mat.tocsc()
# B_00_mat.tocsc()

M0_dense = csr_matrix.todense(M0_mat)
adj_D10_dense = csr_matrix.todense(adj_D10_mat)

B_00_dense = csr_matrix.todense(B_00_mat)


# Print M0 matrix
print("M0 matrix : ")
print(M0_dense)
# Adjoint D10 matrix
print("adjoint D10 matrix")
print(adj_D10_dense.shape)
print(adj_D10_dense)
print("B00 matrix")
print(B_00_dense)
print(B_00_dense.shape)


bd_form_const = v0_dl * Constant(1) * ds
B_vec = assemble(bd_form_const).vector().get_local()
print("B vector")
print(B_vec)


