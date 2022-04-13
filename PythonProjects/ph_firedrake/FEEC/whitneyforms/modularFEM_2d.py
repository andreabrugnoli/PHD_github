from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

tol = 1e-10

n_el = int(input("Enter number of element : "))
L = n_el*np.sqrt(2)

deg = 1

mesh = RectangleMesh(n_el, n_el, L, L)
n_ver = FacetNormal(mesh)

P_0 = FiniteElement("CG", triangle, deg)
P_1e = FiniteElement("N1curl", triangle, deg)
P_1f = FiniteElement("RT", triangle, deg)
P_2 = FiniteElement("DG", triangle, deg-1)

V_0 = FunctionSpace(mesh, P_0)
V_1e = FunctionSpace(mesh, P_1e)
V_1f = FunctionSpace(mesh, P_1f)
V_2 = FunctionSpace(mesh, P_2)

v_0 = TestFunction(V_0)
u_0 = TrialFunction(V_0)

v_1e = TestFunction(V_1e)
u_1e = TrialFunction(V_1e)

v_1f = TestFunction(V_1f)
u_1f = TrialFunction(V_1f)

v_2 = TestFunction(V_2)
u_2 = TrialFunction(V_2)

# Primal variables
v2_pr = TestFunction(V_2)
v1e_pr = TestFunction(V_1e)

f2_pr = TrialFunction(V_2)
f1e_pr = TrialFunction(V_1e)

e0_pr = TrialFunction(V_0)
e1f_pr = TrialFunction(V_1f)


# Dual variables
v0_dl = TestFunction(V_0)
v1f_dl = TestFunction(V_1f)

f0_dl = TrialFunction(V_0)
f1f_dl = TrialFunction(V_1f)

e2_dl = TrialFunction(V_2)
e1e_dl = TrialFunction(V_1e)

# Construction of the primal system
m_2 = v2_pr * f2_pr * dx
m_1e = dot(v1e_pr, f1e_pr) * dx

d_21 = v2_pr * div(e1f_pr) * dx
d_10 = dot(v1e_pr, grad(e0_pr)) * dx

M2_petsc = assemble(m_2, mat_type='aij')
M1e_petsc = assemble(m_1e, mat_type='aij')

D21_petsc = assemble(d_21, mat_type='aij')
D10_petsc = assemble(d_10, mat_type='aij')

M1e_mat = csr_matrix(M1e_petsc.M.handle.getValuesCSR()[::-1])
M2_mat = csr_matrix(M2_petsc.M.handle.getValuesCSR()[::-1])

D21_mat = csr_matrix(D21_petsc.M.handle.getValuesCSR()[::-1])
D10_mat = csr_matrix(D10_petsc.M.handle.getValuesCSR()[::-1])

M2_dense = csr_matrix.todense(M2_mat)
M1e_dense = csr_matrix.todense(M1e_mat)

D21_dense = csr_matrix.todense(D21_mat)
D10_dense = csr_matrix.todense(D10_mat)


# Print M matrix
print("M1e matrix : ")
print(M1e_dense)
print("M2 matrix : ")
print(M2_dense)
# D matrix
print("D10 matrix : ")
print(D10_dense)
print("D21 matrix : ")
print(D21_dense)

M2_mat.tocsc()
M1e_mat.tocsc()

D10_mat.tocsc()
D21_mat.tocsc()

D_0 = spsolve(M1e_mat, D10_mat)
# D_0.tolil()
D_0_dense = csr_matrix.todense(D_0)

D_1 = spsolve(M2_mat, D21_mat)
# D_0.tolil()
D_1_dense = csr_matrix.todense(D_1)

# Coincidence matrix
D_0_dense[abs(D_0_dense) < tol] = 0.0

print("d0 coincidence matrix ")
print(D_0_dense)
print("d1 coincidence matrix ")
print(D_1_dense)

# D_0[abs(D_0) < tol] = 0.0
# print("D_0")
# print(D_0)
# print(D_0.shape)
# print(D_0[abs(D_0)>tol])

# Dual matrices

m_0 = v0_dl * f0_dl * dx
m_1f = dot(v1f_dl, f1f_dl) * dx

adj_d_21 = - dot(grad(v0_dl), e1e_dl) * dx
adj_d_10 = - div(v1f_dl) * e2_dl * dx

bd_form_01 = v0_dl * dot(e1f_pr, n_ver) * ds
bd_form_10 = dot(v1f_dl, n_ver) * e0_pr * ds


M0_petsc = assemble(m_0, mat_type='aij')
M1f_petsc = assemble(m_1f, mat_type='aij')

adj_D10_petsc = assemble(adj_d_10, mat_type='aij')
adj_D21_petsc = assemble(adj_d_21, mat_type='aij')

B_01_petsc = assemble(bd_form_01, mat_type='aij')
B_10_petsc = assemble(bd_form_10, mat_type='aij')

M0_mat = csr_matrix(M0_petsc.M.handle.getValuesCSR()[::-1])
M1f_mat = csr_matrix(M1f_petsc.M.handle.getValuesCSR()[::-1])

adj_D10_mat = csr_matrix(adj_D10_petsc.M.handle.getValuesCSR()[::-1])
adj_D21_mat = csr_matrix(adj_D21_petsc.M.handle.getValuesCSR()[::-1])

B_10_mat = csr_matrix(B_10_petsc.M.handle.getValuesCSR()[::-1])
B_01_mat = csr_matrix(B_01_petsc.M.handle.getValuesCSR()[::-1])

# M0_mat.tocsc()
# adj_D10_mat.tocsc()
# B_00_mat.tocsc()

M0_dense = csr_matrix.todense(M0_mat)
M1f_dense = csr_matrix.todense(M1f_mat)

adj_D10_dense = csr_matrix.todense(adj_D10_mat)
adj_D21_dense = csr_matrix.todense(adj_D21_mat)

B_01_dense = csr_matrix.todense(B_01_mat)
B_10_dense = csr_matrix.todense(B_10_mat)


# Print M0 matrix
print("M0 matrix : ")
print(M0_dense)
print("M1f matrix : ")
print(M1f_dense)
# Adjoint D10 matrix
print("adjoint D10 matrix")
print(adj_D10_dense.shape)
print(adj_D10_dense)
print("adjoint D21 matrix")
print(adj_D21_dense.shape)
print(adj_D21_dense)
print("B01 matrix")
print(B_01_dense)
print(B_01_dense.shape)
print("B10 matrix")
print(B_10_dense)
print(B_10_dense.shape)



