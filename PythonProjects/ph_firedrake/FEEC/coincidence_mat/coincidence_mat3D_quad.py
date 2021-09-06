from firedrake import *

import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

tol = 1e-10

L = 1
n_el = 2
deg = 1

mesh_quad = UnitSquareMesh(n_el, n_el, quadrilateral=True)
mesh = ExtrudedMesh(mesh_quad, n_el)

# mesh = CubeMesh(n_el, n_el, n_el, L)

P_0 = FiniteElement("CG", hexahedron, deg)
P_1 = FiniteElement("NCE", hexahedron, deg)
P_2 = FiniteElement("NCF", hexahedron, deg)
P_3 = FiniteElement("DG", hexahedron, deg-1)

V_0 = FunctionSpace(mesh, P_0)
V_1 = FunctionSpace(mesh, P_1)
V_2 = FunctionSpace(mesh, P_2)
V_3 = FunctionSpace(mesh, P_3)

u_0 = TrialFunction(V_0)

v_1 = TestFunction(V_1)
u_1 = TrialFunction(V_1)

v_2 = TestFunction(V_2)
u_2 = TrialFunction(V_2)

v_3 = TestFunction(V_3)
u_3 = TrialFunction(V_3)

# Construction of the D_0 co-incidence matrix
m_1 = dot(v_1, u_1) * dx
d_10 = dot(v_1, grad(u_0)) * dx

M1_petsc = assemble(m_1, mat_type='aij')
D10_petsc = assemble(d_10, mat_type='aij')

M1_mat = csr_matrix(M1_petsc.M.handle.getValuesCSR()[::-1])
D10_mat = csr_matrix(D10_petsc.M.handle.getValuesCSR()[::-1])

M1_mat.tocsc()
D10_mat.tocsc()

D_0 = spsolve(M1_mat, D10_mat)
D_0.tolil()

D_0[abs(D_0) < tol] = 0.0
print("D_0")
# print(D_0)
print(D_0[abs(D_0)>tol])
print(D_0.shape)

# Construction of the D_1 co-incidence matrix
m_2 = dot(v_2, u_2) * dx
d_21 = dot(v_2, curl(u_1)) * dx

M2_petsc = assemble(m_2, mat_type='aij' )
D21_petsc = assemble(d_21, mat_type='aij' )

M2_mat = csr_matrix(M2_petsc.M.handle.getValuesCSR()[::-1])
D21_mat = csr_matrix(D21_petsc.M.handle.getValuesCSR()[::-1])

M2_mat.tocsc()
D21_mat.tocsc()

D_1 = spsolve(M2_mat, D21_mat)
D_1.tolil()

D_1[abs(D_1) < tol] = 0.0
print("D_1")
# print(D_1)
print(D_1[abs(D_1)>tol])
print(D_1.shape)

# Construction of the D_2 co-incidence matrix
m_3 = dot(v_3, u_3) * dx
d_32 = dot(v_3, div(u_2)) * dx

M3_petsc = assemble(m_3, mat_type='aij' )
D32_petsc = assemble(d_32, mat_type='aij' )

M3_mat = csr_matrix(M3_petsc.M.handle.getValuesCSR()[::-1])
D32_mat = csr_matrix(D32_petsc.M.handle.getValuesCSR()[::-1])

M3_mat.tocsc()
D32_mat.tocsc()

D_2 = spsolve(M3_mat, D32_mat)
D_2.tolil()

D_2[abs(D_2) < tol] = 0.0
print("D_2")
# print(D_2)
print(D_2[abs(D_2)>tol])
print(D_2.shape)
