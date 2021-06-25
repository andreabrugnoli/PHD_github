from fenics import *
import mshr

import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from vedo.dolfin import plot

tol = 1e-10

L = 1
n_el = 1
deg = 1

mesh = BoxMesh(Point(0,0,0), Point(L, L, L), n_el, n_el, n_el)

# domain = mshr.Box(Point(0,0,0), Point(L,L,L))
# mesh = mshr.generate_mesh(domain, n_el)

# mesh_plot = plot(mesh) # mode="mesh", interactive=0

# vmesh = mesh_plot.actors[0].lineWidth(0)
# vmesh.cutWithPlane(origin=(0,0,0), normal=(1,-1,0))
# plot(vmesh, interactive=1)

# V_0 = FunctionSpace(mesh, "CG", deg)
# V_1 = FunctionSpace(mesh, "N1curl", deg)
# V_2 = FunctionSpace(mesh, "N1div", deg)
# V_3 = FunctionSpace(mesh, "DG", deg-1)


# V_0 = FunctionSpace(mesh, "P- Lambda", deg, 0)
# V_1 = FunctionSpace(mesh, "P- Lambda", deg, 1)
# V_2 = FunctionSpace(mesh, "P- Lambda", deg, 2)
# V_3 = FunctionSpace(mesh, "P- Lambda", deg, 3)

P_0 = FiniteElement("CG", tetrahedron, deg, variant='point')
P_1 = FiniteElement("N1curl", tetrahedron, deg, variant='integral')
P_2 = FiniteElement("N1div", tetrahedron, deg, variant='point')
P_3 = FiniteElement("DG", tetrahedron, deg-1, variant='point')

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

M1_petsc = PETScMatrix()
D10_petsc = PETScMatrix()

assemble(m_1, M1_petsc)
assemble(d_10, D10_petsc)

M1_mat = csr_matrix(M1_petsc.mat().getValuesCSR()[::-1])
D10_mat = csr_matrix(D10_petsc.mat().getValuesCSR()[::-1])

M1_mat.tocsc()
D10_mat.tocsc()

D_0 = spsolve(M1_mat, D10_mat)
D_0.tolil()

# D_0[abs(D_0) < tol] = 0.0
# print(D_0)
print(D_0[abs(D_0)>tol])

# Construction of the D_1 co-incidence matrix
m_2 = dot(v_2, u_2) * dx
d_21 = dot(v_2, curl(u_1)) * dx

M2_petsc = PETScMatrix()
D21_petsc = PETScMatrix()

assemble(m_2, M2_petsc)
assemble(d_21, D21_petsc)

M2_mat = csr_matrix(M2_petsc.mat().getValuesCSR()[::-1])
D21_mat = csr_matrix(D21_petsc.mat().getValuesCSR()[::-1])

M2_mat.tocsc()
D21_mat.tocsc()

D_1 = spsolve(M2_mat, D21_mat)
D_1.tolil()

# D_1[abs(D_1) < tol] = 0.0
# print(D_1)
print(D_1[abs(D_1)>tol])

# Construction of the D_3 co-incidence matrix
m_3 = dot(v_3, u_3) * dx
d_32 = dot(v_3, div(u_2)) * dx

M3_petsc = PETScMatrix()
D32_petsc = PETScMatrix()

assemble(m_3, M3_petsc)
assemble(d_32, D32_petsc)

M3_mat = csr_matrix(M3_petsc.mat().getValuesCSR()[::-1])
D32_mat = csr_matrix(D32_petsc.mat().getValuesCSR()[::-1])

M3_mat.tocsc()
D32_mat.tocsc()

D_2 = spsolve(M3_mat, D32_mat)
D_2.tolil()

# D_2[abs(D_2) < tol] = 0.0
# print(D_2)

print(D_2[abs(D_2)>tol])
