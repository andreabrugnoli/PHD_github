from firedrake import *
from tools_plotting import setup
import numpy as np
np.set_printoptions(threshold=np.inf)


import os
os.environ["OMP_NUM_THREADS"] = "1"

tol = 1e-10

import matplotlib.pyplot as plt

n_el = 1
L_x = 1
L_y = 1
L_z = 1

deg = 1

quad = input("quad mesh? ")

if quad:
    mesh_quad = RectangleMesh(1, 1, L_x, L_y, quadrilateral=quad)
    #
    # * 1: plane x == 0
    # * 2: plane x == Lx
    # * 3: plane y == 0
    # * 4: plane y == Ly
    mesh = ExtrudedMesh(mesh_quad, n_el)

    # ds_v side facets of the mesh. This can be combined
    # with boundary markers from the base mesh, such as ds_v(1).

    # ds_t top surface of the mesh.

    # ds_b bottom surface of the mesh.

    # ds_tb both the top and bottom surfaces mesh.

    P_0 = FiniteElement("CG", hexahedron, deg)
    P_1 = FiniteElement("NCE", hexahedron, deg)
    P_2 = FiniteElement("NCF", hexahedron, deg)
    P_3 = FiniteElement("DG", hexahedron, deg - 1)

else:
    mesh = BoxMesh(n_el, n_el, n_el, L_x, L_y, L_z)

    # * 1: plane x == 0
    # * 2: plane x == 1
    # * 3: plane y == 0
    # * 4: plane y == 1
    # * 5: plane z == 0
    # * 6: plane z == 1

    P_0 = FiniteElement("CG", triangle, deg)
    P_1 = FiniteElement("N1curl", triangle, deg)
    P_2 = FiniteElement("RT", triangle, deg)
    P_3 = FiniteElement("DG", triangle, deg - 1)

n_ver = FacetNormal(mesh)

V_0 = FunctionSpace(mesh, P_0)
V_1 = FunctionSpace(mesh, P_1)
V_2 = FunctionSpace(mesh, P_2)
V_3 = FunctionSpace(mesh, P_3)

dx = Measure('dx')
ds = Measure('ds')

v_0 = TestFunction(V_0)
u_0 = TrialFunction(V_0)

v_1 = TestFunction(V_1)
u_1 = TrialFunction(V_1)

v_2 = TestFunction(V_2)
u_2 = TrialFunction(V_2)

v_3 = TestFunction(V_3)
u_3 = TrialFunction(V_3)

f_0 = Function(V_0)
f_1 = Function(V_1)
f_2 = Function(V_2)
f_3 = Function(V_3)

# * 1: plane x == 0
# * 2: plane x == Lx
# * 3: plane y == 0
# * 4: plane y == Ly
if quad:
    b_L = v_0 * dot(u_2, n_ver) * ds_v(1)
    b_R = v_0 * dot(u_2, n_ver) * ds_v(2)
else:
    b_L = v_0 * dot(u_2, n_ver) * ds(1)
    b_R = v_0 * dot(u_2, n_ver) * ds(2)

petsc_BL = assemble(b_L, mat_type='aij').M.handle
B_L = np.array(petsc_BL.convert("dense").getDenseArray())
B_L[abs(B_L) < tol] = 0.0

dofs2_L = np.where(B_L.any(axis=0))[0]
print("Left boundary matrix")
# print(B_L)
B_L = B_L[:, dofs2_L]
print(B_L)

petsc_BR = assemble(b_R, mat_type='aij').M.handle
B_R = np.array(petsc_BR.convert("dense").getDenseArray())
B_R[abs(B_R) < tol] = 0.0

dofs2_R = np.where(B_R.any(axis=0))[0]
print("Right boundary matrix")
# print(B_R)
B_R = B_R[:, dofs2_R]
print(B_R)
print(B_R.shape, B_L.shape)

# n_0 = V_0.dim()
# for i in range(n_0):
#     zeros_n0 = np.zeros((n_0,))
#     zeros_n0[i] =1
#
#     f_0.vector().set_local(zeros_n0)
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     trisurf(f_0, axes=ax)
# plt.show()

# n_1 = V_1.dim()
# for i in range(n_1):
#     zeros_n1 = np.zeros((n_1,))
#     zeros_n1[i] = 1
#
#     f_1.vector().set_local(zeros_n1)
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     quiver(f_1, axes=ax)
#
#

# n_1til = V_1til.dim()
# for i in range(n_1til):
#     zeros_n1til = np.zeros((n_1til,))
#     zeros_n1til[i] = 1
#
#     f_1til.vector().set_local(zeros_n1til)
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     quiver(f_1til, axes=ax)