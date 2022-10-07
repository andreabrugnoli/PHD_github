from firedrake import *
from tools_plotting import setup
import numpy as np
np.set_printoptions(threshold=np.inf)
import os
os.environ["OMP_NUM_THREADS"] = "1"

from triangle_mesh import create_triangle
from matplotlib import pyplot as plt

L_x = 1
L_y = 1

tol = 1e-10


deg = 1
quad = False #input("quad mesh? ")


# triplot(mesh)

def curl2D(u):
    return as_vector([- u.dx(1), u.dx(0)])

def rot2D(u_vec):
    return u_vec[1].dx(0) - u_vec[0].dx(1)


if quad:
    mesh = RectangleMesh(1, 1, L_x, L_y, quadrilateral=quad)

    #
    # * 1: plane x == 0
    # * 2: plane x == Lx
    # * 3: plane y == 0
    # * 4: plane y == Ly

    P_0 = FiniteElement("CG", quadrilateral, deg)
    P_1 = FiniteElement("RTCE", quadrilateral, deg)
    P_1til = FiniteElement("RTCF", quadrilateral, deg)
    P_2 = FiniteElement("DG", quadrilateral, deg - 1)

    V_0 = FunctionSpace(mesh, P_0)
    V_1 = FunctionSpace(mesh, P_1)
    V_1til = FunctionSpace(mesh, P_1til)
    V_2 = FunctionSpace(mesh, P_2)

    # mesh_int = IntervalMesh(1, L_x)
    # mesh = ExtrudedMesh(mesh_int, 1)
    #
    #
    # CG_deg = FiniteElement("CG", interval, deg)
    # DG_deg = FiniteElement("DG", interval, deg - 1)
    #
    # P_CG1_DG = TensorProductElement(CG_deg, DG_deg)
    # P_DG_CG1 = TensorProductElement(DG_deg, CG_deg)
    #
    # RT_horiz = HDivElement(P_CG1_DG)
    # RT_vert = HDivElement(P_DG_CG1)
    # RT_quad = RT_horiz + RT_vert
    #
    # NED_horiz = HCurlElement(P_CG1_DG)
    # NED_vert = HCurlElement(P_DG_CG1)
    # NED_quad = NED_horiz + NED_vert
    #
    #
    # V_0 = FunctionSpace(mesh, "CG", deg)
    # V_1 = FunctionSpace(mesh, NED_quad)
    # V_1til = FunctionSpace(mesh, RT_quad)
    # V_2 = FunctionSpace(mesh, "DG", deg-1)

else:
    mesh = create_triangle(h=1)
    triplot(mesh)
    plt.show()
    P_0 = FiniteElement("CG", triangle, deg)
    P_1 = FiniteElement("N1curl", triangle, deg)
    P_1til = FiniteElement("RT", triangle, deg)
    P_2 = FiniteElement("DG", triangle, deg - 1)

    V_0 = FunctionSpace(mesh, P_0)
    V_1 = FunctionSpace(mesh, P_1)
    V_1til = FunctionSpace(mesh, P_1til)
    V_2 = FunctionSpace(mesh, P_2)


n_ver = FacetNormal(mesh)


dx = Measure('dx')
ds = Measure('ds')

v_0 = TestFunction(V_0)
u_0 = TrialFunction(V_0)

v_1 = TestFunction(V_1)
u_1 = TrialFunction(V_1)

v_1til = TestFunction(V_1til)
u_1til = TrialFunction(V_1til)

v_2 = TestFunction(V_2)
u_2 = TrialFunction(V_2)

f_0 = Function(V_0)
f_1 = Function(V_1)
f_1til = Function(V_1til)
f_2 = Function(V_2)

# * 1: plane x == 0
# * 2: plane x == Lx
# * 3: plane y == 0
# * 4: plane y == Ly

# b_L = v_0 * dot(u_1til, n_ver) * ds(1)
# petsc_BL = assemble(b_L, mat_type='aij').M.handle
# B_L = np.array(petsc_BL.convert("dense").getDenseArray())
# B_L[abs(B_L) < tol] = 0.0
# dofs2_L = np.where(B_L.any(axis=0))[0]
#
# print("Left boundary matrix")
# B_L = B_L[:, dofs2_L]
# print(B_L)
#
# b_R = v_0 * dot(u_1til, n_ver) * ds(2)
# petsc_BR = assemble(b_R, mat_type='aij').M.handle
# B_R = np.array(petsc_BR.convert("dense").getDenseArray())
# B_R[abs(B_R) < tol] = 0.0
# dofs2_R = np.where(B_R.any(axis=0))[0]
#
# print("Right boundary matrix")
# B_R = B_R[:, dofs2_R]
# print(B_R)


x, y = SpatialCoordinate(mesh)
g = as_vector([3*x, sin(y)])
f_1til.assign(interpolate(g, V_1til))

b_boundary_split = v_0 * dot(f_1til, n_ver) * ds(1) + v_0 * dot(f_1til, n_ver) * ds(2) \
        + v_0 * dot(f_1til, n_ver) * ds(3) + v_0 * dot(f_1til, n_ver) * ds(4)

b_boundary = v_0 * dot(f_1til, n_ver) * ds


B_bd = assemble(b_boundary).vector().get_local()
B_bd_split = assemble(b_boundary_split).vector().get_local()

print("B bd")
print(B_bd)
print("B bd split")
print(B_bd_split)


# b_D = v_0 * dot(u_1til, n_ver) * ds(3)
# petsc_BD = assemble(b_D, mat_type='aij').M.handle
# B_D = np.array(petsc_BD.convert("dense").getDenseArray())
#
# print("Lower boundary matrix")
# print(B_D)

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


