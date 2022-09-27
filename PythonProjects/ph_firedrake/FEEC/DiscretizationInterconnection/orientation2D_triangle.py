from firedrake import *
from tools_plotting import setup
import numpy as np
np.set_printoptions(threshold=np.inf)
import os
os.environ["OMP_NUM_THREADS"] = "1"

from FEEC.DiscretizationInterconnection.triangle_mesh import create_reference_triangle
from matplotlib import pyplot as plt

L_x = 1
L_y = 1

tol = 1e-10

deg = 2


def curl2D(u):
    return as_vector([- u.dx(1), u.dx(0)])

def rot2D(u_vec):
    return u_vec[1].dx(0) - u_vec[0].dx(1)

h = 1
mesh = create_reference_triangle(h)
# triplot(mesh)
# plt.show()
P_0 = FiniteElement("CG", triangle, deg)
P_1 = FiniteElement("N1curl", triangle, deg, variant='integral')
P_1til = FiniteElement("RT", triangle, deg, variant='integral')
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

b_1 = v_0 * dot(u_1til, n_ver) * ds(1)
petsc_B1 = assemble(b_1, mat_type='aij').M.handle
B_1 = np.array(petsc_B1.convert("dense").getDenseArray())
B_1[abs(B_1) < tol] = 0.0
bddofs_1 = np.where(B_1.any(axis=0))[0]

print("1 boundary matrix")
B_1 = B_1[:, bddofs_1]
print(B_1)

b_2 = v_0 * dot(u_1til, n_ver) * ds(2)
petsc_B2 = assemble(b_2, mat_type='aij').M.handle
B_2 = np.array(petsc_B2.convert("dense").getDenseArray())
B_2[abs(B_2) < tol] = 0.0
bddofs_2 = np.where(B_2.any(axis=0))[0]

print("2 boundary matrix")
B_2 = B_2[:, bddofs_2]
print(B_2)

b_3 = v_0 * dot(u_1til, n_ver) * ds(3)
petsc_B3 = assemble(b_3, mat_type='aij').M.handle
B_3 = np.array(petsc_B3.convert("dense").getDenseArray())
B_3[abs(B_3) < tol] = 0.0
bddofs_3 = np.where(B_3.any(axis=0))[0]

print("3 boundary matrix")
B_3 = B_3[:, bddofs_3]
print(B_3)


# x, y = SpatialCoordinate(mesh)
# g = as_vector([3*x, sin(y)])
# f_1til.assign(interpolate(g, V_1til))

# b_boundary_split = v_0 * dot(f_1til, n_ver) * ds(1) + v_0 * dot(f_1til, n_ver) * ds(2) \
#         + v_0 * dot(f_1til, n_ver) * ds(3)
# b_boundary = v_0 * dot(f_1til, n_ver) * ds
#
#
# B_bd = assemble(b_boundary).vector().get_local()
# B_bd_split = assemble(b_boundary_split).vector().get_local()

# print("B bd")
# print(B_bd)
# print("B bd split")
# print(B_bd_split)


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
#     ax.set_xlim(-h, 2*h)
#     ax.set_ylim(-h, 2*h)
#     quiver(f_1, axes=ax)
#
#     plt.show()


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
#
#     plt.show()



