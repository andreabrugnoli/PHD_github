from firedrake import *
from tools_plotting import setup
import numpy as np
np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt

L_x = 1
L_y = 2

deg = 1

mesh = RectangleMesh(1, 1, L_x, L_y, quadrilateral=True)

# triplot(mesh)

def curl2D(u):
    return as_vector([- u.dx(1), u.dx(0)])

def rot2D(u_vec):
    return u_vec[1].dx(0) - u_vec[0].dx(1)


P_0 = FiniteElement("CG", quadrilateral, deg)
P_1 = FiniteElement("RTCE", quadrilateral, deg)
P_1til = FiniteElement("RTCF", quadrilateral, deg)
P_2 = FiniteElement("DG", quadrilateral, deg-1)

V_0 = FunctionSpace(mesh, P_0)
V_1 = FunctionSpace(mesh, P_1)
V_1til = FunctionSpace(mesh, P_1til)
V_2 = FunctionSpace(mesh, P_2)


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

n_1 = V_1.dim()
for i in range(n_1):
    zeros_n1 = np.zeros((n_1,))
    zeros_n1[i] = 1

    f_1.vector().set_local(zeros_n1)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    quiver(f_1, axes=ax)

n_1til = V_1til.dim()
for i in range(n_1til):
    zeros_n1til = np.zeros((n_1til,))
    zeros_n1til[i] = 1

    f_1til.vector().set_local(zeros_n1til)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    quiver(f_1til, axes=ax)

plt.show()


