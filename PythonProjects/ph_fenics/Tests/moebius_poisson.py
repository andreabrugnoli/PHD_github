from dolfin import *

import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from tools_plotting import setup

import os

# Input mesh
mesh = Mesh ("moebius.xml.gz")

# ax = plt.axes(projection='3d')
# mesh_plot = plot(mesh, axes=ax) # mode="mesh", interactive=0
#
#
# plt.show()

# Define and solve problem as usual on this mesh
V = FunctionSpace(mesh , "Lagrange", 1)
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v))*dx
f = Constant(1.0)
L = f*v*dx
u = Function(V)
bc = DirichletBC(V ,0.0, " on_boundary ")
solve(a == L, u, bc)


# plt.figure()
# c = plot(u)
# plt.colorbar(c)
# plot(u)
#
# plt.savefig("/home/andrea/" + "u.eps", format="eps")
#
plot(u)

plt.show()