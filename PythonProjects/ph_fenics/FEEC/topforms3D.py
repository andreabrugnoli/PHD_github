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

u_3 = interpolate(Constant(1), V_3)

print(u_3.vector().get_local())