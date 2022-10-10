import numpy as np

import ufl
from fenics import *
"This gives an error, no broken spaces in Fenics"
# Create mesh
n_el = 1
Lx = 1
Ly = 1
msh = RectangleMesh(Point(0, 0), Point(Lx, Ly), n_el, n_el, "right/left")

P0 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
P1 = ufl.FiniteElement("N1curl", msh.ufl_cell(), 1)
P1til = ufl.FiniteElement("RT", msh.ufl_cell(), 1)
P2 = ufl.FiniteElement("DG", msh.ufl_cell(), 1)

V0 = FunctionSpace(msh, P0)
V1 = FunctionSpace(msh, P1)
V1til = FunctionSpace(msh, P1til)
V2 = FunctionSpace(msh, P2)

P0_b = ufl.BrokenElement(P0)
P1_b = ufl.BrokenElement(P1)
P1til_b = ufl.BrokenElement(P2)
P2_b = ufl.BrokenElement(P2)

V0_b = FunctionSpace(msh, P0_b)
V1_b = FunctionSpace(msh, P1_b)
V1til_b = FunctionSpace(msh, P1til_b)
V2_b = FunctionSpace(msh, P2_b)

print(V0_b.dim())
