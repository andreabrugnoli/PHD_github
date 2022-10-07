import numpy as np

import ufl
from dolfinx import cpp as _cpp
from dolfinx import fem
from dolfinx.fem import (Constant, Function, FunctionSpace, dirichletbc,
                         extract_function_spaces, form,
                         locate_dofs_geometrical, locate_dofs_topological)
from dolfinx.io import XDMFFile
from dolfinx.mesh import (CellType, GhostMode, create_rectangle,
                          locate_entities_boundary)
from ufl import div, dx, grad, inner

from mpi4py import MPI
from petsc4py import PETSc

# Create mesh
n_el = 1
L_x = 1
L_y = 1
msh = create_rectangle(MPI.COMM_WORLD,
                       [np.array([0, 0]), np.array([L_x, L_y])],
                       [n_el, n_el],
                       CellType.triangle, GhostMode.none)


# Function to mark x = 0, x = 1 and y = 0
def left_boundary(x):
    return np.isclose(x[0], 0.0)


# Function to mark the lid (y = 1)
def right_boundary(x):
    return np.isclose(x[0], L_x)


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


# No-slip boundary condition for velocity field (`V`) on boundaries
# where x = 0, x = 1, and y = 0
noslip = np.zeros(msh.geometry.dim, dtype=PETSc.ScalarType)
facets_left = locate_entities_boundary(msh, 1, left_boundary)
bc0 = dirichletbc(noslip, locate_dofs_topological(V0, 1, facets_left), V0)

