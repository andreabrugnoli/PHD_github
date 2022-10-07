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
msh = create_rectangle(MPI.COMM_WORLD,
                       [np.array([0, 0]), np.array([1, 1])],
                       [32, 32],
                       CellType.triangle, GhostMode.none)


# Function to mark x = 0, x = 1 and y = 0
def noslip_boundary(x):
    return np.logical_or(np.logical_or(np.isclose(x[0], 0.0),
                                       np.isclose(x[0], 1.0)),
                         np.isclose(x[1], 0.0))


# Function to mark the lid (y = 1)
def lid(x):
    return np.isclose(x[1], 1.0)


# Lid velocity
def lid_velocity_expression(x):
    return np.stack((np.ones(x.shape[1]), np.zeros(x.shape[1])))

P2 = ufl.VectorElement("Lagrange", msh.ufl_cell(), 2)
P1 = ufl.FiniteElement("Lagrange", msh.ufl_cell(), 1)
V, Q = FunctionSpace(msh, P2), FunctionSpace(msh, P1)

# No-slip boundary condition for velocity field (`V`) on boundaries
# where x = 0, x = 1, and y = 0
noslip = np.zeros(msh.geometry.dim, dtype=PETSc.ScalarType)
facets = locate_entities_boundary(msh, 1, noslip_boundary)
bc0 = dirichletbc(noslip, locate_dofs_topological(V, 1, facets), V)

# Driving velocity condition u = (1, 0) on top boundary (y = 1)
lid_velocity = Function(V)
lid_velocity.interpolate(lid_velocity_expression)
facets = locate_entities_boundary(msh, 1, lid)
bc1 = dirichletbc(lid_velocity, locate_dofs_topological(V, 1, facets))

# Collect Dirichlet boundary conditions
bcs = [bc0, bc1]

# Define variational problem
(u, p) = ufl.TrialFunction(V), ufl.TrialFunction(Q)
(v, q) = ufl.TestFunction(V), ufl.TestFunction(Q)
f = Constant(msh, (PETSc.ScalarType(0), PETSc.ScalarType(0)))

a = form([[inner(grad(u), grad(v)) * dx, inner(p, div(v)) * dx],
          [inner(div(u), q) * dx, None]])
L = form([inner(f, v) * dx, inner(Constant(msh, PETSc.ScalarType(0)), q) * dx])

a_p11 = form(inner(p, q) * dx)
a_p = [[a[0][0], None],
       [None, a_p11]]

A = fem.petsc.assemble_matrix_nest(a, bcs=bcs)
A.assemble()

P11 = fem.petsc.assemble_matrix(a_p11, [])
P = PETSc.Mat().createNest([[A.getNestSubMatrix(0, 0), None], [None, P11]])
P.assemble()

b = fem.petsc.assemble_vector_nest(L)

# Modify ('lift') the RHS for Dirichlet boundary conditions
fem.petsc.apply_lifting_nest(b, a, bcs=bcs)

# Sum contributions from ghost entries on the owner
for b_sub in b.getNestSubVecs():
    b_sub.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

# Set Dirichlet boundary condition values in the RHS
bcs0 = fem.bcs_by_block(extract_function_spaces(L), bcs)
fem.petsc.set_bc_nest(b, bcs0)

# Create nullspace vector
null_vec = fem.petsc.create_vector_nest(L)

# Set velocity part to zero and the pressure part to a non-zero constant
null_vecs = null_vec.getNestSubVecs()
null_vecs[0].set(0.0), null_vecs[1].set(1.0)

# Normalize the vector, create a nullspace object, and attach it to the
# matrix
null_vec.normalize()
nsp = PETSc.NullSpace().create(vectors=[null_vec])
assert nsp.test(A)
A.setNullSpace(nsp)

ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A, P)
ksp.setType("minres")
ksp.setTolerances(rtol=1e-9)
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)

# Define the matrix blocks in the preconditioner with the velocity and
# pressure matrix index sets
nested_IS = P.getNestISs()
ksp.getPC().setFieldSplitIS(
    ("u", nested_IS[0][0]),
    ("p", nested_IS[0][1]))

# Set the preconditioners for each block
ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
ksp_u.setType("preonly")
ksp_u.getPC().setType("gamg")
ksp_p.setType("preonly")
ksp_p.getPC().setType("jacobi")

# Monitor the convergence of the KSP
ksp.setFromOptions()

u, p = Function(V), Function(Q)
x = PETSc.Vec().createNest([_cpp.la.petsc.create_vector_wrap(u.x), _cpp.la.petsc.create_vector_wrap(p.x)])
ksp.solve(b, x)

norm_u_0 = u.x.norm()
norm_p_0 = p.x.norm()
if MPI.COMM_WORLD.rank == 0:
    print("(A) Norm of velocity coefficient vector (nested, iterative): {}".format(norm_u_0))
    print("(A) Norm of pressure coefficient vector (nested, iterative): {}".format(norm_p_0))

with XDMFFile(MPI.COMM_WORLD, "out_stokes/velocity.xdmf", "w") as ufile_xdmf:
    u.x.scatter_forward()
    ufile_xdmf.write_mesh(msh)
    ufile_xdmf.write_function(u)

with XDMFFile(MPI.COMM_WORLD, "out_stokes/pressure.xdmf", "w") as pfile_xdmf:
    p.x.scatter_forward()
    pfile_xdmf.write_mesh(msh)
    pfile_xdmf.write_function(p)

A = fem.petsc.assemble_matrix_block(a, bcs=bcs)
A.assemble()
P = fem.petsc.assemble_matrix_block(a_p, bcs=bcs)
P.assemble()
b = fem.petsc.assemble_vector_block(L, a, bcs=bcs)

# Set near nullspace for pressure
null_vec = A.createVecLeft()
offset = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
null_vec.array[offset:] = 1.0
null_vec.normalize()
nsp = PETSc.NullSpace().create(vectors=[null_vec])
assert nsp.test(A)
A.setNullSpace(nsp)

# Build IndexSets for each field (global dof indices for each field)
V_map = V.dofmap.index_map
Q_map = Q.dofmap.index_map
offset_u = V_map.local_range[0] * V.dofmap.index_map_bs + Q_map.local_range[0]
offset_p = offset_u + V_map.size_local * V.dofmap.index_map_bs
is_u = PETSc.IS().createStride(V_map.size_local * V.dofmap.index_map_bs, offset_u, 1, comm=PETSc.COMM_SELF)
is_p = PETSc.IS().createStride(Q_map.size_local, offset_p, 1, comm=PETSc.COMM_SELF)

# Create Krylov solver
ksp = PETSc.KSP().create(msh.comm)
ksp.setOperators(A, P)
ksp.setTolerances(rtol=1e-9)
ksp.setType("minres")
ksp.getPC().setType("fieldsplit")
ksp.getPC().setFieldSplitType(PETSc.PC.CompositeType.ADDITIVE)
ksp.getPC().setFieldSplitIS(
    ("u", is_u),
    ("p", is_p))

# Configure velocity and pressure sub KSPs
ksp_u, ksp_p = ksp.getPC().getFieldSplitSubKSP()
ksp_u.setType("preonly")
ksp_u.getPC().setType("gamg")
ksp_p.setType("preonly")
ksp_p.getPC().setType("jacobi")

# Monitor the convergence of the KSP
opts = PETSc.Options()
opts["ksp_monitor"] = None
opts["ksp_view"] = None
ksp.setFromOptions()

# Compute solution
x = A.createVecRight()
ksp.solve(b, x)

# Create Functions and scatter x solution
u, p = Function(V), Function(Q)
offset = V_map.size_local * V.dofmap.index_map_bs
u.x.array[:offset] = x.array_r[:offset]
p.x.array[:(len(x.array_r) - offset)] = x.array_r[offset:]

norm_u_1 = u.x.norm()
norm_p_1 = p.x.norm()
if MPI.COMM_WORLD.rank == 0:
    print("(B) Norm of velocity coefficient vector (blocked, iterative): {}".format(norm_u_1))
    print("(B) Norm of pressure coefficient vector (blocked, iterative): {}".format(norm_p_1))
assert np.isclose(norm_u_1, norm_u_0)
assert np.isclose(norm_p_1, norm_p_0)