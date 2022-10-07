
from fenics import *
import numpy as np

# Create mesh and define function space
mesh = UnitSquareMesh(2, 2)
V = FunctionSpace(mesh, "Lagrange", 1)

# Define Dirichlet boundary (x = 0 or x = 1)
def boundary(x):
    return x[0] < DOLFIN_EPS or x[0] > 1.0 - DOLFIN_EPS

# Define boundary condition
u0 = Constant(2.0)
bc = DirichletBC(V, u0, boundary)

boundary_dofs = sorted(bc.get_boundary_values().keys())

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("10*exp(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.02)", degree = 2)
g = Expression("sin(5*x[0])", degree = 2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

A = PETScMatrix()
b = PETScVector()

print("Nodes at the boundary")
print(boundary_dofs)

A= assemble(a)
print("A without bc:")
print(A.array())

bc.apply(A)
print("A with bc")
print(A.array())

b= assemble(L)
print("b without bc:")
print(np.transpose(b.get_local()))

bc.apply(b)
print("b with bc")
print(np.transpose(b.get_local()))

print("If assemble system is used:")
assemble_system(a, L, bcs = bc, A_tensor=A, b_tensor= b)
print("Matrix A:")
print(A.array())

print("Vector b:")
print(np.transpose(b.get_local()))

# # Compute solution
# u = Function(V)
# solve(a == L, u, bc)

# # Save solution in VTK format
# file = File("poisson.pvd")
# file << u

# # Plot solution
# plot(u)
# plt.show()