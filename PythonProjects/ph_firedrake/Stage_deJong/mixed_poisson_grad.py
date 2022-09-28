from firedrake import *

# Create mesh
N = 10

# * 1: plane x == 0
# * 2: plane x == Lx
# * 3: plane y == 0
# * 4: plane y == Ly

msh = UnitSquareMesh(N, N)

# Define finite elements spaces and build mixed space

msh = UnitSquareMesh(N, N)
V = FunctionSpace(msh, "N1curl", 1)
W = FunctionSpace(msh, "CG", 1)
Z = V*W

# Define trial and test functions
(sigma, u) = TrialFunctions(Z)
(tau, v) = TestFunctions(Z)


# Define source function
x, y = SpatialCoordinate(msh)
f = 10*exp(-((x-0.5)**2 + (y-0.5)**2)/0.02)
g = sin(5*x)


# Define variational form
a = (dot(grad(u),tau) + dot(sigma,tau) + dot(sigma,grad(v)))*dx
L = -f*v*dx - g*v*ds


# Boundaries

bc1 = DirichletBC(Z.sub(1), 0, 1)
bc2 = DirichletBC(Z.sub(1), 0, 2)
bcs = [bc1, bc2]

# Compute solution
uh = Function(Z)
solve(a==L, uh, bcs=bcs)

(sigma, u) = uh.split()

from matplotlib import pyplot as plt

quiver(sigma)

plt.show()

trisurf(u)

plt.show()

# write xdmf file
# sigma.name = "Stress"
# u.name = "Velocity"
# xdmf = io.XDMFFile(MPI.COMM_WORLD, "Dual_Weak64M1DEdge.xdmf", "w")
# xdmf.write_mesh(mesh)
# xdmf.write_function(sigma)
# xdmf.write_function(u)
# xdmf.close()