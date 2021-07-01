from dolfin import *

# Scaled variables
l, w = 1, 0.1
mu_, lambda_ = 1, 1
rho = 10
gamma = (w/l)**2
wind = (0, 0.0, 0)

# Create mesh and define function space
mesh = BoxMesh(Point(0, 0, 0), Point(l, w, w), 50, 5, 5)
V = VectorFunctionSpace(mesh, "P", 1)

# Define boundary condition
def clamped_boundary(x, on_boundary):
    return on_boundary and (near(x[0], 0) or near(x[0], l))
bc = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)

# Define strain and stress
def epsilon(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
    return lambda_ * nabla_grad(u) * Identity(3) + 2 * mu_ * epsilon(u)


# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant((0, 0, -rho * gamma))
T = Constant(wind)
a = inner(sigma(u), epsilon(v)) * dx
L = dot(f, v) * dx + dot(T, v) * ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

################################ Plot solution
from vedo.dolfin import plot

plt = plot(u,
           mode="displaced mesh",
           lighting='plastic',
           axes=1,
           viewup='z',
           interactive=0)

vmesh = plt.actors[0].lineWidth(0)
vmesh.cutWithPlane(origin=(.5,0,0), normal=(1,2,1))
plot(vmesh, interactive=1)