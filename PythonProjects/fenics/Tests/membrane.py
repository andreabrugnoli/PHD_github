from fenics import *

import mshr
# Set pressure function:
T = 10.0 # tension
A = 1.0
# pressure amplitude
R = 0.3
# radius of domain
theta = 0.2
x0 = 0.6*R*cos(theta)
y0 = 0.6*R*sin(theta)
sigma = 1
#sigma = 50 # large value for verification
n = 40

# approx no of elements in radial direction
domain = mshr.Circle(Point(0.,0.),1.0,n)
mesh = mshr.generate_mesh(domain, n, "cgal")
V = FunctionSpace(mesh, "Lagrange", 1)
# Define boundary condition w=0
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0.0), boundary)
# Define variational problem
w = TrialFunction(V)
v = TestFunction(V)
a = inner(nabla_grad(w), nabla_grad(v))*dx
f = Expression("4*exp(-0.5*(pow((R*x[0] - x0)/sigma, 2))-0.5*(pow((R*x[1] - y0)/sigma, 2)))", R=R, x0=x0, y0=y0, sigma=sigma, degree=2)
L = f*v*dx
# Compute solution
w = Function(V)
problem = LinearVariationalProblem(a, L, w, bc)
solver = LinearVariationalSolver(problem)
solver.parameters["linear_solver"] = "cg"
solver.parameters["preconditioner"] = "ilu"
solver.solve()

# Plot scaled solution, mesh and pressure
import matplotlib.pyplot as plt
plot(mesh, title="Mesh over scaled domain")
plot(w, title="Scaled deflection")
f = interpolate(f, V)
plot(f, title="Scaled pressure")
plt.show()

# Find maximum real deflection
max_w = w.vector().get_local().max()
max_D = A*max_w/(8*pi*sigma*T)
print("Maximum real deflection is", max_D)

viz_w = plot(w, wireframe=False, title="Scaled membrane deflection", rescale=False, axes=True, # include axes
                basename="deflection", # default plotfile name
                )

plt.show()


# Verification for "flat" pressure (large sigma)
if sigma >= 50:
    w_exact = Expression("1 - x[0]*x[0] - x[1]*x[1]", degree=2)
    w_e = interpolate(w_exact, V)

    import numpy as np
    dev = np.abs(w_e.vector().get_local() - w.vector().get_local()).max()
    print("sigma=%g: max deviation=%e"  % (sigma, dev))