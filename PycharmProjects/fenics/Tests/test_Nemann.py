# import os
# os.environ['DOLFIN_DIR'] = "/home/a.brugnoli/anaconda3/envs/fenicsproject/share/dolfin/cmake:/home/a.brugnoli/anaconda3/pkgs/fenics-2017.2.0-np113py36_3/share/dolfin/cmake:$DOLFIN_DIR"
# os.environ['CMAKE_MODULE_PATH'] = "/home/a.brugnoli/anaconda3/envs/fenicsproject/share/dolfin/cmake:/home/a.brugnoli/anaconda3/pkgs/fenics-2017.2.0-np113py36_3/share/dolfin/cmake:$CMAKE_MODULE_PATH"
# os.environ['PETSC_DIR'] ="/home/a.brugnoli/anaconda3/include:/home/a.brugnoli/anaconda3/pkgs/petsc-3.8.4-blas_openblas_0/include:/home/a.brugnoli/anaconda3/lib:/home/a.brugnoli/anaconda3/envs/fenicsproject/include/:$PETSC_DIR"
# print(os.environ['DOLFIN_DIR'])
# print(os.environ['CMAKE_MODULE_PATH'])
# print(os.environ['PETSC_DIR'])


from fenics import *

# Create mesh and define function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Constant(0.0);

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression("2*tan(-(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2)) / 0.05)", degree=2)
g = Expression("exp(17*x[0])", degree=2)
a = inner(grad(u), grad(v))*dx
L = f*v*dx + g*v*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution and mesh
import matplotlib.pyplot as plt
plot(u)
plot(mesh)
plt.show()


# Save solution to file in VTK format
vtkfile = File('poisson/solution.pvd')
vtkfile << u
