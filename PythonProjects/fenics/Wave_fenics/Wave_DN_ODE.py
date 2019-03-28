from fenics import *
import numpy as np
np.set_printoptions(threshold=np.inf)
from math import pi
import mshr
import matplotlib.pyplot as plt

import scipy.linalg as la

# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
L = 1
n = 10

mesh = IntervalMesh(n, 0, L)

# Domain, Subdomains, Boundary, Suboundaries
tol = 1E-14


class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0, tol)


class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], L, tol)


kk = 3
gamma = kk/n


class Omega_L(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= gamma*L


class Omega_R(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] > gamma*L


# Initialize mesh function for boundary domains

domains = MeshFunction("size_t", mesh, mesh.topology().dim())
domains.set_all(2)
omega_L = Omega_L()
omega_R = Omega_R()
omega_L.mark(domains, 0)
omega_R.mark(domains, 1)

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(2)

left = Left()
right = Right()
left.mark(boundaries,  0)
right.mark(boundaries, 1)


dx = Measure('dx', domain=mesh, subdomain_data=domains)
ds = Measure('ds', domain=mesh, subdomain_data=boundaries)
n = FacetNormal(mesh)

deg_p = 2
deg_q = 2

# Finite element defition
P_p = FiniteElement('P', mesh.ufl_cell(), deg_p)
P_q = FiniteElement('P', mesh.ufl_cell(), deg_q)
V = FunctionSpace(mesh, MixedElement([P_p, P_q]))

# bcs = [bc_eq_l, bc_eq_u]

alpha = TrialFunction(V)
al_p, al_q = split(alpha)

v = TestFunction(V)
v_p, v_q = split(v)

# Initial Conditions

# Forms and corresponding matrices
m = dot(v, alpha)*dx


j_div = v_p*al_q.dx(0)*dx(0)
j_divIP = -v_q.dx(0)*al_p*dx(0)

j_grad = v_q*al_p.dx(0)*dx(1)
j_gradIP = -v_p.dx(0)*al_q*dx(1)

j = j_grad + j_gradIP + j_div + j_divIP



# Assemble the stiffness matrix and the mass matrix.
J, M = PETScMatrix(), PETScMatrix()
b = PETScVector()

f = inner(Constant(1), v_p)*dx

assemble_system(j, f, A_tensor=J, b_tensor=b)
assemble_system(m, f, A_tensor=M, b_tensor=b)


eigenvalues, eigvectors = la.eig(J.array(), M.array())
omega_all = np.imag(eigenvalues)

index = omega_all > 0

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()
print('Numerical and Exact eigenvalues')
for i in range(4):

    omega_ex = (2*i+1)/2*pi
    print(omega[i], omega_ex)
