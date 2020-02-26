# Mindlin plate written with the port Hamiltonian approach

from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)
from math import floor, sqrt, pi
import matplotlib.pyplot as plt

import scipy.linalg as la


E = 2e11
nu = 0.3
h = 0.05
rho = 8000  # kg/m^3
D = E * h ** 3 / (1 - nu ** 2) / 12.

L = 10
l_x = L
l_y = L

n = 3
deg = 5


# Plate bending stiffness :math:`D=\dfrac{Eh^3}{12(1-\nu^2)}` and shear stiffness :math:`F = \kappa Gh`
# with a shear correction factor :math:`\kappa = 5/6` for a homogeneous plate
# of thickness :math:`h`::
# E = Constant(7e10)
# nu = Constant(0.3)
# h = Constant(0.2)
# rho = Constant(2000)  # kg/m^3
# k = Constant(5. / 6.)

 # Useful Matrices

D_b = as_tensor([
  [D, D * nu, 0],
  [D * nu, D, 0],
  [0, 0, D * (1 - nu) / 2]
])

fl_rot = 12. / (E * h ** 3)

C_b_vec = as_tensor([
    [fl_rot, -nu*fl_rot, 0],
    [-nu*fl_rot, fl_rot, 0],
    [0, 0, fl_rot * 2 * (1 + nu)]
])

# Vectorial Formulation possible only
def bending_moment_vec(kk):
    return dot(D_b, kk)

def bending_curv_vec(MM):
    return dot(C_b_vec, MM)

def tensor_divDiv_vec(u):
    return u[0].dx(0).dx(0) + u[1].dx(1).dx(1) + 2 * u[2].dx(0).dx(1)

def Gradgrad_vec(u):
    return as_vector([ u.dx(0).dx(0), u.dx(1).dx(1), 2 * u.dx(0).dx(1) ])


# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
L = 1
l_x = L
l_y = L
n_x, n_y = n, n
mesh = UnitSquareMesh(n_x, n_y, quadrilateral=False)

# plot(mesh)
# plt.show()


# Finite element defition

name_FEp = 'Bell'
name_FEq = 'Bell'

if name_FEp == 'Morley':
    deg_p = 2
elif name_FEp == 'Hermite':
    deg_p = 3
elif name_FEp == 'Argyris' or name_FEp == 'Bell':
    deg_p = 5

if name_FEq == 'Morley':
    deg_q = 2
elif name_FEq == 'Hermite':
    deg_q = 3
elif name_FEq == 'Argyris' or name_FEq == 'Bell':
    deg_q = 5

V_p = FunctionSpace(mesh, name_FEp, deg_p)
V_q = VectorFunctionSpace(mesh, name_FEq, deg_q, dim=3)
V = V_p * V_q

n_Vp = V.sub(0).dim()
n_Vq = V.sub(1).dim()
n_V  = V.dim()
print(n_V)

v = TestFunction(V)
v_p, v_q = split(v)

e_v = TrialFunction(V)
e_p, e_q = split(e_v)

al_p = rho * h * e_p
al_q = bending_curv_vec(e_q)

# alpha = TrialFunction(V)
# al_pw, al_pth, al_qth, al_qw = split(alpha)

# e_p = 1. / (rho * h) * al_p
# e_q = bending_moment_vec(al_q)

dx = Measure('dx')
m_p = dot(v_p, al_p) * dx
m_q = inner(v_q, al_q) * dx
m = m_p + m_q

j_divDiv = -v_p * tensor_divDiv_vec(e_q) * dx
j_divDivIP = tensor_divDiv_vec(v_q) * e_p * dx

j_gradgrad = inner(v_q, Gradgrad_vec(e_p) ) * dx
j_gradgradIP = -inner(Gradgrad_vec(v_p), e_q) * dx


j = j_divDiv + j_divDivIP #

# Assemble the stiffness matrix and the mass matrix.
J = assemble(j, mat_type='aij')
M = assemble(m, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())

tol = 10**(-6)

eigenvalues, eigvectors = la.eig(JJ, MM)
omega_all = np.imag(eigenvalues)

index = omega_all>=tol

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]


omega.sort()

n_om = 10

from math import pi

omega_tilde = np.sqrt(omega)* L * (rho*h/D)**(0.25)

for n in range(n_om):
    print(omega_tilde[n])



