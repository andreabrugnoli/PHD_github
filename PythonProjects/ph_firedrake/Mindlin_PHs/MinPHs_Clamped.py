# Mindlin plate written with the port Hamiltonian approach


from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)

import scipy.linalg as la
import matplotlib.pyplot as plt
from petsc4py import PETSc

#parameters["matnest"] = False

E = 1 #(7e10)
nu = (0.3)
h = (0.01)
rho = 1  #(2000)  # kg/m^3
k =  0.8601 # 5./6. #
L = 1

nreq = 30
n = 5
deg = 2

"""
Compute the eigenvalues of the Laplacian with Dirichlet boundary conditions on the square.
Use a mesh of n x n squares, divided into triangles, Lagrange elements of degree deg, and
request nreq eigenpairs. If export_eigenfunctions=True, write the eigenfunctions to
PVD files. Return values are the number of converged eigenpairs and the computed eigenvalues.
"""

# Plate bending stiffness :math:`D=\dfrac{Eh^3}{12(1-\nu^2)}` and shear stiffness :math:`F = \kappa Gh`
# with a shear correction factor :math:`\kappa = 5/6` for a homogeneous plate
# of thickness :math:`h`::
# E = Constant(7e10)
# nu = Constant(0.3)
# h = Constant(0.2)
# rho = Constant(2000)  # kg/m^3
# k = Constant(5. / 6.)

D = E * h ** 3 / (1 - nu ** 2) / 12.
G = E / 2 / (1 + nu)
F = G * h * k

# I_w = 1. / (rho * h)
I_phi = 12. / (rho * h ** 3)

# Useful Matrices

D_b = as_tensor([
  [D, D * nu, 0],
  [D * nu, D, 0],
  [0, 0, D * (1 - nu) / 2]
])

fl_rot = 12. / (E * h ** 3)
C_b = as_tensor([
    [fl_rot, -nu*fl_rot, 0],
    [-nu*fl_rot, fl_rot, 0],
    [0, 0, fl_rot * (1 + nu)/2]
])

C_b_vec = as_tensor([
    [fl_rot, -nu*fl_rot, 0],
    [-nu*fl_rot, fl_rot, 0],
    [0, 0, fl_rot * 2 * (1 + nu)]
])

# # Operators and functions for tensorial formulation

# def gradSym(u):
#   #return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
#   return sym(nabla_grad(u))
#
# def strain2voigt(eps):
#   return as_vector([eps[0, 0], eps[1, 1], 2 * eps[0, 1]])
#
# def voigt2stress(S):
#   return as_tensor([[S[0], S[2]], [S[2], S[1]]])
#
# def bending_moment(u):
#   return voigt2stress(dot(D_b, strain2voigt(u)))
#
#
# def bending_curv(u):
#   return voigt2stress(dot(C_b, strain2voigt(u)))

# Operators for vectorial formulation

def bending_moment_vec(kk):
    return dot(D_b, kk)

def bending_curv_vec(MM):
    return dot(C_b_vec, MM)

def tensor_div_vec(u):
    return as_vector([ u[0].dx(0) + u[2].dx(1), u[1].dx(1) + u[2].dx(0)])

def gradSym_vec(u):
    return as_vector([ u[0].dx(0),  u[1].dx(1), u[0].dx(1) + u[1].dx(0)])


# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
L = 1
l_x = L
l_y = L
n_x, n_y = n, n
mesh = UnitSquareMesh(n_x, n_y)

# plot(mesh)
# plt.show()

# Finite element defition

V_pw = FunctionSpace(mesh, "CG", deg)
V_pth = VectorFunctionSpace(mesh, "CG", deg)
V_qth = VectorFunctionSpace(mesh, "CG", deg, dim = 3)
V_qw = VectorFunctionSpace(mesh, "CG", deg)

print(V_qth.dim())

V = V_pw * V_pth * V_qth * V_qw   # MixedFunctionSpace([V_pw, V_pth, V_qth, V_qw])

v = TestFunction(V)
v_pw, v_pth, v_qth, v_qw = split(v)
#
# alpha = TrialFunction(V)
# al_pw, al_pth, al_qth, al_qw = split(alpha)
#
# e_pw = 1. / (rho * h) * al_pw
# e_pth = I_phi * al_pth
# e_qth = bending_moment_vec(al_qth)
# e_qw = F * al_qw

e_v = TrialFunction(V)
e_pw, e_pth, e_qth, e_qw = split(e_v)

al_pw = rho*h*e_pw
al_pth = (rho*h**3)/12. * e_pth
al_qth = bending_curv_vec(e_qth)
al_qw = 1./F *e_qw


## m = inner(v, alpha) * dx
m = inner(v_pw, al_pw) * dx + inner(v_pth, al_pth) * dx + inner(v_qth, al_qth) * dx + inner(v_qw, al_qw) * dx

# # For tensor notation
# D_div = v_pw * div(e_qw) * dx
# D_divIP = -div(v_qw) * e_pw * dx
#
# D_divSym = dot(v_pth, div(e_qth)) * dx
# D_divSymIP = -dot(div(v_qth), e_pth) * dx
#
# D_grad = dot(v_qw, grad(e_pw)) * dx
# D_gradIP = -dot(grad(v_pw), e_qw) * dx
#
# D_gradSym = inner(v_qth, gradSym(e_pth)) * dx
# D_gradSymIP = -inner(gradSym(v_pth), e_qth) * dx
#
# D_Id = dot(v_pth, e_qw) * dx
# D_IdIP = -dot(v_qw, e_pth) * dx

# For vector notation
D_div = v_pw * div(e_qw) * dx
D_divIP = -div(v_qw) * e_pw * dx

D_divSym = dot(v_pth, tensor_div_vec(e_qth)) * dx
D_divSymIP = -dot(tensor_div_vec(v_qth), e_pth) * dx

D_grad = dot(v_qw, grad(e_pw)) * dx
D_gradIP = -dot(grad(v_pw), e_qw) * dx

D_gradSym = inner(v_qth, gradSym_vec(e_pth)) * dx
D_gradSymIP = -inner(gradSym_vec(v_pth), e_qth) * dx

D_Id = dot(v_pth, e_qw) * dx
D_IdIP = -dot(v_qw, e_pth) * dx


# j_red = D_Id + D_div
# j_com = D_divIP + D_divSym + D_divSymIP + D_IdIP
j = D_div + D_divIP + D_divSym + D_divSymIP + D_Id + D_IdIP
# j = D_grad + D_gradIP + D_gradSym + D_gradSymIP + D_Id + D_IdIP
#
#
bc_w = DirichletBC(V.sub(0), Constant(0.0), "on_boundary")
bc_th = DirichletBC(V.sub(1), (Constant(0.0), Constant(0.0)), "on_boundary")

bcs = [bc_w, bc_th]
# Assemble the stiffness matrix and the mass matrix.

#parameters["matnest"] = False

J = assemble(j, bcs = bcs, mat_type= "aij")
M = assemble(m, bcs = bcs, mat_type= "aij")

petsc_j = J.M.handle
petsc_m = M.M.handle


JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())

eigenvalues, vr = la.eig(JJ, MM)

tol = 10**(-3)


omega_all = np.imag(eigenvalues)

index = omega_all > tol

omega = omega_all[index]
omega.sort()

# vr_omega = vr[:, index]
# perm = np.argsort(omega)
# vr_omega = vr_omega[:, perm]
#

nconv = len(omega)

omega_tilde = np.zeros((len(omega),1))
for i in range(len(omega)):
    omega_tilde[i] = omega[i]*L*((2*(1+nu)*rho)/E)**0.5

print("Smallest positive normalized eigenvalues computed: ")
for i in range(min(nconv, nreq)):
    print('Omega tilde num ', i+1,': ', omega_tilde[i])



