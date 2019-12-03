# Mindlin plate written with the port Hamiltonian approach

from firedrake import *
import numpy as np

np.set_printoptions(threshold=np.inf)
from math import floor, sqrt, pi
import matplotlib.pyplot as plt
from firedrake.plot import _two_dimension_triangle_func_val
from scipy.io import savemat
import scipy.linalg as la

E = 2e11
nu = 0.3
h = 0.01
rho = 8000  # kg/m^3
D = E * h ** 3 / (1 - nu ** 2) / 12.

L = 1
l_x = L
l_y = L

n = 3 #int(input("N element on each side: "))

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
    [fl_rot, -nu * fl_rot, 0],
    [-nu * fl_rot, fl_rot, 0],
    [0, 0, fl_rot * 2 * (1 + nu)]
])


# Vectorial Formulation possible only
def bending_moment_vec(kk):
    return dot(D_b, kk)

def bending_curv_vec(MM):
    return dot(C_b_vec, MM)

def tensor_divDiv_vec(MM):
    return MM[0].dx(0).dx(0) + MM[1].dx(1).dx(1) + 2 * MM[2].dx(0).dx(1)

def Gradgrad_vec(u):
    return as_vector([ u.dx(0).dx(0), u.dx(1).dx(1), 2 * u.dx(0).dx(1) ])

def tensor_Div_vec(MM):
    return as_vector([ MM[0].dx(0) + MM[2].dx(1), MM[2].dx(0) + MM[1].dx(1) ])

# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
L = 1
l_x = L
l_y = L
n_x, n_y = n, n
mesh = UnitSquareMesh(n_x, n_y, quadrilateral=False)

# plot(mesh)
# plt.show()

name_FEp = 'Argyris'
name_FEq = 'DG'
deg_q = 0

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

Vp = FunctionSpace(mesh, name_FEp, deg_p)
Vq = VectorFunctionSpace(mesh, name_FEq, deg_q, dim=3)
V = Vp * Vq

n_Vp = V.sub(0).dim()
n_Vq = V.sub(1).dim()
n_V = V.dim()
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
ds = Measure('ds')
m_p = dot(v_p, al_p) * dx
m_q = inner(v_q, al_q) * dx
m = m_p + m_q

j_divDiv = -v_p * tensor_divDiv_vec(e_q) * dx
j_divDivIP = tensor_divDiv_vec(v_q) * e_p * dx

j_gradgrad = inner(v_q, Gradgrad_vec(e_p)) * dx
j_gradgradIP = -inner(Gradgrad_vec(v_p), e_q) * dx

j = j_gradgrad + j_gradgradIP  #


# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1

# bc_input = input('Select Boundary Condition:')   #'SSSS'
#
# bc_1, bc_3, bc_2, bc_4 = bc_input
#
# bc_dict = {1: bc_1, 2: bc_2, 3: bc_3, 4: bc_4}
#
# n = FacetNormal(mesh)
# # s = as_vector([-n[1], n[0]])
#
# V_qn = FunctionSpace(mesh, 'Lagrange', 1)
# V_Mnn = FunctionSpace(mesh, 'Lagrange', 1)
#
# Vu = V_qn * V_Mnn
#
# q_n, M_nn = TrialFunction(Vu)
#
# v_omn = dot(grad(v_p), n)
#
# b_vec = []
# for key,val in bc_dict.items():
#     if val == 'C':
#         b_vec.append( v_p * q_n * ds(key) + v_omn * M_nn * ds(key))
#     elif val == 'S':
#         b_vec.append(v_p * q_n * ds(key))

n = FacetNormal(mesh)

V_qn = FunctionSpace(mesh, 'DG', 0)
V_Mnn = FunctionSpace(mesh, 'DG', 0)

Vu = V_qn * V_Mnn

q_n, M_nn = TrialFunction(Vu)

v_omn = dot(grad(v_p), n)

b_u = v_p * q_n * ds + v_omn * M_nn * ds
# Assemble the stiffness matrix and the mass matrix.


J = assemble(j, mat_type='aij')
M = assemble(m, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())


if b_u:
    B_u = assemble(b_u, mat_type='aij')
    petsc_b_u = B_u.M.handle
    B_in = np.array(petsc_b_u.convert("dense").getDenseArray())
    boundary_dofs = np.where(B_in.any(axis=0))[0]  # np.where(~np.all(B_in == 0, axis=0) == True) #
    B_in = B_in[:, boundary_dofs]
else:
    B_in = np.empty((n_V, 0))

n_u = len(B_in.T)
# print(N_u)

QQ = la.inv(MM)
A_sys = JJ @ QQ
B_sys = B_in
C_sys = B_in.T @ QQ
D_sys = np.zeros((n_u, n_u))

pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/PH/PH_Control/Matrices_EB/'
A_file = 'A'; B_file = 'B'; C_file = 'C'; D_file = 'D';
savemat(pathout + A_file, mdict={A_file: np.array(A_sys)}, appendmat=True)
savemat(pathout + B_file, mdict={B_file: np.array(B_sys)}, appendmat=True)
savemat(pathout + C_file, mdict={C_file: np.array(C_sys)}, appendmat=True)
savemat(pathout + D_file, mdict={D_file: np.array(D_sys)}, appendmat=True)





