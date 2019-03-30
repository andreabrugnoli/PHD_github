# EB beam written with the port Hamiltonian approach

from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)

import sys
sys.path.append('/home/a.brugnoli/GitProjects/PythonProjects/Reduction_Index2DAE')

from module_reduction import proj_matrices
from math import floor, sqrt, pi
import matplotlib.pyplot as plt

import scipy.linalg as la
from scipy.io import savemat


E = 2e9
rho = 2700  # kg/m^3
nu = 0.3

b = 0.05
h = 0.01
A = b * h

I = 1./12 * b * h**3

EI = E * I
L = 1


n = 30
deg = 3


# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
L = 1

mesh = IntervalMesh(n, L)

# plot(mesh)
# plt.show()


# Finite element defition

V_p = FunctionSpace(mesh, "Hermite", deg)
V_q = FunctionSpace(mesh, "Hermite", deg)

V = V_p * V_q

n_V = V.dim()
n_Vp = V_p.dim()
n_Vq = V_q.dim()

v = TestFunction(V)
v_p, v_q = split(v)

e_v = TrialFunction(V)
e_p, e_q = split(e_v)

al_p = rho * A * e_p
al_q = 1./EI * e_q

dx = Measure('dx')
ds = Measure('ds')
m_p = v_p * al_p * dx
m_q = v_q * al_q * dx
m =  m_p + m_q

j_divDiv = -v_p * e_q.dx(0).dx(0) * dx
j_divDivIP = v_q.dx(0).dx(0) * e_p * dx

j_gradgrad = v_q * e_p.dx(0).dx(0) * dx
j_gradgradIP = -v_p.dx(0).dx(0) * e_q * dx

j = j_gradgrad + j_gradgradIP
# j = j_divDiv + j_divDivIP

# bc_w = DirichletBC(V.sub(0), Constant(0.0), 1)
# bc_M = DirichletBC(V.sub(0), Constant(0.0), 2)
# boundary_dofs = sorted(bc_w.nodes)

gCC_Hess = [- v_p * ds(1), + v_p.dx(0) * ds(1), - v_p * ds(2), v_p.dx(0) * ds(2)]
gSS_Hess = [- v_p * ds(1), - v_p * ds(2)]

gCF_Hess = [- v_p * ds(1), + v_p.dx(0) * ds(1)]

gFF_divDiv = [v_q * ds(1), - v_q.dx(0) * ds(1), + v_q * ds(2), - v_q.dx(0) * ds(2)]
gSS_divDiv = [v_q * ds(1), v_q * ds(2)]

gCF_divDiv = [+ v_q * ds(2), - v_q.dx(0) * ds(2)]

g_l = gCF_Hess
g_r = []  # gCF_divDiv

G_L = np.zeros((n_V, len(g_l)))
G_R = np.zeros((n_V, len(g_r)))



for counter, item in enumerate(g_l):
    G_L[:, counter] = assemble(item).vector().get_local()

for counter, item in enumerate(g_r):
    G_R[:, counter] = assemble(item).vector().get_local()

G = np.concatenate((G_L, G_R), axis=1)

x = SpatialCoordinate(mesh)
a = 3*L/4
b = L
ctr_loc = conditional(And(ge(x[0], a), le(x[0], b)), 1, 0)
b_p =  v_p.dx(0) * ds(2) # v_p * ctr_loc * dx

# Assemble the stiffness matrix and the mass matrix.
Bp = assemble(b_p)

petsc_j = assemble(j, mat_type='aij').M.handle
petsc_m = assemble(m, mat_type='aij').M.handle

J = np.array(petsc_j.convert("dense").getDenseArray())
M = np.array(petsc_m.convert("dense").getDenseArray())

n_lmb = len(G.T)

Z_lmb = np.zeros((n_lmb, n_lmb))

J_full = np.vstack([ np.hstack([J, G]),
                    np.hstack([-G.T, Z_lmb])
                ])

Z_el = np.zeros((n_V, n_lmb))

E_full = np.vstack([ np.hstack([M,       Z_el]),
                    np.hstack([Z_el.T, Z_lmb])
                 ])

B_full = np.zeros((len(E_full), 1))
B_full[:n_V] = Bp.vector().get_local().reshape((-1, 1))


# Reduction projection matrices
A_full = -J_full

n_red = 20
s0 = 0.00001

tol = 1e-16
V1, V2 = proj_matrices(E_full, A_full, B_full, s0, n_red, n_Vp, n_V, tol)

# V_red = np.vstack((V1, V2))

M1 = E_full[:n_Vp, :n_Vp]
M2 = E_full[n_Vp:n_V, n_Vp:n_V]

G = J_full[n_Vp:n_V, :n_Vp]
N = J_full[n_V:, :n_Vp]


M1_red = V1.T @ M1 @ V1
M2_red = V2.T @ M2 @ V2

n1_red = len(M1_red)
n2_red = len(M1_red) + len(M2_red)

E_red = la.block_diag(M1_red, M2_red, Z_lmb)

G_red = V2.T @ G @ V1
N_red = N @ V1

J_red = np.zeros(E_red.shape)
J_red[n1_red:, :n1_red] = np.concatenate((G_red, N_red), axis=0)
J_red[:n1_red, n1_red:] = -np.concatenate((G_red, N_red)).T

B1_red = V1.T @ B_full[:n_Vp]

B_red = np.zeros((len(E_red), 1))
B_red[:n1_red] = B1_red


UT_N, S_N, V_N = np.linalg.svd(N.T, full_matrices=True)
VT_N = V_N.T

S_N = np.diag(S_N)

U = la.block_diag(UT_N, np.eye((n_Vq, n_Vq)), VT_N)

E_til = U.T @ E_full @ U
J_til = U.T @ J_full @ U
B_til = U.T @ B_full

E_til = E_til[n_lmb:, n_lmb:]; J_til = J_til[n_lmb:, n_lmb:]; B_til = B_til[n_lmb:]
pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/ReductionPHDAEind2/'

E_file = 'E'; J_file = 'J'; B_file = 'B'
savemat(pathout + E_file, mdict={E_file: E_full})
savemat(pathout + J_file, mdict={J_file: J_full})
savemat(pathout + B_file, mdict={B_file: B_full})

E_file = 'Etil'; J_file = 'Jtil'; B_file = 'Btil'
savemat(pathout + E_file, mdict={E_file: E_til})
savemat(pathout + J_file, mdict={J_file: J_til})
savemat(pathout + B_file, mdict={B_file: B_til})
#
# Er_file = 'Er2'; Jr_file = 'Jr2'; Br_file = 'Br2'
# savemat(pathout + Er_file, mdict={Er_file: E_red})
# savemat(pathout + Jr_file, mdict={Jr_file: J_red})
# savemat(pathout + Br_file, mdict={Br_file: B_red})

# tol = 1e-9
# eigenvalues, eigvectors = la.eig(J_full, E_full)
# omega_all = np.imag(eigenvalues)
#
# index = omega_all > 0#  tol
#
# omega = omega_all[index]
# eigvec_omega = eigvectors[:, index]
# perm = np.argsort(omega)
# eigvec_omega = eigvec_omega[:, perm]
#
# omega.sort()
#
# eigenvalues_r, eigvectors_r = la.eig(J_red, E_red)
# omega_all_r = np.imag(eigenvalues_r)
#
# index_r = omega_all_r > 0
#
# omega_r = omega_all_r[index_r]
# eigvec_omega_r = eigvectors_r[:, index_r]
# perm_r = np.argsort(omega_r)
# eigvec_omega_r = eigvec_omega[:, perm_r]
#
# omega_r.sort()
#
# k_n = omega**(0.5)*L*(rho*A/(EI))**(0.25)
# k_n_r = omega_r**(0.5)*L*(rho*A/(EI))**(0.25)
# print("Smallest positive normalized eigenvalues computed: ")
# for i in range(len(omega_r)):
#    print(k_n[i], k_n_r[i])
#
#
# print(E_red.shape)
# plt.plot(np.real(eigenvalues), np.imag(eigenvalues), 'o')
# plt.plot(np.real(eigenvalues_r), np.imag(eigenvalues_r), '+')
# plt.show()