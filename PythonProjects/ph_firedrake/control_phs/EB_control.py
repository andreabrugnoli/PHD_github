# Mindlin plate written with the port Hamiltonian approach

from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)
from math import floor, sqrt, pi
import matplotlib.pyplot as plt
import scipy.linalg as la
from modules_ph.classes_phsystem import SysPhdaeRig
from control import lqr, ctrb, obsv

# E = 2e11
# rho = 7900  # kg/m^3

# nu = 0.3
#
# b = 0.05
# h = 0.01
# A = b * h
#
# I = 1./12 * b * h**3
#
# EI = E * I
# L = 1

L =0.3
rho = 0.0643
A = 1
EI = 37.0116
n = 5
deg = 3

mesh = IntervalMesh(n, L)

# plot(mesh)
# plt.show()


# Finite element defition

V_p = FunctionSpace(mesh, "Hermite", deg)
V_q = FunctionSpace(mesh, "DG", 1)

V = V_p * V_q

n_V = V.dim()
print(n_V)

n_p = V_p.dim()
n_q = V_q.dim()

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
m_form =  m_p + m_q

j_gradgrad = v_q * e_p.dx(0).dx(0) * dx
j_gradgradIP = -v_p.dx(0).dx(0) * e_q * dx


j_form = j_gradgrad + j_gradgradIP

J = assemble(j_form, mat_type='aij')
M = assemble(m_form, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())

b_F = v_p * ds(2)
b_M = v_p.dx(0) * ds(2)

B_F = assemble(b_F).vector().get_local()
B_M = assemble(b_M).vector().get_local()

B = np.zeros((n_V, 2))
B[:, 0] = B_F
B[:, 1] = B_M

gCF_Hess = [- v_p * ds(1), + v_p.dx(0) * ds(1)]

g_l = gCF_Hess
g_r = []

G_L = np.zeros((n_V, len(g_l)))
G_R = np.zeros((n_V, len(g_r)))


for counter, item in enumerate(g_l):
    G_L[:, counter] = assemble(item).vector().get_local()

for counter, item in enumerate(g_r):
    G_R[:, counter] = assemble(item).vector().get_local()

G = np.concatenate((G_L, G_R), axis=1)

n_lmb = G.shape[1]

# print(G)
# print(n_lmb)

# n_lmb = 2
# G = np.zeros((n_V, n_lmb))
# G[:2, :2] = np.eye(2)

G_ortho = la.null_space(G.T).T

Z_lmb = np.zeros((n_lmb, n_lmb))
J_aug = np.vstack([np.hstack([JJ, G]),
                   np.hstack([-G.T, Z_lmb])
                   ])

M_aug = la.block_diag(MM, Z_lmb)

B_aug = np.concatenate((B, np.zeros((n_lmb, B.shape[1]))))

beam = SysPhdaeRig(n_V + n_lmb, n_lmb, 0, n_p, n_q, E=M_aug, J=J_aug, B=B_aug)

beam_ode, T = beam.dae_to_odeCE()[:2]

J_sys = beam_ode.J
Q_sys = beam_ode.Q
B_sys = beam_ode.B

A_sys = J_sys @ Q_sys
C_sys = B_sys.T @ Q_sys

Cmat = ctrb(A_sys, B_sys)
Omat = obsv(A_sys, C_sys)

tol_r = 1e-40
rank_C = np.linalg.matrix_rank(Cmat)
rank_O = np.linalg.matrix_rank(Omat)

print(Cmat.shape, Omat.shape)
print(rank_C, n_V)
print(rank_O, n_V)

u, s, v = np.linalg.svd(Omat)
print(s)