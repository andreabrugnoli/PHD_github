
from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)

import scipy.linalg as la
from scipy import signal

from scipy.io import savemat

import matplotlib
import matplotlib.pyplot as plt

plt.close('all')
matplotlib.rcParams['text.usetex'] = True

E = 2e11
rho = 7900  # kg/m^3
nu = 0.3

b = 0.05
h = 0.01
A = b * h

I = 1./12 * b * h**3

EI = E * I
L = 1

n = 2
deg = 3


mesh = IntervalMesh(n, L)

# plot(mesh)
# plt.show()

# Finite element defition

V_p = FunctionSpace(mesh, "Hermite", deg)
V_q = FunctionSpace(mesh, "Hermite", deg)

V = V_p * V_q

n_V = V.dim()
n_p = V_p.dim()

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
m = m_p + m_q

j_divDiv = -v_p * e_q.dx(0).dx(0) * dx
j_divDivIP = v_q.dx(0).dx(0) * e_p * dx

j_gradgrad = v_q * e_p.dx(0).dx(0) * dx
j_gradgradIP = -v_p.dx(0).dx(0) * e_q * dx

j = j_gradgrad + j_gradgradIP
# j = j_divDiv + j_divDivIP

gCC_Hess = [- v_p * ds(1), + v_p.dx(0) * ds(1), - v_p * ds(2), v_p.dx(0) * ds(2)]
gSS_Hess = [- v_p * ds(1), - v_p * ds(2)]

gCF_Hess = [- v_p * ds(1), + v_p.dx(0) * ds(1)]

gFF_divDiv = [v_q * ds(1), - v_q.dx(0) * ds(1), + v_q * ds(2), - v_q.dx(0) * ds(2)]
gSS_divDiv = [v_q * ds(1), v_q * ds(2)]

gCF_divDiv = [+ v_q * ds(2), - v_q.dx(0) * ds(2)]

g_l = gCF_Hess
g_r = [] # gCF_divDiv

G_L = np.zeros((n_V, len(g_l)))
G_R = np.zeros((n_V, len(g_r)))

for counter, item in enumerate(g_l):
    G_L[:, counter] = assemble(item).vector().get_local()

for counter, item in enumerate(g_r):
    G_R[:, counter] = assemble(item).vector().get_local()

G_lmb = np.concatenate((G_L, G_R), axis=1)

# Assemble the stiffness matrix and the mass matrix.
J = assemble(j, mat_type='aij')
M = assemble(m, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle


b_F = v_p * ds(2)
b_M = -v_p.dx(0) * ds(2)
B_F = assemble(b_F).vector().get_local()

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())

n_lmb = len(G_lmb.T)
n_tot = n_V + n_lmb

Z_u = np.zeros((n_lmb, n_lmb))


J_aug = np.vstack([ np.hstack([JJ, G_lmb]),
                    np.hstack([-G_lmb.T, Z_u])
                ])

Z_al_u = np.zeros((n_V, n_lmb))
Z_u_al = np.zeros((n_lmb, n_V))

M_aug = np.vstack([np.hstack([MM, Z_al_u]),
                   np.hstack([Z_u_al,    Z_u])])

B_aug = np.zeros((n_tot, 1))
B_aug[:n_V] = B_F.reshape((-1, 1))

n_u = len(B_aug.T)
Omega = la.null_space(np.concatenate((M_aug, B_aug.T)))

assert len(Omega.T) == n_lmb

E_4reg = np.concatenate((M_aug, Omega.T @ J_aug))
A_4reg = np.concatenate((J_aug, np.zeros((n_lmb, n_tot))))
B_4reg = np.concatenate((B_aug, np.zeros((n_lmb, n_u))))

rankEreg = np.linalg.matrix_rank(E_4reg)
assert rankEreg == n_V
q = n_tot - n_V

U, Sigma_all, VT = np.linalg.svd(E_4reg, full_matrices=True)
Sigma = Sigma_all[:n_V]

UT = U.T
V = VT.T

Atil = UT @ A_4reg @ V
A11 = Atil[:n_V, :n_V]
A12 = Atil[:n_V, -q:]
A21 = Atil[-2*q:, :n_V]
A22 = Atil[-2*q:, -q:]

A22plus = np.linalg.solve(A22.T @ A22, A22.T)

V11 = V[:n_V, :n_V]
V12 = V[:n_V, -q:]
V21 = V[-q:, :n_V]
V22 = V[-q:, -q:]

Btil = UT @ B_4reg

B1 = Btil[:n_V, :]
B2 = Btil[n_V:, :]

invSigma = np.reciprocal(Sigma)
F = np.diag(invSigma) @ (A11 - A12 @ A22plus @ A21)
G = np.diag(invSigma) @ (B1 - A12 @ A22plus @ B2)

C1 = B_aug[:n_V, :].T
C2 = B_aug[n_V:, :].T

Vtil11 = V11 - V12 @ A22plus @ A21
Vtil21 = V21 - V22 @ A22plus @ A21

V12plus = V12 @ A22plus @ B2
V22plus = V22 @ A22plus @ B2

H = C1 @ Vtil11 + C2 @ Vtil21
L = - C1 @ V12plus - C2 @ V22plus

T_z2x = np.concatenate((Vtil11, Vtil21))
T_u2x = np.concatenate((V12plus, V22plus))

sys = signal.StateSpace(F, G, H, L)

t_fin = 1.
n_ev = 1000
t_ev = np.linspace(0, t_fin, num = n_ev)
u = np.ones_like(t_ev)
# t_out, y_out, x_out = signal.lsim(sys, u, t_ev)

t_out, y_out = signal.step(sys, T=t_ev)
plt.plot(t_out, y_out)
plt.show()
# pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/ReductionPHDAEind2/'
#
# F_file = 'F'; G_file = 'G'; H_file = 'H'; L_file = 'L'
# T_z2x_file = 'T_z2x'; T_u2x_file = 'T_u2x'
# savemat(pathout + F_file, mdict={F_file: F})
# savemat(pathout + G_file, mdict={G_file: G})
# savemat(pathout + H_file, mdict={H_file: H})
# savemat(pathout + L_file, mdict={L_file: L})
# savemat(pathout + T_z2x_file, mdict={T_z2x_file: T_z2x})
# savemat(pathout + T_u2x_file, mdict={T_u2x_file: T_u2x})
