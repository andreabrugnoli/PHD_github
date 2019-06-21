# EB beam written with the port Hamiltonian approach

from firedrake import *
import numpy as np
import scipy.linalg as la

import matplotlib
import matplotlib.pyplot as plt
from scipy.io import savemat

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


# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
L = 1

mesh = IntervalMesh(n, L)

# plot(mesh)
# plt.show()


# Finite element defition

V_p = FunctionSpace(mesh, "Hermite", deg)
V_q = FunctionSpace(mesh, "Hermite", deg)

V = V_p * V_q

n = V.dim()
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
g_r = [] # gCF_divDiv

G_L = np.zeros((n, len(g_l)))
G_R = np.zeros((n, len(g_r)))


for counter, item in enumerate(g_l):
    G_L[:, counter] = assemble(item).vector().get_local()

for counter, item in enumerate(g_r):
    G_R[:, counter] = assemble(item).vector().get_local()

G = np.concatenate((G_L, G_R), axis=1)

# Assemble the stiffness matrix and the mass matrix.
J = assemble(j, mat_type='aij')
M = assemble(m, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())

n_V = V.dim()
n_lmb = len(G.T)

GannL = la.null_space(G.T).T
T = np.concatenate((GannL, la.inv(G.T @ G) @ G.T))

M_til = T @ MM @ T.T
J_til = T @ JJ @ T.T

b_F = v_p * ds(2)
b_M = v_p.dx(0) * ds(2)

B_Ffl = assemble(b_F).vector().get_local()
B_Mfl = assemble(b_M).vector().get_local()

B = np.zeros((n_V, 2))
B[:, 0] = B_Ffl
B[:, 1] = B_Mfl

B_til = T @ B
G_til = T @ G


J_ode = J_til[:n_V-n_lmb, :n_V-n_lmb]
M_ode = M_til[:n_V-n_lmb, :n_V-n_lmb]

B_ode = B_til[:n_V-n_lmb, :]

pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/FloatingFramePH/PH_matrices/'
M_file = 'Mode_pH'; J_file = 'Jode_pH'; B_file = 'Bode_pH'
savemat(pathout + M_file, mdict={M_file: M_ode})
savemat(pathout + J_file, mdict={J_file: J_ode})
savemat(pathout + B_file, mdict={B_file: B_ode})
