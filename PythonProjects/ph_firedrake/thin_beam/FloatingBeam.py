# EB beam written with the port Hamiltonian approach

from firedrake import *
import numpy as np
import scipy.linalg as la
from scipy import signal

import matplotlib
import matplotlib.pyplot as plt
from scipy.io import savemat

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
coeff_norm = L*(rho*A/EI)**(0.25)

n = 3
deg = 3

mesh = IntervalMesh(n, L)
x = SpatialCoordinate(mesh)
# plot(mesh)
# plt.show()


# Finite element defition

Vp = FunctionSpace(mesh, "Hermite", deg)
Vq = FunctionSpace(mesh, "Hermite", deg)

n_Vp = Vp.dim()
n_Vq = Vq.dim()
n_V = n_Vp + n_Vq

v_p = TestFunction(Vp)
v_q = TestFunction(Vq)

e_p = TrialFunction(Vp)
e_q = TrialFunction(Vq)

al_p = rho * A * e_p
al_q = 1./EI * e_q

dx = Measure('dx')
ds = Measure('ds')

isnot_P = conditional(gt(x[0], L/3), 1., 0.)
m_test = v_p * isnot_P * dx
m_p = v_p * al_p * isnot_P * dx
m_q = v_q * al_q * dx

petsc_m_q = assemble(m_q, mat_type='aij').M.handle
Mq_FEM = np.array(petsc_m_q.convert("dense").getDenseArray())

petsc_m_p = assemble(m_p, mat_type='aij').M.handle
Mp_FEM = np.array(petsc_m_p.convert("dense").getDenseArray())

plt.spy(Mp_FEM); plt.show()

n_rig = 2
print(Mp_FEM[:n_rig, :n_rig])
Mp_f = Mp_FEM[n_rig:, n_rig:]
Mq = Mq_FEM


Mp_fr = np.zeros((n_Vp - n_rig, n_rig))
Mp_fr[:, 0] = assemble(v_p * rho * A * dx).vector().get_local()[n_rig:]
Mp_fr[:, 1] = assemble(v_p * rho * A * x[0] * dx).vector().get_local()[n_rig:]

Mp_r = np.zeros((n_rig, n_rig))
m_tot = rho * A * L
Mp_r[0][0] = m_tot
Mp_r[1][1] = 1/3 * m_tot * L**2
Mp_r[0][1] = m_tot * L/2
Mp_r[1][0] = m_tot * L/2

Mp = np.zeros((n_Vp, n_Vp))
Mp[:n_rig, :n_rig] = Mp_r
Mp[n_rig:, :n_rig] = Mp_fr
Mp[:n_rig, n_rig:] = Mp_fr.T
Mp[n_rig:, n_rig:] = Mp_f

M_all = la.block_diag(Mp, Mq)
Q_all = la.block_diag(la.inv(Mp), la.inv(Mq))

j_divDiv = -v_p * e_q.dx(0).dx(0) * dx
j_divDivIP = v_q.dx(0).dx(0) * e_p * dx

j_gradgrad = v_q * e_p.dx(0).dx(0) * dx
j_gradgradIP = -v_p.dx(0).dx(0) * e_q * dx

petcs_j_grgr = assemble(j_gradgrad).M.handle
D_f = np.array(petcs_j_grgr.convert("dense").getDenseArray())[:, n_rig:]

J_all = np.zeros((n_V, n_V))
J_all[n_Vp:, n_rig:n_Vp] = D_f
J_all[n_rig:n_Vp, n_Vp:] = -D_f.T

tau_CP = np.array([[1, L], [0, 1]])


b_F = v_p * ds(1)
b_M = v_p.dx(0) * ds(2)

B_Ffl = assemble(b_F).vector().get_local()
B_Mfl = assemble(b_M).vector().get_local()
#
# B_all = np.zeros((n_V, 4))
# B_all[:n_Vp, 2] = B_Ffl
# B_all[:n_Vp, 3] = B_Mfl
# B_all[:n_rig, :n_rig] = np.eye(2)
# B_all[:n_rig, n_rig:] = tau_CP.T
#
# pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/FloatingFramePH/'
# M_file = 'M_pH'; Q_file = 'Q_pH'; J_file = 'J_pH'; B_file = 'B_pH'
# savemat(pathout + M_file, mdict={M_file: M_all})
# savemat(pathout + Q_file, mdict={Q_file: Q_all})
# savemat(pathout + J_file, mdict={J_file: J_all})
# savemat(pathout + B_file, mdict={B_file: B_all})

# tol = 1e-6
# eigenvalues, eigvectors = la.eig(J_all, M_all) # la.eig(J_all[2:, 2:], M_all[2:, 2:]) #
# omega_all = np.imag(eigenvalues)
#
# index = omega_all > 0
#
# omega = omega_all[index]
# eigvec_omega = eigvectors[:, index]
# perm = np.argsort(omega)
# eigvec_omega = eigvec_omega[:, perm]
#
# omega.sort()
# k_n = np.sqrt(omega)*coeff_norm
# print("Smallest positive normalized eigenvalues computed: ")
# for i in range(2*n):
#     print(k_n[i])

#
# eigvec_w = eigvec_omega[:n_Vp, :]
# eigvec_w_real = np.real(eigvec_w)
# eigvec_w_imag = np.imag(eigvec_w)
#
# eig_funH2 = Function(Vp)
# Vp_4proj = FunctionSpace(mesh, "CG", 2)
# eig_funH1 = Function(Vp_4proj)
#
# n_fig = 3
# plot_eigenvector = True
# if plot_eigenvector:
#
#     for i in range(n_fig):
#         z_real = eigvec_w_real[:, i]
#         z_imag = eigvec_w_imag[:, i]
#
#         tol = 1e-6
#         fntsize = 20
#
#         eig_funH2.vector()[:] = z_imag
#         eig_funH1.assign(project(eig_funH2, Vp_4proj))
#         plot(eig_funH1)
#         plt.xlabel('$x$', fontsize=fntsize)
#         plt.title('Eigenvector $e_p$', fontsize=fntsize)
#
#         # if i<4:
#         #     plt.figure()
#         #     plt.plot([0, L], z_real[:n_rig])
#         #     plt.xlabel('$x$', fontsize=fntsize)
#         #     plt.title('Eigenvector $e_p$', fontsize=fntsize)
#         # else:
#         #     eig_funH2.vector()[:] = np.concatenate((np.array([0, 0]), z_real[n_rig:]))
#         #     eig_funH1.assign(project(eig_funH2, Vp_4proj))
#         #     plot(eig_funH1)
#         #     plt.xlabel('$x$', fontsize=fntsize)
#         #     plt.title('Eigenvector $e_p$', fontsize=fntsize)
#
#     plt.show()