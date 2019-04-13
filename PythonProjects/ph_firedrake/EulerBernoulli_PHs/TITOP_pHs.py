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
m_p = v_p * al_p * dx
m_q = v_q * al_q * dx


petsc_mq = assemble(m_q, mat_type='aij').M.handle
Mq = np.array(petsc_mq.convert("dense").getDenseArray())

petsc_mp = assemble(m_p, mat_type='aij').M.handle
Mp_FEM = np.array(petsc_mp.convert("dense").getDenseArray())

n_rig = 2
Mp_f = Mp_FEM[n_rig:, n_rig:]

Mp_fr = np.zeros((n_Vp - n_rig, n_rig))
Mp_fr[:, 0] = assemble(rho * A * v_p * dx).vector().get_local()[n_rig:]
Mp_fr[:, 1] = assemble(rho * A * v_p * x[0] * dx).vector().get_local()[n_rig:]

Mp_r = np.zeros((n_rig, n_rig))
m_tot =  rho * A * L
Mp_r[0][0] = m_tot
Mp_r[1][1] = 1/3 * m_tot * L**2
Mp_r[0][1] = m_tot * L/2
Mp_r[1][0] = m_tot * L/2

Mp = np.zeros((n_Vp, n_Vp))
Mp[:n_rig, :n_rig] = Mp_r
Mp[n_rig:n_Vp, :n_rig] = Mp_fr
Mp[:n_rig, n_rig:] = Mp_fr.T
Mp[n_rig:, n_rig:] = Mp_f


M_all = la.block_diag(Mp, Mq)
Q_all = la.block_diag(la.inv(Mp), la.inv(Mq))


j_divDiv = -v_p * e_q.dx(0).dx(0) * dx
j_divDivIP = v_q.dx(0).dx(0) * e_p * dx

j_gradgrad = v_q * e_p.dx(0).dx(0) * dx
j_gradgradIP = -v_p.dx(0).dx(0) * e_q * dx

petcs_j_qp = assemble(j_gradgrad).M.handle
D_f = np.array(petcs_j_qp.convert("dense").getDenseArray())[:, n_rig:]

D_r = np.zeros((n_Vq, n_rig))
D_r[:, 0] = assemble(v_q.dx(0).dx(0) * dx).vector().get_local()
D_r[:, 1] = assemble(v_q.dx(0).dx(0) * x[0] * dx).vector().get_local()

J_all = np.zeros((n_V, n_V))
J_all[n_Vp:, :n_rig] = D_r
J_all[:n_rig, n_Vp:] = -D_r.T
J_all[n_Vp:, n_rig:n_Vp] = D_f
J_all[n_rig:n_Vp, n_Vp:] = -D_f.T


b_21 = v_p * ds(2)
b_22 = v_p.dx(0) * ds(2)
b_33 = -v_q.dx(0) * ds
b_34 = v_q * ds - x[0] * v_q.dx(0) * ds

n_u = 4

B_21 = assemble(b_21).vector().get_local().reshape((-1, 1))
B_22 = assemble(b_22).vector().get_local().reshape((-1, 1))
B_33 = assemble(b_33).vector().get_local().reshape((-1, 1))
B_34 = assemble(b_34).vector().get_local().reshape((-1, 1))

B_all = np.zeros((n_V, 1))
B_all[:n_Vp] = B_21

pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/ReductionPHDAEind2/'
Q_file = 'Q_22P'; J_file = 'J_22P'; B_file = 'B_22P'
savemat(pathout + Q_file, mdict={Q_file: Q_all})
savemat(pathout + J_file, mdict={J_file: J_all})
savemat(pathout + B_file, mdict={B_file: B_all})

# tol = 1e-6
# eigenvalues, eigvectors = la.eig(J_all, M_all)
# omega_all = np.imag(eigenvalues)
#
# index = omega_all >= -tol
#
# omega = omega_all[index]
# eigvec_omega = eigvectors[:, index]
# perm = np.argsort(omega)
# eigvec_omega = eigvec_omega[:, perm]
#
# omega.sort()
#
# k_n = omega**(0.5)*L*(rho*A/(EI))**(0.25)
# print("Smallest positive normalized eigenvalues computed: ")
# for i in range(4):
#     print(k_n[i])
#
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
#
#         # eig_funH2.vector()[:] = z_real
#         # eig_funH1.assign(project(eig_funH2, Vp_4proj))
#         # plot(eig_funH1)
#
#         eig_funH2.vector()[:] = z_imag
#         eig_funH1.assign(project(eig_funH2, Vp_4proj))
#         plot(eig_funH1)
#         plt.xlabel('$x$', fontsize=fntsize)
#         plt.title('Eigenvector $e_p$', fontsize=fntsize)
#
#     plt.show()
