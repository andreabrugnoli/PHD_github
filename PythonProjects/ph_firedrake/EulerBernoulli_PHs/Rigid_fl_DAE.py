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

n = 50
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

isnot_P = conditional(ne(x[0], 0.0), 1., 0.)
m_p = v_p * al_p * dx
m_q = v_q * al_q * dx


petsc_m_q = assemble(m_q, mat_type='aij').M.handle
Mq = np.array(petsc_m_q.convert("dense").getDenseArray())

petsc_m_p = assemble(m_p, mat_type='aij').M.handle
Mp_FEM = np.array(petsc_m_p.convert("dense").getDenseArray())

n_rig = 2
Mp_f = Mp_FEM[n_rig:, n_rig:]

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

n_lmb = 2
G = np.eye(n_lmb)
n_tot = n_V + n_lmb
M_aug = np.zeros((n_tot, n_tot))
M_aug[:n_V, :n_V] = M_all

J_aug = np.zeros((n_tot, n_tot))
J_aug[:n_V, :n_V] = J_all
J_aug[:n_lmb, n_V:] = G
J_aug[n_V:, :n_lmb] = -G.T


b_21 = v_p * ds(2)
b_22 = v_p.dx(0) * ds(2)

B_21 = assemble(b_21).vector().get_local().reshape((-1, 1))
B_22 = assemble(b_22).vector().get_local().reshape((-1, 1))

B_aug = np.zeros((n_tot, 1))
B_aug[:n_Vp] = B_21

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
t_ev = np.linspace(0, t_fin, num=n_ev)
u = np.ones_like(t_ev)
t_out, y_out, x_out = signal.lsim(sys, u, t_ev)

e_out = np.zeros((n_tot, n_ev))

for i in range(n_ev):
    e_out[:, i] = np.reshape(T_z2x @ np.reshape(x_out[i, :], (n_V, 1)) + T_u2x, n_tot)

# plt.plot(t_out, e_out[0, :])
# # plt.plot(t_out, y_out)
#
#
# w, mag, phase = signal.bode(sys)
# plt.figure()
# plt.semilogx(w, mag)    # Bode magnitude plot
# plt.figure()
# plt.semilogx(w, phase)  # Bode phase plot
#
# plt.show()

# pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/ReductionPHDAEind2/'
# M_file = 'M_22P'; Q_file = 'Q_22P'; J_file = 'J_22P'; B_file = 'B_22P'
# savemat(pathout + M_file, mdict={M_file: M_all})
# savemat(pathout + Q_file, mdict={Q_file: Q_all})
# savemat(pathout + J_file, mdict={J_file: J_all})
# savemat(pathout + B_file, mdict={B_file: B_all})

tol = 1e-6
eigenvalues, eigvectors = la.eig(J_aug, M_aug) # la.eig(J_all[2:, 2:], M_all[2:, 2:]) #
omega_all = np.imag(eigenvalues)

index = omega_all > 0

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()
k_n = np.sqrt(omega)*coeff_norm
print("Smallest positive normalized eigenvalues computed: ")
for i in range(5):
    print(k_n[i])

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
