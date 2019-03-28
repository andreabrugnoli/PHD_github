# Mindlin plate written with the port Hamiltonian approach

from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)
from math import floor, sqrt, pi
import matplotlib.pyplot as plt

import scipy.linalg as la


E = 2e11
rho = 7900  # kg/m^3
nu = 0.3

b = 0.05
h = 0.01
A = b * h

I = 1./12 * b * h**3

EI = E * I
L = 0.1


n = 100
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

print(V.dim())

n_w = V.sub(0).dim()

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


j = j_divDiv + j_divDivIP
# j = j_gradgrad + j_gradgradIP


bc_w = DirichletBC(V.sub(0), Constant(0.0), 1)
# bc_M = DirichletBC(V.sub(0), Constant(0.0), 2)

g_Hess = - v_p * ds + v_p.dx(0) * ds

boundary_dofs = sorted(bc_w.nodes)

# Assemble the stiffness matrix and the mass matrix.


J = assemble(j, mat_type='aij')
M = assemble(m, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())

# B_u = assemble(b_u_omn, mat_type='aij')
# B_y = assemble(b_y_omn, mat_type='aij')
#
# petsc_b_u =  B_u.M.handle
# petsc_b_y =  B_y.M.handle
#
# # print(B_u.array().shape)
# B_in  = np.array(petsc_b_u.convert("dense").getDenseArray())[:, boundary_dofs]
# B_out = np.array(petsc_b_y.convert("dense").getDenseArray())[boundary_dofs, :]

#
# N_al = V.dim()
# N_u = len(boundary_dofs)
# # print(N_u)
#
# Z_u = np.zeros((N_u, N_u))
#
#
# J_aug = np.vstack([ np.hstack([JJ, B_in]),
#                     np.hstack([B_out, Z_u])
#                 ])
#
# Z_al_u = np.zeros((N_al, N_u))
# Z_u_al = np.zeros((N_u, N_al))
#
# M_aug = np.vstack([ np.hstack([MM, Z_al_u]),
#                     np.hstack([Z_u_al,    Z_u])
#                  ])
#

tol = 0
eigenvalues, eigvectors = la.eig(JJ, MM)
omega_all = np.imag(eigenvalues)

index = omega_all>=tol

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()

k_n = omega**(0.5)*L*(rho*A/(EI))**(0.25)
print("Smallest positive normalized eigenvalues computed: ")
for i in range(10):
    print(k_n[i], omega[i])

plt.plot(np.real(eigenvalues), np.imag(eigenvalues), 'bo')
plt.show()

# eigvec_w = eigvec_omega[dofs_Vpw, :]
# eigvec_w_real = np.real(eigvec_w)
# eigvec_w_imag = np.imag(eigvec_w)

# import matplotlib
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# from matplotlib import cm
#
# plt.close('all')
# matplotlib.rcParams['text.usetex'] = True
# matplotlib.rcParams['text.latex.unicode'] = True

# n_fig = 5
#
# if plot_eigenvector == 'y':
#
#     for i in range(n_fig):
#         z_real = eigvec_w_real[:, i]
#         z_imag = eigvec_w_imag[:, i]
#
#         tol = 1e-6
#         fntsize = 20
#
#         if matplotlib.is_interactive():
#             plt.ioff()
#
#
#         z1 = z_real
#         minZ1 = min(z1)
#         maxZ1 = max(z1)
#
#         if minZ1 != maxZ1:
#
#             fig1 = plt.figure(i)
#
#             ax1 = fig1.add_subplot(111, projection='3d')
#             # ax1.zaxis._axinfo['label']['space_factor'] = 20
#
#             ax1.set_xbound(min(x) - tol, max(x) + tol)
#             ax1.set_xlabel('$x$', fontsize=fntsize)
#
#             ax1.set_ybound(min(y) - tol, max(y) + tol)
#             ax1.set_ylabel('$y$', fontsize=fntsize)
#
#             ax1.set_title('$v_{e_{p,w}}$', fontsize=fntsize)
#
#             ax1.set_zlim3d(minZ1 - 0.01*abs(minZ1), maxZ1 + 0.01*abs(maxZ1))
#             ax1.w_zaxis.set_major_locator(LinearLocator(10))
#             ax1.w_zaxis.set_major_formatter(FormatStrFormatter('%.04f'))
#
#             ax1.plot_trisurf(x, y, z1, cmap=cm.jet, linewidth=0, antialiased=False)
#
#             # path_out1 = "/home/a.brugnoli/PycharmProjects/Mindlin_Phs_fenics/Figures_Eig_Min/RealEig/"
#             # plt.savefig(path_out1 + "Case" + case_study + "_el" + str(n) + "_deg" + str(deg) + "_thick_" + \
#             #             str(thick) + "_eig_" + str(i+1) + ".eps", format="eps")
#
#
#     for i in range(n_fig):
#         z2 = z_imag
#         minZ2 = min(z2)
#         maxZ2 = max(z2)
#
#         if minZ2 != maxZ2:
#
#             fig2 = plt.figure(n_fig + i+1)
#
#             ax2 = fig2.add_subplot(111, projection='3d')
#             # ax2.zaxis._axinfo['label']['space_factor'] = 20
#
#             ax2.set_xlim(min(x) - tol, max(x) + tol)
#             ax2.set_xlabel('$x$', fontsize=fntsize)
#
#             ax2.set_ylim(min(y) - tol, max(y) + tol)
#             ax2.set_ylabel('$y$', fontsize=fntsize)
#
#             # ax2.set_zlabel('$v_{e_{p,w}}$', fontsize=fntsize)
#             ax2.set_title('$v_{e_{p,w}}$', fontsize=fntsize)
#
#             ax2.set_zlim(minZ2 - 0.01 * abs(minZ2), maxZ2 + 0.01 * abs(maxZ2))
#             ax2.w_zaxis.set_major_locator(LinearLocator(10))
#             ax2.w_zaxis.set_major_formatter(FormatStrFormatter('%.04f'))
#
#             ax2.plot_trisurf(x, y, z2, cmap=cm.jet, linewidth=0, antialiased=False)
#
#             # path_out2 = "/home/a.brugnoli/PycharmProjects/Mindlin_Phs_fenics/Figures_Eig_Min/ImagEig/"
#             # plt.savefig(path_out2 + "Case" + case_study + "_el" + str(n) + "_deg" + str(deg) + "_thick_" \
#             #             + str(thick) + "_eig_" + str(i+1) + ".eps", format="eps")
#
#     plt.show()
