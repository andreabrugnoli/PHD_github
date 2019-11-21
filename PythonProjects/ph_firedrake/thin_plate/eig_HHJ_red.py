# Mindlin plate written with the port Hamiltonian approach

from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt

import scipy.linalg as la
from modules_ph.classes_phsystem import SysPhdaeRig

import matplotlib
import matplotlib.pyplot as plt
from firedrake.plot import _two_dimension_triangle_func_val
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

matplotlib.rcParams['text.usetex'] = True

n = 5
r = 2 #int(input('Degree for FE: '))

E = 2e11 # Pa
rho = 8000  # kg/m^3
# E = 1
# rho = 1  # kg/m^3

nu = 0.3
h = 0.01

plot_eigenvector = 'y'

bc_input = input('Select Boundary Condition: ')

L = 1

D = E * h ** 3 / (1 - nu ** 2) / 12
fl_rot = 12 / (E * h ** 3)
norm_coeff = L ** 2 * np.sqrt(rho*h/D)
# Useful Matrices

# Operators and functions
def gradSym(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
    # return sym(nabla_grad(u))

def bending_curv(momenta):
    kappa = fl_rot * ((1+nu)*momenta - nu * Identity(2) * tr(momenta))
    return kappa


# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
n_x, n_y = n, n
mesh = UnitSquareMesh(n_x, n_y, quadrilateral=False)

# domain = mshr.Rectangle(Point(0, 0), Point(l_x, l_y))
# mesh = mshr.generate_mesh(domain, n, "cgal")

# plot(mesh)
# plt.show()


# Finite element defition

Vp = FunctionSpace(mesh, 'CG', r+1)
Vq = FunctionSpace(mesh, 'HHJ', r)
V = Vp * Vq

n_Vp = V.sub(0).dim()
n_Vq = V.sub(1).dim()
n_V = V.dim()
print(n_V)

v = TestFunction(V)
v_p, v_q = split(v)

e = TrialFunction(V)
e_p, e_q = split(e)

al_p = rho * h * e_p
al_q = bending_curv(e_q)

dx = Measure('dx')
ds = Measure('ds')
dS = Measure("dS")

m_form = inner(v_p, al_p) * dx + inner(v_q, al_q) * dx

n_ver = FacetNormal(mesh)
s_ver = as_vector([-n_ver[1], n_ver[0]])

e_mnn = inner(e_q, outer(n_ver, n_ver))
v_mnn = inner(v_q, outer(n_ver, n_ver))

e_mns = inner(e_q, outer(n_ver, s_ver))
v_mns = inner(v_q, outer(n_ver, s_ver))

j_1 = - inner(grad(grad(v_p)), e_q) * dx \
      + jump(grad(v_p), n_ver) * dot(dot(e_q('+'), n_ver('+')), n_ver('+')) * dS \
      + dot(grad(v_p), n_ver) * dot(dot(e_q, n_ver), n_ver) * ds

j_2 = + inner(v_q, grad(grad(e_p))) * dx \
      - dot(dot(v_q('+'), n_ver('+')), n_ver('+')) * jump(grad(e_p), n_ver) * dS \
      - dot(dot(v_q, n_ver), n_ver) * dot(grad(e_p), n_ver) * ds


j_form = j_1 + j_2

bc_1, bc_2, bc_3, bc_4 = bc_input

bc_dict = {1: bc_1, 2: bc_2, 3: bc_3, 4: bc_4}

# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1

bcs = []
boundary_dofs = []

for key, val in bc_dict.items():

    if val == 'C':
        bc_p = DirichletBC(Vp, Constant(0.0), key)
        for node in bc_p.nodes:
            boundary_dofs.append(node)
        bcs.append(bc_p)

    elif val == 'S':
        bc_p = DirichletBC(Vp, Constant(0.0), key)
        for node in bc_p.nodes:
            boundary_dofs.append(node)
        bcs.append(bc_p)
        bc_q = DirichletBC(Vq, Constant(((0.0, 0.0), (0.0, 0.0))), key)
        for node in bc_q.nodes:
            boundary_dofs.append(n_Vp + node)
        bcs.append(bc_q)

    elif val == 'F':
        bc_q = DirichletBC(Vq, Constant(((0.0, 0.0), (0.0, 0.0))), key)
        for node in bc_q.nodes:
            boundary_dofs.append(n_Vp + node)
        bcs.append(bc_q)


boundary_dofs = sorted(list(set(boundary_dofs)))
n_lmb = len(boundary_dofs)

G = np.zeros((n_V, n_lmb))
for (i, j) in enumerate(boundary_dofs):
    G[j, i] = 1

J = assemble(j_form, mat_type='aij')
M = assemble(m_form, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())

x, y = SpatialCoordinate(mesh)
con_dom1 = And(And(gt(x, L / 4), lt(x, 3 * L / 4)), And(gt(y, L / 4), lt(y, 3 * L / 4)))
Dom_f = conditional(con_dom1, 1., 0.)
B_f = assemble(v_p * Dom_f * dx).vector().get_local()

Z_lmb = np.zeros((n_lmb, n_lmb))

J_aug = np.vstack([np.hstack([JJ, G]),
                   np.hstack([-G.T, Z_lmb])
                ])

E_aug = la.block_diag(MM, Z_lmb)
B_aug = np.concatenate((B_f, np.zeros(n_lmb, )), axis=0).reshape((-1, 1))

plate_full = SysPhdaeRig(n_V+n_lmb, n_lmb, 0, n_Vp, n_Vq, E_aug, J_aug, B_aug)
# plate_full = SysPhdaeRig(n_V, 0, 0, n_Vp, n_Vq, MM, JJ, G)

print(n_lmb, np.linalg.matrix_rank(G))
s0 = 100
n_red = 30
plate_red, V_red = plate_full.reduce_system(s0, n_red)
Vall_red = la.block_diag(V_red, np.eye(n_lmb))

E_red = plate_red.E
J_red = plate_red.J
B_red = plate_red.B

# J_red = np.vstack([np.hstack([J_red, B_red]),
#                       np.hstack([-B_red.T, Z_lmb])
#                       ])
#
# E_red = la.block_diag(E_red, Z_lmb)

tol = 10 ** (-6)

eigenvaluesF, eigvectorsF = la.eig(J_aug, E_aug)
omega_allF = np.imag(eigenvaluesF)

indexF = omega_allF >= tol

omega_full = omega_allF[indexF]
eigvec_full = eigvectorsF[:, indexF]
permF = np.argsort(omega_full)
eigvec_full = eigvec_full[:, permF]
omega_full.sort()

print(J_red.shape)
eigenvaluesR, eigvectorsR = la.eig(J_red, E_red)
omega_allR = np.imag(eigenvaluesR)

index = omega_allR >= tol

omega_red = omega_allR[index]
eigvec_red = eigvectorsR[:, index]
permR = np.argsort(omega_red)
eigvec_red = eigvec_red[:, permR]
omega_red.sort()

print(Vall_red.shape, eigvec_red.shape)

plt.plot(np.real(eigenvaluesF), np.imag(eigenvaluesF), 'r+', np.real(eigenvaluesR), np.imag(eigenvaluesR), 'bo')
plt.legend(("Eigenvalues full", "Eigenvalues reduced"))
plt.show()

# NonDimensional China Paper

n_om = 5

omegaF_tilde = L**2*sqrt(rho*h/D)*omega_full
omegaR_tilde = L**2*sqrt(rho*h/D)*omega_red

# for i in range(n_om):
#     print(omegaF_tilde[i])

for i in range(n_om):
    print(omegaF_tilde[i], omegaR_tilde[i])

n_fig = 5
plot_eigenvectors = False
if plot_eigenvectors:

    fntsize = 15

    # for i in range(n_fig):
    #     eig_real_w = Function(Vp)
    #     eig_imag_w = Function(Vp)
    #
    #     eig_real_p = np.real(eigvec_full[:n_Vp, i])
    #     eig_imag_p = np.imag(eigvec_full[:n_Vp, i])
    #     eig_real_w.vector()[:] = eig_real_p
    #     eig_imag_w.vector()[:] = eig_imag_p
    #
    #     Vp_CG = FunctionSpace(mesh, 'Lagrange', 3)
    #     eig_real_wCG = project(eig_real_w, Vp_CG)
    #     eig_imag_wCG = project(eig_imag_w, Vp_CG)
    #
    #     norm_real_eig = np.linalg.norm(eig_real_wCG.vector().get_local())
    #     norm_imag_eig = np.linalg.norm(eig_imag_wCG.vector().get_local())
    #
    #     if norm_imag_eig > norm_real_eig:
    #         triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_imag_wCG, 10)
    #     else:
    #         triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_real_wCG, 10)
    #
    #     figure = plt.figure()
    #     ax = figure.add_subplot(111, projection="3d")
    #
    #     ax.plot_trisurf(triangulation, z_goodeig, cmap=cm.jet)
    #
    #     ax.set_xbound(-tol, l_x + tol)
    #     ax.set_xlabel('$x [m]$', fontsize=fntsize)
    #
    #     ax.set_ybound(-tol, l_y + tol)
    #     ax.set_ylabel('$y [m]$', fontsize=fntsize)
    #
    #     ax.set_title('$v_{e_{w}}$', fontsize=fntsize)
    #
    #     ax.w_zaxis.set_major_locator(LinearLocator(10))
    #     ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))
    #
    #     # path_out2 = "/home/a.brugnoli/PycharmProjects/firedrake/Kirchhoff_PHs/Eig_Kirchh/Imag_Eig/"
    #     # plt.savefig(path_out2 + "Case" + bc_input + "_el" + str(n) + "_FE_" + name_FEp + "_eig_" + str(i+1) + ".eps", format="eps")


    for i in range(n_fig):
        eig_real_w = Function(Vp)
        eig_imag_w = Function(Vp)

        eig_real_p = np.real((Vall_red @ eigvec_red[:, i])[:n_Vp])
        eig_imag_p = np.imag((Vall_red @ eigvec_red[:, i])[:n_Vp])
        eig_real_w.vector()[:] = eig_real_p
        eig_imag_w.vector()[:] = eig_imag_p

        Vp_CG = FunctionSpace(mesh, 'Lagrange', 3)
        eig_real_wCG = project(eig_real_w, Vp_CG)
        eig_imag_wCG = project(eig_imag_w, Vp_CG)

        norm_real_eig = np.linalg.norm(eig_real_wCG.vector().get_local())
        norm_imag_eig = np.linalg.norm(eig_imag_wCG.vector().get_local())

        if norm_imag_eig > norm_real_eig:
            triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_imag_wCG, 10)
        else:
            triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_real_wCG, 10)

        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")

        ax.plot_trisurf(triangulation, z_goodeig, cmap=cm.jet)

        ax.set_xbound(-tol, 1 + tol)
        ax.set_xlabel('$x [m]$', fontsize=fntsize)

        ax.set_ybound(-tol, 1 + tol)
        ax.set_ylabel('$y [m]$', fontsize=fntsize)

        ax.set_title('$v_{e_{w}}$', fontsize=fntsize)

        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

        # path_out2 = "/home/a.brugnoli/PycharmProjects/firedrake/Kirchhoff_PHs/Eig_Kirchh/Imag_Eig/"
        # plt.savefig(path_out2 + "Case" + bc_input + "_el" + str(n) + "_FE_" + name_FEp + "_eig_" + str(i+1) + ".eps", format="eps")

plt.show()
