# Mindlin plate written with the port Hamiltonian approach

from firedrake import *
import numpy as np

np.set_printoptions(threshold=np.inf)
from math import floor, sqrt, pi
import matplotlib.pyplot as plt
from firedrake.plot import _two_dimension_triangle_func_val

import scipy.linalg as la
import scipy.sparse as spa
import scipy.sparse.linalg as sp_la

n = 10

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
L = 1
l_x = L
l_y = L
n_x, n_y = n, n
mesh = UnitSquareMesh(n_x, n_y, quadrilateral=False)

# plot(mesh)
# plt.show()

name_FEp = 'Bell'
name_FEq = 'DG'
deg_q = 3

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

v_q = as_tensor([[v_q[0], v_q[1]],
                 [v_q[1], v_q[2]]])

e_q = as_tensor([[e_q[0], e_q[1]],
                 [e_q[1], e_q[2]]])


al_p = rho * h * e_p
al_q = bending_curv(e_q)

dx = Measure('dx')
ds = Measure('ds')
m_p = dot(v_p, al_p) * dx
m_q = inner(v_q, al_q) * dx
m = m_p + m_q


j_gradgrad = inner(v_q, grad(grad(e_p))) * dx
j_gradgradIP = -inner(grad(grad(v_p)), e_q) * dx

j = j_gradgrad + j_gradgradIP


# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1

bc_1, bc_3, bc_2, bc_4 = bc_input

bc_dict = {1: bc_1, 2: bc_2, 3: bc_3, 4: bc_4}

n = FacetNormal(mesh)
# s = as_vector([-n[1], n[0]])

V_qn = FunctionSpace(mesh, 'Lagrange', 1)
V_Mnn = FunctionSpace(mesh, 'Lagrange', 1)

Vu = V_qn * V_Mnn

q_n, M_nn = TrialFunction(Vu)

v_omn = dot(grad(v_p), n)

b_vec = []
for key,val in bc_dict.items():
    if val == 'C':
        b_vec.append(v_p * q_n * ds(key) + v_omn * M_nn * ds(key))
    elif val == 'S':
        b_vec.append(v_p * q_n * ds(key))

b_u = sum(b_vec)
# Assemble the stiffness matrix and the mass matrix.


J = assemble(j, mat_type='aij')
M = assemble(m, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle

JJ = spa.csr_matrix(petsc_j.getValuesCSR()[::-1])
MM = spa.csr_matrix(petsc_m.getValuesCSR()[::-1])

if b_u:
    B_u = assemble(b_u, mat_type='aij')
    petsc_b_u = B_u.M.handle
    B_in = spa.csr_matrix(petsc_b_u.getValuesCSR()[::-1])

    rows, cols = spa.csr_matrix.nonzero(B_in)
    set_cols = np.array(list(set(cols)))

    n_lmb = len(set_cols)
    G = np.zeros((n_V, n_lmb))
    for r, c in zip(rows, cols):
        ind_col = np.where(set_cols == c)[0]
        G[r, ind_col] = B_in[r, c]

# G_ortho = la.null_space(G.T).T
#
# G_ortho = spa.csr_matrix(G_ortho)
#
# J_til = G_ortho.dot(JJ.dot(G_ortho.transpose()))
# M_til = G_ortho.dot(MM.dot(G_ortho.transpose()))

Z_lmb = spa.csr_matrix((n_lmb, n_lmb))
J_til = spa.vstack((spa.hstack((JJ, G)), spa.hstack((-G.T, Z_lmb))))
M_til = spa.block_diag((MM, Z_lmb))

tol = 10 ** (-6)

n_om = 10

shift = 10/norm_coeff

eigenvalues, eigvectors = sp_la.eigs(J_til, k=2*n_om, M=M_til, sigma=shift, which='LM', tol=1e-6, maxiter=5000)

omega_all = np.imag(eigenvalues)

index = omega_all >= tol

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()

# NonDimensional China Paper


omega_tilde = L**2*sqrt(rho*h/D)*omega
for i in range(min(6, len(omega_tilde))):
    print(omega_tilde[i])


plot_eigenvectors = True
if plot_eigenvectors:

    fntsize = 15

    import matplotlib
    from matplotlib import pyplot as plt
    from matplotlib.ticker import LinearLocator, FormatStrFormatter
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D

    plt.close('all')
    matplotlib.rcParams['text.usetex'] = True

    for i in range(n_om):
        eig_real_w = Function(Vp)
        eig_imag_w = Function(Vp)

        eig_real_p = np.real(eigvec_omega[:n_Vp, i])
        eig_imag_p = np.imag(eigvec_omega[:n_Vp, i])
        eig_real_w.vector()[:] = eig_real_p
        eig_imag_w.vector()[:] = eig_imag_p

        Vp_CG = FunctionSpace(mesh, 'Lagrange', 3)
        eig_real_wCG = project(eig_real_w, Vp_CG)
        eig_imag_wCG = project(eig_imag_w, Vp_CG)

        norm_real_eig = np.linalg.norm(eig_real_wCG.vector().get_local())
        norm_imag_eig = np.linalg.norm(eig_imag_wCG.vector().get_local())

        if norm_imag_eig > norm_real_eig:
            # triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_imag_wCG, 10)
            triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_imag_w, 10)

        else:
            # triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_real_wCG, 10)
            triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_real_w, 10)


        figure = plt.figure(i)
        ax = figure.add_subplot(111, projection="3d")

        ax.plot_trisurf(triangulation, z_goodeig, cmap=cm.jet)

        ax.set_xbound(-tol, l_x + tol)
        ax.set_xlabel('$x [m]$', fontsize=fntsize)

        ax.set_ybound(-tol, l_y + tol)
        ax.set_ylabel('$y [m]$', fontsize=fntsize)

        ax.set_title('$v_{e_{w}}$', fontsize=fntsize)

        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

        path_out2 = "/home/a.brugnoli/PycharmProjects/firedrake/Kirchhoff_PHs/Eig_Kirchh/Imag_Eig/"
        # plt.savefig(path_out2 + "Case" + bc_input + "_el" + str(n) + "_FE_" + name_FEp + "_eig_" + str(i+1) + ".eps", format="eps")

plt.show()





