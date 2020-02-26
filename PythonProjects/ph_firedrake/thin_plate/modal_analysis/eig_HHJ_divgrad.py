# Mindlin plate written with the port Hamiltonian approach

from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt

import scipy.linalg as la

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

j_1 = + inner(grad(v_p), div(e_q)) * dx \
      - dot(grad(v_p), s_ver) * dot(dot(e_q, s_ver), n_ver) * ds \
      - dot(grad(v_p('+')), s_ver('+')) * jump(dot(e_q, s_ver), n_ver) * dS \
      - jump(grad(v_p), n_ver) * dot(dot(e_q('+'), n_ver('+')), n_ver('+')) * dS \
      - dot(grad(v_p), n_ver) * dot(dot(e_q, n_ver), n_ver) * ds
      # - dot(grad(v_p('+')), s_ver('+')) * dot(dot(e_q('+'), s_ver('+')), n_ver('+')) * dS

j_2 = - inner(div(v_q), grad(e_p)) * dx \
  + dot(dot(v_q, s_ver), n_ver) * dot(grad(e_p), s_ver) * ds \
  + jump(dot(v_q, s_ver), n_ver) * dot(grad(e_p('+')), s_ver('+')) * dS \
  + dot(dot(v_q('+'), n_ver('+')), n_ver('+')) * jump(grad(e_p), n_ver) * dS \
  + dot(dot(v_q, n_ver), n_ver) * dot(grad(e_p), n_ver) * ds
  #  + dot(dot(v_q('+'), s_ver('+')), n_ver('+')) * dot(grad(e_p('+')), s_ver('+')) * dS

j_form = j_1 + j_2

bc_1, bc_2, bc_3, bc_4 = bc_input

bc_dict = {1: bc_1, 3: bc_2, 2: bc_3, 4: bc_4}

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
        bc_q = DirichletBC(Vq, Constant(((0.0, 0.0), (0.0, 0.0))), key)

        for node in bc_p.nodes:
            boundary_dofs.append(node)
        bcs.append(bc_p)

        for node in bc_q.nodes:
            boundary_dofs.append(n_Vp + node)
        bcs.append(bc_q)

    elif val == 'F':

        bc_q = DirichletBC(Vq, Constant(((0.0, 0.0), (0.0, 0.0))), key)
        for node in bc_q.nodes:
            boundary_dofs.append(n_Vp + node)
        bcs.append(bc_q)

boundary_dofs = sorted(boundary_dofs)
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

Z_lmb = np.zeros((n_lmb, n_lmb))

J_aug = np.vstack([np.hstack([JJ, G]),
                   np.hstack([-G.T, Z_lmb])
                ])

M_aug = la.block_diag(MM, Z_lmb)
tol = 10**(-9)

eigenvalues, eigvectors = la.eig(J_aug, M_aug)
omega_all = np.imag(eigenvalues)

tol = 10**(-9)
index = omega_all >= tol

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()

omega_tilde = omega * norm_coeff
n_om = 10

for i in range(n_om):
    print(omega_tilde[i])

n_fig = 5

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

    for i in range(n_fig):
        eig_real_w = Function(Vp)
        eig_imag_w = Function(Vp)

        eig_real_p = np.real(eigvec_omega[:n_Vp, i])
        eig_imag_p = np.imag(eigvec_omega[:n_Vp, i])
        eig_real_w.vector()[:] = eig_real_p
        eig_imag_w.vector()[:] = eig_imag_p

        norm_real_eig = np.linalg.norm(eig_real_w.vector().get_local())
        norm_imag_eig = np.linalg.norm(eig_imag_w.vector().get_local())

        if norm_imag_eig > norm_real_eig:
            triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_imag_w, 10)
        else:
            triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_real_w, 10)


        figure = plt.figure(i)
        ax = figure.add_subplot(111, projection="3d")

        ax.plot_trisurf(triangulation, z_goodeig, cmap=cm.jet)

        ax.set_xbound(-tol, 1 + tol)
        ax.set_xlabel('$x [m]$', fontsize=fntsize)

        ax.set_ybound(-tol, 1 + tol)
        ax.set_ylabel('$y [m]$', fontsize=fntsize)

        ax.set_title('$v_{e_{w}}$', fontsize=fntsize)

        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

        path_out2 = "/home/a.brugnoli/PycharmProjects/firedrake/Kirchhoff_PHs/Eig_Kirchh/Imag_Eig/"
        # plt.savefig(path_out2 + "Case" + bc_input + "_el" + str(n) + "_FE_" + name_FEp + "_eig_" + str(i+1) + ".eps", format="eps")

plt.show()
