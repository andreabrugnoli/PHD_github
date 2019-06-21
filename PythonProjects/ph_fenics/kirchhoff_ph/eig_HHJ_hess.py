# Mindlin plate written with the port Hamiltonian approach

from fenics import *
import numpy as np
np.set_printoptions(threshold=np.inf)
import mshr
import matplotlib.pyplot as plt

import scipy.linalg as la

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

matplotlib.rcParams['text.usetex'] = True

n = 5
r = 2 #int(input('Degree for FE: '))
nreq = 10

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


mesh = RectangleMesh(Point(0, 0), Point(L, L), n, n, "right/left")

# domain = mshr.Rectangle(Point(0, 0), Point(l_x, l_y))
# mesh = mshr.generate_mesh(domain, n, "cgal")

# plot(mesh)
# plt.show()

# Domain, Subdomains, Boundary, Suboundaries


class Left(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 0.0) < DOLFIN_EPS and on_boundary


class Right(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - L) < DOLFIN_EPS and on_boundary


class Lower(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1] - 0.0) < DOLFIN_EPS and on_boundary


class Upper(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1] - L) < DOLFIN_EPS and on_boundary


# Boundary conditions on rotations
left = Left()
right = Right()
lower = Lower()
upper = Upper()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
left.mark(boundaries, 1)
lower.mark(boundaries, 2)
right.mark(boundaries, 3)
upper.mark(boundaries, 4)

# Finite element defition

CG = FiniteElement('CG', mesh.ufl_cell(), r + 1)
HHJ = FiniteElement('HHJ', mesh.ufl_cell(), r)
V = FunctionSpace(mesh, CG * HHJ)
n_V = V.dim()
print(n_V)


v = TestFunction(V)
v_p, v_q = split(v)

e = TrialFunction(V)
e_p, e_q = split(e)

al_p = rho * h * e_p
al_q = bending_curv(e_q)

dx = Measure('dx')
ds = Measure('ds', subdomain_data=boundaries)
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
loc_dict = {1: left, 2: lower, 3: right, 4: upper}

bcs = []
for key, val in bc_dict.items():

    if val == 'C':
        bcs.append(DirichletBC(V.sub(0), Constant(0.0), loc_dict[key]))
    elif val == 'S':
        bcs.append(DirichletBC(V.sub(0), Constant(0.0), loc_dict[key]))
        bcs.append(DirichletBC(V.sub(1), Constant(((0.0, 0.0), (0.0, 0.0))), loc_dict[key]))
    elif val == 'F':
        bcs.append(DirichletBC(V.sub(1), Constant(((0.0, 0.0), (0.0, 0.0))), loc_dict[key]))

boundary_dofs = []
for bc in bcs:
    for key in bc.get_boundary_values().keys():
        boundary_dofs.append(key)
boundary_dofs = sorted(boundary_dofs)
n_lmb = len(boundary_dofs)

G = np.zeros((n_V, n_lmb))
for (i, j) in enumerate(boundary_dofs):
    G[j, i] = 1

J, M, B = PETScMatrix(), PETScMatrix(), PETScMatrix()
n_V = V.dim()

J, M = PETScMatrix(), PETScMatrix()

J = assemble(j_form)
M = assemble(m_form)
JJ = J.array()
MM = M.array()

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

for i in range(nreq):
    print(omega_tilde[i])

d = mesh.geometry().dim()
dofV_x = V.tabulate_dof_coordinates().reshape((-1, d))

dofs_Vp = V.sub(0).dofmap().dofs()
dofs_Vq = V.sub(1).dofmap().dofs()

dofVpw_x = dofV_x[dofs_Vp]

x = dofVpw_x[:, 0]
y = dofVpw_x[:, 1]

eigvec_w = eigvec_omega[dofs_Vp, :]
eigvec_w_real = np.real(eigvec_w)
eigvec_w_imag = np.imag(eigvec_w)

n_fig = 5

if plot_eigenvector == 'y':

    for i in range(n_fig):
        z_real = eigvec_w_real[:, i]
        z_imag = eigvec_w_imag[:, i]

        tol = 1e-6
        fntsize = 20

        if np.linalg.norm(z_real) > np.linalg.norm(z_imag):
            z = z_real
        else:
            z = z_imag

        minZ = min(z)
        maxZ = max(z)

        if minZ != maxZ:

            fig = plt.figure()

            ax = fig.add_subplot(111, projection='3d')

            ax.set_xbound(min(x) - tol, max(x) + tol)
            ax.set_xlabel('$x$', fontsize=fntsize)

            ax.set_ybound(min(y) - tol, max(y) + tol)
            ax.set_ylabel('$y$', fontsize=fntsize)

            ax.set_title('$v_{e_{p,w}}$', fontsize=fntsize)

            ax.set_zlim3d(minZ - 0.01*abs(minZ), maxZ + 0.01*abs(maxZ))
            ax.w_zaxis.set_major_locator(LinearLocator(10))
            ax.w_zaxis.set_major_formatter(FormatStrFormatter('%.2g'))

            ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0, antialiased=False)

            # path_out = "/home/a.brugnoli/PycharmProjects/Mindlin_Phs_fenics/Figures_Eig_Min/RealEig/"
            # plt.savefig(path_out1 + "Case" + case_study + "_el" + str(n) + "_deg" + str(deg) + "_thick_" + \
            #             str(thick) + "_eig_" + str(i+1) + ".eps", format="eps")

    plt.show()
