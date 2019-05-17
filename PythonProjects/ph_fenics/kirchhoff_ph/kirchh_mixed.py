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

n = 8
deg = 2 #int(input('Degree for FE: '))
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

norm_coeff = L ** 2 * np.sqrt(rho*h/D)
# Useful Matrices

D_b = as_tensor([
  [D, D * nu, 0],
  [D * nu, D, 0],
  [0, 0, D * (1 - nu) / 2]
])

fl_rot = 12. / (E * h ** 3)
C_b = as_tensor([
    [fl_rot, -nu * fl_rot, 0],
    [-nu * fl_rot, fl_rot, 0],
    [0, 0, fl_rot * (1 + nu) / 2]
])

# Operators and functions
def gradSym(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
    # return sym(nabla_grad(u))

def strain2voigt(eps):
    return as_vector([eps[0, 0], eps[1, 1], 2 * eps[0, 1]])

def voigt2stress(S):
    return as_tensor([[S[0], S[2]], [S[2], S[1]]])

def bending_moment(u):
    return voigt2stress(dot(D_b, strain2voigt(u)))

def bending_curv(u):
    return voigt2stress(dot(C_b, strain2voigt(u)))

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

Pp = FiniteElement('CG', mesh.ufl_cell(), deg)
Pq = TensorElement('CG', mesh.ufl_cell(), deg, symmetry=True)

element = MixedElement([Pp, Pq])
V = FunctionSpace(mesh, element)

v = TestFunction(V)
v_p, v_q = split(v)

e = TrialFunction(V)
e_p, e_q = split(e)

al_p = rho * h * e_p
al_q = bending_curv(e_q)

dx = Measure('dx')
ds = Measure('ds', subdomain_data=boundaries)

m_form = inner(v_p, al_p) * dx + inner(v_q, al_q) * dx

n_ver = FacetNormal(mesh)
s_ver = as_vector([-n_ver[1], n_ver[0]])

e_mnn = inner(e_q, outer(n_ver, n_ver))
v_mnn = inner(v_q, outer(n_ver, n_ver))

e_mns = inner(e_q, outer(n_ver, s_ver))
v_mns = inner(v_q, outer(n_ver, s_ver))

j_graddiv = dot(grad(v_p),  div(e_q)) * dx + v_p * dot(grad(e_mns), s_ver) * ds
j_divgrad = - dot(div(v_q), grad(e_p)) * dx - dot(grad(v_mns), s_ver) * e_p * ds

# j_graddiv = inner(grad(v_p),  div(e_q)) * dx
# j_divgrad = - inner(div(v_q), grad(e_p)) * dx

j_form = j_graddiv + j_divgrad

bc_1, bc_2, bc_3, bc_4 = bc_input

bc_dict = {1: bc_1, 2: bc_2, 3: bc_3, 4: bc_4}

Pqn = FiniteElement('CG', mesh.ufl_cell(), deg)
Pomn = FiniteElement('CG', mesh.ufl_cell(), deg)

element_u = MixedElement([Pqn, Pomn])

Vu = FunctionSpace(mesh, element_u)

q_n, om_n = TrialFunction(Vu)

b_vec = []
for key,val in bc_dict.items():
    if val == 'C':
        b_vec.append(v_p * q_n * ds(key))
    elif val == 'S':
        b_vec.append(v_p * q_n * ds(key) + v_mnn * om_n * ds(key))
    elif val == 'F':
        b_vec.append(v_mnn * om_n * ds(key))


J, M, B = PETScMatrix(), PETScMatrix(), PETScMatrix()
n_V = V.dim()

J = assemble(j_form)
M = assemble(m_form)
if b_vec:
    B = assemble(sum(b_vec))
    G = B.array()
    boundary_dofs = np.where(G.any(axis=0))[0]
    G = G[:, boundary_dofs]
    n_lmb = len(boundary_dofs)
else:
    G = np.empty((n_V, 0))
    n_lmb = 0

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