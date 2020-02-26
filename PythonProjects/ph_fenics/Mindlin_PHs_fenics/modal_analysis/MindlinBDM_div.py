# Mindlin plate written with the port Hamiltonian approach
# with weak symmetry

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

n = 5 #int(input("Number of elements for side: "))
deg = 0 #int(input('Degree for FE: '))
nreq = 10

E = 1
nu = 0.3

rho = 1
k = 0.8601
L = 1
h = 0.1

plot_eigenvector = 'y'

bc_input = input('Select Boundary Condition: ')   #'SSSS'

G = E / 2 / (1 + nu)
F = G * h * k


# Useful Matrices
D = E * h ** 3 / (1 - nu ** 2) / 12.
fl_rot = 12. / (E * h ** 3)

# The unit square mesh is divided in :math:`N\times N` quadrilaterals::

L = 1
l_x = L
l_y = L

n_x, n_y = n, n
mesh = RectangleMesh(Point(0, 0), Point(L, L), n_x, n_y, "right/left")
d = mesh.geometry().dim()


# domain = mshr.Rectangle(Point(0, 0), Point(l_x, l_y))
# mesh = mshr.generate_mesh(domain, n, "cgal")

# plot(mesh)
# plt.show()

# Operators and functions
def gradSym(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
    # return sym(nabla_grad(u))

def bending_moment(kappa):
    momenta = D * ((1-nu) * kappa + nu * Identity(d) * tr(kappa))
    return momenta

def bending_curv(momenta):
    kappa = fl_rot * ((1+nu)*momenta - nu * Identity(d) * tr(momenta))
    return kappa

# Domain, Subdomains, Boundary, Suboundaries
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 0.0) < DOLFIN_EPS and on_boundary

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - l_x) < DOLFIN_EPS and on_boundary

class Lower(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1] - 0.0) < DOLFIN_EPS and on_boundary

class Upper(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1] - l_y) < DOLFIN_EPS and on_boundary

# Boundary conditions on rotations
left = Left()
right = Right()
lower = Lower()
upper = Upper()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(5)
left.mark(boundaries, 1)
lower.mark(boundaries, 2)
right.mark(boundaries, 3)
upper.mark(boundaries, 4)


# Finite element defition

P_pw = FiniteElement('DG', triangle, deg)
P_skw = FiniteElement('DG', triangle, deg)
P_pth = VectorElement('DG', triangle, deg)
P_qth = VectorElement('BDM', triangle, deg + 1)
P_qw = FiniteElement('BDM', triangle, deg + 1)
# P_qth = VectorElement('RT', triangle, 1)
# P_qw = FiniteElement('RT', triangle, 1)


element = MixedElement([P_pw, P_skw, P_pth, P_qth, P_qw])
V = FunctionSpace(mesh, element)

n_V = V.dim()
print(n_V)

v = TestFunction(V)
v_pw, v_skw, v_pth, v_qth, v_qw = split(v)

e = TrialFunction(V)
e_pw, e_skw, e_pth, e_qth, e_qw = split(e)

al_pw = rho * h * e_pw
al_pth = (rho * h ** 3) / 12. * e_pth
al_qth = bending_curv(e_qth)
al_qw = 1. / F * e_qw

v_skw = as_tensor([[0, v_skw],
                    [-v_skw, 0]])
al_skw = as_tensor([[0, e_skw],
                    [-e_skw, 0]])

# v_skw = skew(v_skw)
# al_skw = skew(e_skw)


dx = Measure('dx')
ds = Measure('ds', subdomain_data=boundaries)

m = v_pw * al_pw * dx \
    + dot(v_pth, al_pth) * dx \
    + inner(v_qth, al_qth) * dx + inner(v_qth, al_skw) * dx \
    + dot(v_qw, al_qw) * dx \
    + inner(v_skw, e_qth) * dx

j_div = v_pw * div(e_qw) * dx
j_divIP = -div(v_qw) * e_pw * dx

j_divSym = dot(v_pth, div(e_qth)) * dx
j_divSymIP = -dot(div(v_qth), e_pth) * dx

# j_grad = dot(v_qw, grad(e_pw)) * dx
# j_gradIP = -dot(grad(v_pw), e_qw) * dx
#
# j_gradSym = inner(v_qth, gradSym(e_pth)) * dx
# j_gradSymIP = -inner(gradSym(v_pth), e_qth) * dx

j_Id = dot(v_pth, e_qw) * dx
j_IdIP = -dot(v_qw, e_pth) * dx

j_alldiv = j_div + j_divIP + j_divSym + j_divSymIP + j_Id + j_IdIP
# j_allgrad = j_grad + j_gradIP + j_gradSym + j_gradSymIP + j_Id + j_IdIP

j = j_alldiv

bc_1, bc_2, bc_3, bc_4 = bc_input

bc_dict = {1: bc_1, 2: bc_2, 3: bc_3, 4: bc_4}


n = FacetNormal(mesh)
s = as_vector([ -n[1], n[0] ])

v_qn = dot(v_qw, n)
v_Mnn = inner(v_qth, outer(n, n))
v_Mns = inner(v_qth, outer(n, s))

w_t = e_pw
om_n = dot(e_pth, n)
om_s = dot(e_pth, s)

b_vec = []
for key,val in bc_dict.items():
    if val == 'F':
        b_vec.append( v_qn * w_t * ds(key) + v_Mnn * om_n * ds(key) + v_Mns * om_s * ds(key))
    elif val == 'S':
        b_vec.append(v_Mnn * om_n * ds(key) )

b_u = sum(b_vec)


J, M, B = PETScMatrix(), PETScMatrix(), PETScMatrix()

J = assemble(j)
M = assemble(m)

if b_vec:
    B = assemble(b_u)
    B_in = B.array()
    boundary_dofs = np.where(B_in.any(axis=0))[0]
    B_in = B_in[:, boundary_dofs]
    n_u = len(boundary_dofs)

else:
    n_u = 0
    B_in = np.zeros((n_V, n_u))

JJ = J.array()
MM = M.array()

Z_u = np.zeros((n_u, n_u))


J_aug = np.vstack([ np.hstack([JJ, B_in]),
                    np.hstack([-B_in.T, Z_u])
                ])
M_aug = la.block_diag(MM, Z_u)
tol = 10**(-6)

# plt.spy(J_aug); plt.show()

eigenvalues, eigvectors = la.eig(J_aug, M_aug)
omega_all = np.imag(eigenvalues)

index = omega_all >= tol

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()

omega_tilde = omega*L*((2*(1+nu)*rho)/E)**0.5

for i in range(nreq):
    print(omega_tilde[i])

dofV_x = V.tabulate_dof_coordinates().reshape((-1, d))

dofs_Vpw = V.sub(0).dofmap().dofs()
dofs_Vpth = V.sub(1).dofmap().dofs()
dofs_Vqth = V.sub(2).dofmap().dofs()
dofs_Vqw = V.sub(3).dofmap().dofs()

dofVpw_x = dofV_x[dofs_Vpw]

x = dofVpw_x[:, 0]
y = dofVpw_x[:, 1]

eigvec_w = eigvec_omega[dofs_Vpw, :]
eigvec_w_real = np.real(eigvec_w)
eigvec_w_imag = np.imag(eigvec_w)

n_fig = nreq

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