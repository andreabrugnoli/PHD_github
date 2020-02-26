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

n = 10 #int(input("Number of elements for side: "))
deg = 1 #int(input('Degree for FE: '))
nreq = 10

E = 1 #(7e10)
nu = (0.3)
thick = 'y' #input("Thick plate: ")
if thick == 'y':
    h = 0.1
else:
    h = 0.01

plot_eigenvector = 'y'

bc_input = input('Select Boundary Condition: ')   #'SSSS'


rho = 1 #(2000)  # kg/m^3
if (bc_input == 'CCCC' or  bc_input == 'CCCF'):
    k =  0.8601 # 5./6. #
elif bc_input == 'SSSS':
    k = 0.8333
elif bc_input == 'SCSC':
    k = 0.822
else: k =  0.8601

L = 1

D = E * h ** 3 / (1 - nu ** 2) / 12.
G = E / 2 / (1 + nu)
F = G * h * k

# I_w = 1. / (rho * h)
pho_rot = (rho * h ** 3)/12.

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

L = 1
l_x = L
l_y = L

n_x, n_y = n, n
mesh = RectangleMesh(Point(0, 0), Point(L, L), n_x, n_y, "right/left")

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

P_pw = FiniteElement('CG', triangle, deg)
P_pth = VectorElement('CG', triangle, deg)
P_qth = TensorElement('CG', triangle, deg, shape=(2, 2), symmetry=True)
P_qw = VectorElement('CG', triangle, deg) # FiniteElement('RT', triangle, deg) # 


element = MixedElement([P_pw, P_pth, P_qth, P_qw])
V = FunctionSpace(mesh, element)

v = TestFunction(V)
v_pw, v_pth, v_qth, v_qw = split(v)

e_v = TrialFunction(V)
e_pw, e_pth, e_qth, e_qw = split(e_v)

al_pw = rho * h * e_pw
al_pth = (rho * h ** 3) / 12. * e_pth
al_qth = bending_curv(e_qth)
al_qw = 1. / F * e_qw

dx = Measure('dx')
ds = Measure('ds', subdomain_data=boundaries)

m = inner(v_pw, al_pw) * dx + inner(v_pth, al_pth) * dx + inner(v_qth, al_qth) * dx + inner(v_qw, al_qw) * dx

j_div = v_pw * div(e_qw) * dx
j_divIP = -div(v_qw) * e_pw * dx

j_divSym = dot(v_pth, div(e_qth)) * dx
j_divSymIP = -dot(div(v_qth), e_pth) * dx

j_grad = dot(v_qw, grad(e_pw)) * dx
j_gradIP = -dot(grad(v_pw), e_qw) * dx

j_gradSym = inner(v_qth, gradSym(e_pth)) * dx
j_gradSymIP = -inner(gradSym(v_pth), e_qth) * dx

j_Id = dot(v_pth, e_qw) * dx
j_IdIP = -dot(v_qw, e_pth) * dx

j_alldiv = j_div + j_divIP + j_divSym + j_divSymIP + j_Id + j_IdIP
j_allgrad = j_grad + j_gradIP + j_gradSym + j_gradSymIP + j_Id + j_IdIP

j = j_allgrad

bc_1, bc_2, bc_3, bc_4 = bc_input

bc_dict = {1: bc_1, 2: bc_2, 3: bc_3, 4: bc_4}

n = FacetNormal(mesh)
s = as_vector([ -n[1], n[0] ])

P_qn = FiniteElement('CG', triangle, deg)
P_Mnn = FiniteElement('CG', triangle, deg)
P_Mns = FiniteElement('CG', triangle, deg)

element_u = MixedElement([P_qn, P_Mnn, P_Mns])

Vu = FunctionSpace(mesh, element_u)

q_n, M_nn, M_ns = TrialFunction(Vu)

v_omn = dot(v_pth, n)
v_oms = dot(v_pth, s)

b_vec = []
for key,val in bc_dict.items():
    if val == 'C':
        b_vec.append( v_pw * q_n * ds(key) + v_omn * M_nn * ds(key) + v_oms * M_ns * ds(key))
    elif val == 'S':
        b_vec.append(v_pw * q_n * ds(key) + v_oms * M_ns * ds(key))


b_u = sum(b_vec)

J, M, B = PETScMatrix(), PETScMatrix(), PETScMatrix()

J = assemble(j)
M = assemble(m)
B = assemble(b_u)

JJ = J.array()
MM = M.array()
B_in  = B.array()

boundary_dofs = np.where(B_in.any(axis=0))[0]
B_in = B_in[:,boundary_dofs]

N_al = V.dim()
N_u = len(boundary_dofs)
# print(N_u)

Z_u = np.zeros((N_u, N_u))

J_aug = np.vstack([ np.hstack([JJ, B_in]),
                    np.hstack([-B_in.T, Z_u])
                ])

Z_al_u = np.zeros((N_al, N_u))
Z_u_al = np.zeros((N_u, N_al))

M_aug = np.vstack([ np.hstack([MM, Z_al_u]),
                    np.hstack([Z_u_al,    Z_u])
                 ])
tol = 10**(-9)

eigenvalues, eigvectors = la.eig(J_aug, M_aug)
omega_all = np.imag(eigenvalues)

index = omega_all>=tol

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()

nconv = len(omega)

omega_tilde = omega*L*((2*(1+nu)*rho)/E)**0.5

for i in range(min(nconv, nreq)):
    print(omega_tilde[i])

d = mesh.geometry().dim()
dofV_x = V.tabulate_dof_coordinates().reshape((-1, d))

dofs_Vpw = V.sub(0).dofmap().dofs()
dofs_Vpth = V.sub(1).dofmap().dofs()
dofs_Vqth = V.sub(2).dofmap().dofs()
dofs_Vqw = V.sub(3).dofmap().dofs()

dofVpw_x = dofV_x[dofs_Vpw]

x = dofVpw_x[:,0]
y = dofVpw_x[:,1]

eigvec_w = eigvec_omega[dofs_Vpw, :]
eigvec_w_real = np.real(eigvec_w)
eigvec_w_imag = np.imag(eigvec_w)



n_fig = 5

if plot_eigenvector == 'y':

    for i in range(n_fig):
        z_real = eigvec_w_real[:, i]
        z_imag = eigvec_w_imag[:, i]

        tol = 1e-6
        fntsize = 20

        z1 = z_real
        minZ1 = min(z1)
        maxZ1 = max(z1)

        if minZ1 != maxZ1:

            fig1 = plt.figure(i)

            ax1 = fig1.add_subplot(111, projection='3d')
            # ax1.zaxis._axinfo['label']['space_factor'] = 20

            ax1.set_xbound(min(x) - tol, max(x) + tol)
            ax1.set_xlabel('$x$', fontsize=fntsize)

            ax1.set_ybound(min(y) - tol, max(y) + tol)
            ax1.set_ylabel('$y$', fontsize=fntsize)

            ax1.set_title('$v_{e_{p,w}}$', fontsize=fntsize)

            ax1.set_zlim3d(minZ1 - 0.01*abs(minZ1), maxZ1 + 0.01*abs(maxZ1))
            ax1.w_zaxis.set_major_locator(LinearLocator(10))
            ax1.w_zaxis.set_major_formatter(FormatStrFormatter('%.04f'))

            ax1.plot_trisurf(x, y, z1, cmap=cm.jet, linewidth=0, antialiased=False)

            # path_out1 = "/home/a.brugnoli/PycharmProjects/Mindlin_Phs_fenics/Figures_Eig_Min/RealEig/"
            # plt.savefig(path_out1 + "Case" + case_study + "_el" + str(n) + "_deg" + str(deg) + "_thick_" + \
            #             str(thick) + "_eig_" + str(i+1) + ".eps", format="eps")


    # for i in range(n_fig):
    #     z2 = z_imag
    #     minZ2 = min(z2)
    #     maxZ2 = max(z2)
    #
    #     if minZ2 != maxZ2:
    #
    #         fig2 = plt.figure(n_fig + i+1)
    #
    #         ax2 = fig2.add_subplot(111, projection='3d')
    #         # ax2.zaxis._axinfo['label']['space_factor'] = 20
    #
    #         ax2.set_xlim(min(x) - tol, max(x) + tol)
    #         ax2.set_xlabel('$x$', fontsize=fntsize)
    #
    #         ax2.set_ylim(min(y) - tol, max(y) + tol)
    #         ax2.set_ylabel('$y$', fontsize=fntsize)
    #
    #         # ax2.set_zlabel('$v_{e_{p,w}}$', fontsize=fntsize)
    #         ax2.set_title('$v_{e_{p,w}}$', fontsize=fntsize)
    #
    #         ax2.set_zlim(minZ2 - 0.01 * abs(minZ2), maxZ2 + 0.01 * abs(maxZ2))
    #         ax2.w_zaxis.set_major_locator(LinearLocator(10))
    #         ax2.w_zaxis.set_major_formatter(FormatStrFormatter('%.04f'))
    #
    #         ax2.plot_trisurf(x, y, z2, cmap=cm.jet, linewidth=0, antialiased=False)

            # path_out2 = "/home/a.brugnoli/PycharmProjects/Mindlin_Phs_fenics/Figures_Eig_Min/ImagEig/"
            # plt.savefig(path_out2 + "Case" + case_study + "_el" + str(n) + "_deg" + str(deg) + "_thick_" \
            #             + str(thick) + "_eig_" + str(i+1) + ".eps", format="eps")

    plt.show()
