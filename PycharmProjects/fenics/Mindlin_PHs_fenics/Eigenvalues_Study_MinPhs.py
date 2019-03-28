# Mindlin plate written with the port Hamiltonian approach

from fenics import *
import numpy as np
np.set_printoptions(threshold=np.inf)
import mshr
import matplotlib.pyplot as plt

import scipy.linalg as la

n = 6 #int(input("Number of elements for side: "))
deg = 2 #int(input('Degree for FE: '))
nreq = 10

E = 1 #(7e10)
nu = (0.3)
thick = 'y' #input("Thick plate: ")
if thick == 'y':
    h = 0.1
else:
    h = 0.01

case_study = input("Select the case under study: ") # 'CCCC' #

# save_eigenvector = 'n' #input('Plot Eigenvector: ')
plot_eigenvector = 'y'

rho = 1 #(2000)  # kg/m^3
if (case_study == 'CCCC' or  case_study == 'CCCF'):
    k =  0.8601 # 5./6. #
elif case_study == 'SSSS':
    k = 0.8333
elif case_study == 'SCSC':
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
class ClampedSide(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],0.0) and on_boundary

class FreeSide(SubDomain):
    def inside(self, x, on_boundary):
        return (abs(x[0]-l_x)< DOLFIN_EPS or abs(x[1])<DOLFIN_EPS or abs(x[1]-l_y)<DOLFIN_EPS) and on_boundary


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

class AllBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# Boundary conditions on displacement
free_side = FreeSide()
clamped_side = ClampedSide()
all_boundary = AllBoundary()
# Boundary conditions on rotations
left = Left()
right = Right()
lower = Lower()
upper = Upper()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
right.mark(boundaries, 1)
upper.mark(boundaries, 2)
right.mark(boundaries, 3)
lower.mark(boundaries, 4)

# Finite element defition

P_pw = FiniteElement('CG', triangle, deg)
P_pth = VectorElement('CG', triangle, deg)
P_qth = TensorElement('DG', triangle, deg-1, shape=(2, 2), symmetry=True)
P_qw = VectorElement('DG', triangle, deg-1)


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

if deg <= 1:
    dx_shear = dx(metadata={"quadrature_degree": 2*deg-2})
else: dx_shear = dx

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
bcs_p = []
bcs_q = []


if case_study == 'CCCC':
    bc_w = DirichletBC(V.sub(0), Constant(0.0), all_boundary)
    bc_th = DirichletBC(V.sub(1), (Constant(0.0), Constant(0.0)), all_boundary)

    bcs_p = [bc_w, bc_th]

if case_study == 'SSSS':
    bc_w = DirichletBC(V.sub(0), Constant(0.0), all_boundary)

    bc_ths_l = DirichletBC(V.sub(1).sub(1), Constant(0.0), left)
    bc_ths_r = DirichletBC(V.sub(1).sub(1), Constant(0.0), right)
    bc_ths_d = DirichletBC(V.sub(1).sub(0), Constant(0.0), lower)
    bc_ths_u = DirichletBC(V.sub(1).sub(0), Constant(0.0), upper)

    # bc_Mnn_l = DirichletBC(V.sub(2).sub(0), Constant(0.0), left)
    # bc_Mnn_r = DirichletBC(V.sub(2).sub(0), Constant(0.0), right)
    # bc_Mnn_d = DirichletBC(V.sub(2).sub(2), Constant(0.0), lower)
    # bc_Mnn_u = DirichletBC(V.sub(2).sub(2), Constant(0.0), upper)

    bcs_p = [bc_w, bc_ths_l, bc_ths_r, bc_ths_d, bc_ths_u]
    # bcs_q = [bc_Mnn_l, bc_Mnn_r, bc_Mnn_d, bc_Mnn_u]

if case_study =='SCSC':
    bc_w = DirichletBC(V.sub(0), Constant(0.0), all_boundary)

    bc_ths_l = DirichletBC(V.sub(1).sub(1), Constant(0.0), left)
    bc_ths_r = DirichletBC(V.sub(1).sub(1), Constant(0.0), right)
    bc_th_d = DirichletBC(V.sub(1), (Constant(0.0), Constant(0.0)), lower)
    bc_th_u = DirichletBC(V.sub(1), (Constant(0.0), Constant(0.0)), upper)

    # bc_Mnn_l = DirichletBC(V.sub(2).sub(0), Constant(0.0), left)
    # bc_Mnn_r = DirichletBC(V.sub(2).sub(0), Constant(0.0), right)

    bcs_p = [bc_w, bc_ths_l, bc_ths_r, bc_th_d, bc_th_u]
    # bcs_q = [bc_Mnn_l, bc_Mnn_r]

if case_study == 'CCCF':

    bc_w_l = DirichletBC(V.sub(0), Constant(0.0), left)
    bc_w_d = DirichletBC(V.sub(0), Constant(0.0), lower)
    bc_w_u = DirichletBC(V.sub(0), Constant(0.0), upper)


    bc_th_l = DirichletBC(V.sub(1), (Constant(0.0), Constant(0.0) ), left)
    bc_th_d = DirichletBC(V.sub(1), (Constant(0.0), Constant(0.0) ), lower)
    bc_th_u = DirichletBC(V.sub(1), (Constant(0.0), Constant(0.0) ), upper)

    # bc_Mxx_r = DirichletBC(V.sub(2).sub(0), Constant(0.0), right)
    # bc_Mxy_r = DirichletBC(V.sub(2).sub(1), Constant(0.0), right)
    # bc_Qx_r = DirichletBC(V.sub(3).sub(0), Constant(0.0) , right)


    bcs_p = [bc_w_l, bc_w_d, bc_w_u, bc_th_l, bc_th_d, bc_th_u]
    # bcs_q = [bc_Mxx_r, bc_Mxy_r, bc_Qx_r]

if case_study == 'CFFF':
    bc_w = DirichletBC(V.sub(0), Constant(0.0), clamped_side)
    bc_th = DirichletBC(V.sub(1), (Constant(0.0), Constant(0.0)), clamped_side)

    # bc_Mxx_r = DirichletBC(V.sub(2).sub(0), Constant(0.0), right)
    # bc_Myy_b = DirichletBC(V.sub(2).sub(2), Constant(0.0), lower)
    # bc_Myy_t = DirichletBC(V.sub(2).sub(2), Constant(0.0), upper)
    # bc_Mxy = DirichletBC(V.sub(2).sub(1), Constant(0.0), free_side)
    #
    # bc_Qx_r = DirichletBC(V.sub(3).sub(0), Constant(0.0), right)
    # bc_Qy_b = DirichletBC(V.sub(3).sub(1), Constant(0.0), lower)
    # bc_Qy_t = DirichletBC(V.sub(3).sub(1), Constant(0.0), upper)

    bcs_p = [bc_w, bc_th]
    # bcs_q = [bc_Mxx_r, bc_Mxy, bc_Myy_b, bc_Myy_t, bc_Qx_r, bc_Qy_b, bc_Qy_t]


bcs = bcs_p
# if not bcs:
#     raise ValueError("Empty bcs")
# Assemble the stiffness matrix and the mass matrix.
J, M = PETScMatrix(), PETScMatrix()
b = PETScVector()

f = inner(Constant(1), v_pw)*dx

assemble_system(j, f, bcs=bcs, A_tensor=J, b_tensor=b)
assemble_system(m, f, bcs=bcs, A_tensor=M, b_tensor=b)
[bc.zero(J) for bc in bcs]

# a_lumped = 0
# alpha_vec = [al_pw, al_pth[0], al_pth[1], al_qth[0,0],  al_qth[0,1], al_qth[1,0],  al_qth[1,1], al_qw[0], al_qw[1]]
# v_vec = [v_pw, v_pth[0], v_pth[1], v_qth[0,0],  v_qth[0,1], v_qth[1,0],  v_qth[1,1], v_qw[0], v_qw[1]]
# N = len(alpha_vec)
# for i in range(0, N):
#     a_lumped += alpha_vec[i]*v_vec[i]*dx
#
# # Class for mass matrix lumping
# class LumpingClass(Expression):
#     def eval(self, values, x):
#         for i in range(0, N):
#            values[i] = 1.0
#     def value_shape(self):
#         return (N,)
#
# const = LumpingClass(degree=2)
# M_lumped = PETScMatrix()
# mass_action_form = action(a_lumped, const)
# assemble_system(a_lumped, f, bcs=bcs, A_tensor=M_lumped, b_tensor=b)
# M_lumped.zero()
# M_lumped.set_diagonal(assemble(mass_action_form)) # M_lumped is now lumped


tol = 10**(-6)

eigenvalues, eigvectors = la.eig(J.array(), M.array())
omega_all = np.imag(eigenvalues)

index = omega_all>tol

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

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

plt.close('all')
matplotlib.rcParams['text.usetex'] = True

n_fig = 4

if plot_eigenvector == 'y':

    for i in range(n_fig):
        z_real = eigvec_w_real[:, i]
        z_imag = eigvec_w_imag[:, i]

        tol = 1e-6
        fntsize = 20

        if matplotlib.is_interactive():
            plt.ioff()


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
    #
    #
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




