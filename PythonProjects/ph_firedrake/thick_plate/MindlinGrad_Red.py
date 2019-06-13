# Mindlin plate written with the port Hamiltonian approach
from firedrake import *
import numpy as np
import scipy.linalg as la
import warnings
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from firedrake.plot import _two_dimension_triangle_func_val
from modules_phdae.classes_phsystem import SysPhdaeRig
from mpl_toolkits.mplot3d import Axes3D
plt.rc('text', usetex=True)

n = 5
deg = 2

pho = 1
E = 1
nu = 0.3
thick = 'y'
if thick == 'y':
    h = 0.1
else:
    h = 0.01

plot_eigenvector = True

bc_input = input('Select Boundary Condition: ')
rho = 1 #(2000)  # kg/m^3
if bc_input == 'CCCC' or  bc_input == 'CCCF':
    k = 0.8601 # 5./6. #
elif bc_input == 'SSSS':
    k = 0.8333
elif bc_input == 'SCSC':
    k = 0.822
else: k = 0.8601

L = 1

D = E * h ** 3 / (1 - nu ** 2) / 12.
G = E / 2 / (1 + nu)
F = G * h * k

# Useful Matrices

fl_rot = 12. / (E * h ** 3)

C_b_vec = as_tensor([
    [fl_rot, -nu * fl_rot, 0],
    [-nu * fl_rot, fl_rot, 0],
    [0, 0, fl_rot * 2 * (1 + nu)]
])


# Finite element defition
def bending_curv_vec(MM):
    return dot(C_b_vec, MM)

def gradSym_vec(u):
    return as_vector([ u[0].dx(0),  u[1].dx(1), u[0].dx(1) + u[1].dx(0)])


# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
L = 1
l_x = L
l_y = L
n_x, n_y = n, n
mesh = UnitSquareMesh(n_x, n_y)

# plot(mesh)
# plt.show()

# Finite element defition

V_pw = FunctionSpace(mesh, "CG", deg)
V_pth = VectorFunctionSpace(mesh, "CG", deg)
V_qth = VectorFunctionSpace(mesh, "CG", deg, dim=3)
V_qw = VectorFunctionSpace(mesh, "CG", deg)

V = V_pw * V_pth * V_qth * V_qw

n_V = V.dim()
n_Vp = V_pw.dim() + V_pth.dim()
n_Vq = V_qw.dim() + V_qth.dim()

v = TestFunction(V)
v_pw, v_pth, v_qth, v_qw = split(v)

e_v = TrialFunction(V)
e_pw, e_pth, e_qth, e_qw = split(e_v)

al_pw = rho * h * e_pw
al_pth = (rho * h ** 3) / 12. * e_pth
al_qth = bending_curv_vec(e_qth)
al_qw = 1. / F * e_qw


dx = Measure('dx')
ds = Measure('ds')
m_form = inner(v_pw, al_pw) * dx + inner(v_pth, al_pth) * dx + inner(v_qth, al_qth) * dx + inner(v_qw, al_qw) * dx

j_grad = dot(v_qw, grad(e_pw)) * dx
j_gradIP = -dot(grad(v_pw), e_qw) * dx

j_gradSym = inner(v_qth, gradSym_vec(e_pth)) * dx
j_gradSymIP = -inner(gradSym_vec(v_pth), e_qth) * dx

j_Id = dot(v_pth, e_qw) * dx
j_IdIP = -dot(v_qw, e_pth) * dx

j_allgrad = j_grad + j_gradIP + j_gradSym + j_gradSymIP + j_Id + j_IdIP

j_form = j_allgrad

J = assemble(j_form, mat_type='aij')
M = assemble(m_form, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())


# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1

bc_1, bc_2, bc_3, bc_4 = bc_input

bc_dict = {1: bc_1, 2: bc_3, 3: bc_2, 4: bc_4}

n = FacetNormal(mesh)
s = as_vector([-n[1], n[0] ])

Vf = FunctionSpace(mesh, 'CG', 1)
Vu = Vf * Vf * Vf

q_n, M_nn, M_ns = TrialFunction(Vu)

v_omn = dot(v_pth, n)
v_oms = dot(v_pth, s)

b_vec = []
for key,val in bc_dict.items():
    if val == 'C':
        b_vec.append(v_pw * q_n * ds(key) + v_omn * M_nn * ds(key) + v_oms * M_ns * ds(key))
    elif val == 'S':
        b_vec.append(v_pw * q_n * ds(key) + v_oms * M_ns * ds(key))

b_u = sum(b_vec)


if b_vec:
    B = assemble(b_u, mat_type="aij")
    petsc_b = B.M.handle
    B_in = np.array(petsc_b.convert("dense").getDenseArray())
    boundary_dofs = np.where(B_in.any(axis=0))[0]
    B_in = B_in[:, boundary_dofs]
    n_u = len(boundary_dofs)

else:
    n_u = 0
    B_in = np.zeros((n_V, n_u))

# print(N_u)

tol = 10**(-9)

x, y = SpatialCoordinate(mesh)
con_dom1 = And(And(gt(x, L/4), lt(x, 3*L/4)), And(gt(y, L/4), lt(y, 3*L/4)))
Dom_f = conditional(con_dom1, 1., 0.)
B_f = assemble(v_pw * Dom_f * dx).vector().get_local()

Z_u = np.zeros((n_u, n_u))

J_aug = np.vstack([np.hstack([JJ, B_in]),
                   np.hstack([-B_in.T, Z_u])
                   ])

E_aug = la.block_diag(MM, Z_u)
B_aug = np.concatenate((B_f, np.zeros(n_u, )), axis=0).reshape((-1, 1))

plate_full = SysPhdaeRig(n_V+n_u, n_u, 0, n_Vp, n_Vq, E_aug, J_aug, B_aug)
# plate_full = SysPhdaeRig(n_V, 0, 0, n_Vp, n_Vq, MM, JJ, B_in)

s0 = 0.01
n_red = 100
plate_red, V_red = plate_full.reduce_system(s0, n_red)
Vall_red = la.block_diag(V_red, np.eye(n_u))

E_red = plate_red.E
J_red = plate_red.J
B_red = plate_red.B

# print(np.linalg.matrix_rank(B_in), n_u)

# J_red = np.vstack([np.hstack([J_red, B_red]),
#                       np.hstack([-B_red.T, Z_u])
#                       ])
#
# E_red = la.block_diag(E_red, Z_u)

tol = 10 ** (-6)

eigenvaluesF, eigvectorsF = la.eig(J_aug, E_aug)
omega_allF = np.imag(eigenvaluesF)

indexF = omega_allF >= tol

omega_full = omega_allF[indexF]
eigvec_full = eigvectorsF[:, indexF]
permF = np.argsort(omega_full)
eigvec_full = eigvec_full[:, permF]
omega_full.sort()

eigenvaluesR, eigvectorsR = la.eig(J_red, E_red)
omega_allR = np.imag(eigenvaluesR)

index = omega_allR >= tol

omega_red = omega_allR[index]
eigvec_red = eigvectorsR[:, index]
permR = np.argsort(omega_red)
eigvec_red = eigvec_red[:, permR]
omega_red.sort()

plt.plot(np.real(eigenvaluesF), np.imag(eigenvaluesF), 'r+', np.real(eigenvaluesR), np.imag(eigenvaluesR), 'bo')
plt.legend(("Eigenvalues full", "Eigenvalues reduced"))
plt.show()

# NonDimensional China Paper

n_om = 5

omegaF_tilde = omega_full*L*((2*(1+nu)*rho)/E)**0.5
omegaR_tilde = omega_red*L*((2*(1+nu)*rho)/E)**0.5

for i in range(n_om):
    print(omegaF_tilde[i])

for i in range(n_om):
    print(omegaF_tilde[i], omegaR_tilde[i])

n_fig = 5

n_Vpw = V_pw.dim()
fntsize = 15
if plot_eigenvector:

    # for i in range(n_fig):
    #     eig_real_w = Function(V_pw)
    #     eig_imag_w = Function(V_pw)
    #
    #     eig_real_pw = np.real(eigvec_full[:n_Vpw, i])
    #     eig_imag_pw = np.imag(eigvec_full[:n_Vpw, i])
    #     eig_real_w.vector()[:] = eig_real_pw
    #     eig_imag_w.vector()[:] = eig_imag_pw
    #
    #     norm_real_eig = np.linalg.norm(eig_real_w.vector().get_local())
    #     norm_imag_eig = np.linalg.norm(eig_imag_w.vector().get_local())
    #
    #     if norm_imag_eig > norm_real_eig:
    #         triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_imag_w, 10)
    #     else:
    #         triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_real_w, 10)
    #
    #     figure = plt.figure()
    #     ax = figure.add_subplot(111, projection="3d")
    #
    #     ax.plot_trisurf(triangulation, z_goodeig, cmap=cm.jet)
    #
    #     ax.set_xlabel('$x [m]$', fontsize=fntsize)
    #     ax.set_ylabel('$y [m]$', fontsize=fntsize)
    #     ax.set_title('Eigenvector num ' + str(i + 1), fontsize=fntsize)
    #
    #     ax.w_zaxis.set_major_locator(LinearLocator(10))
    #     ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

    for i in range(n_fig):
        eig_real_w = Function(V_pw)
        eig_imag_w = Function(V_pw)

        eig_real_pw = np.real((Vall_red @ eigvec_red[:, i])[:n_Vpw])
        eig_imag_pw = np.imag((Vall_red @ eigvec_red[:, i])[:n_Vpw])
        eig_real_w.vector()[:] = eig_real_pw
        eig_imag_w.vector()[:] = eig_imag_pw

        norm_real_eig = np.linalg.norm(eig_real_w.vector().get_local())
        norm_imag_eig = np.linalg.norm(eig_imag_w.vector().get_local())

        if norm_imag_eig > norm_real_eig:
            triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_imag_w, 10)
        else:
            triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_real_w, 10)

        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")

        ax.plot_trisurf(triangulation, z_goodeig, cmap=cm.jet)

        ax.set_xlabel('$x [m]$', fontsize=fntsize)
        ax.set_ylabel('$y [m]$', fontsize=fntsize)
        ax.set_title('Eigenvector num ' + str(i + 1), fontsize=fntsize)

        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

plt.show()