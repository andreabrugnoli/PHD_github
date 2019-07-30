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
from mpl_toolkits.mplot3d import Axes3D
plt.rc('text', usetex=True)

deg = 1

rho = 3000
E = 7e10
nu = 0.3
h = 0.003
k = 5/6

ray = 0.1
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
mesh = Mesh("plate_hole.msh")
x, y = SpatialCoordinate(mesh)
plot(mesh)
plt.show()

# Finite element defition

Vp_w = FunctionSpace(mesh, "CG", deg)
Vp_th = VectorFunctionSpace(mesh, "CG", deg)
Vq_th = VectorFunctionSpace(mesh, "CG", deg, dim = 3)
Vq_w = VectorFunctionSpace(mesh, "CG", deg)

V = Vp_w * Vp_th * Vq_th * Vq_w

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

n = FacetNormal(mesh)
s = as_vector([-n[1], n[0] ])

Vf = FunctionSpace(mesh, 'CG', 1)
Vu = Vf * Vf * Vf

q_n, M_nn, M_ns = TrialFunction(Vu)
v_omn = dot(v_pth, n)
v_oms = dot(v_pth, s)
in_boundary = conditional(And(le(x, ray), le(y, ray)), 1.0, 0)

b_u = v_pw * q_n * in_boundary * ds + v_omn * M_nn * in_boundary * ds + v_oms * M_ns * in_boundary * ds

J = assemble(j_form, mat_type='aij')
M = assemble(m_form, mat_type='aij')
B = assemble(b_u, mat_type="aij")

petsc_j = J.M.handle
petsc_m = M.M.handle
petsc_b = B.M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())
B_in = np.array(petsc_b.convert("dense").getDenseArray())

boundary_dofs = np.where(B_in.any(axis=0))[0]
B_in = B_in[:, boundary_dofs]

N_al = V.dim()
N_u = len(boundary_dofs)
# print(N_u)

Z_u = np.zeros((N_u, N_u))

J_aug = np.vstack([ np.hstack([JJ, B_in]),
                    np.hstack([-B_in.T, Z_u])
                ])

Z_al_u = np.zeros((N_al, N_u))
Z_u_al = np.zeros((N_u, N_al))

M_aug = np.vstack([np.hstack([MM, Z_al_u]),
                   np.hstack([Z_u_al,    Z_u])
                 ])
tol = 10**(-9)

eigenvalues, eigvectors = la.eig(J_aug, M_aug)
omega_all = np.imag(eigenvalues)

index = omega_all > tol

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()


omega_fr = omega/(2*pi)

for i in range(5):
    print(omega_fr[i])

n_fig = 5

n_Vpw = Vp_w.dim()
fntsize = 15
for i in range(n_fig):
    # print("Eigenvalue num " + str(i + 1) + ":" + str(omega[i]))
    eig_real_w = Function(Vp_w)
    eig_imag_w = Function(Vp_w)

    eig_real_pw = np.real(eigvec_omega[:n_Vpw, i])
    eig_imag_pw = np.imag(eigvec_omega[:n_Vpw, i])
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