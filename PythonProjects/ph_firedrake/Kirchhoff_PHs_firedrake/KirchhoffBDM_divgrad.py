# Mindlin plate written with the port Hamiltonian approach
# with weak symmetry
from firedrake import *
import numpy as np
import scipy.linalg as la
from modules_phdae.classes_phsystem import SysPhdaeRig, check_positive_matrix
import warnings
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from firedrake.plot import _two_dimension_triangle_func_val
from mpl_toolkits.mplot3d import Axes3D
from math import pi
plt.rc('text', usetex=True)


n_el = 5 #int(input("Number of elements for side: "))
deg = 2 #int(input('Degree for FE: '))

nreq = 5

E = 1
nu = 0.3

rho = 1
L = 1
h = 0.1

plot_eigenvector = 'y'

bc_input = input('Select Boundary Condition: ')   #'SSSS'

# Useful Matrices
D = E * h ** 3 / (1 - nu ** 2) / 12.
fl_rot = 12 / (E * h ** 3)

# The unit square mesh is divided in :math:`N\times N` quadrilaterals::

L = 1

n_x, n_y = n_el, n_el
L_x, L_y = L, L
mesh = RectangleMesh(n_x, n_y, L_x, L_y, quadrilateral=False)

# domain = mshr.Rectangle(Point(0, 0), Point(l_x, l_y))
# mesh = mshr.generate_mesh(domain, n, "cgal")

# plot(mesh)
# plt.show()

# Operators and functions
def gradSym(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)

def bending_moment(kappa):
    momenta = D * ((1-nu) * kappa + nu * Identity(2) * tr(kappa))
    return momenta

def bending_curv(momenta):
    kappa = fl_rot * ((1+nu)*momenta - nu * Identity(2) * tr(momenta))
    return kappa

# Finite element defition


V_p = FunctionSpace(mesh, "CG", deg)
V_sk = FunctionSpace(mesh, "CG", deg)
V_q1 = FunctionSpace(mesh, "RT", deg)
V_q2 = FunctionSpace(mesh, "RT", deg)

V = MixedFunctionSpace([V_p, V_sk, V_q1, V_q2])

n_V = V.dim()
print(n_V)

v = TestFunction(V)
v_p, v_sk, v_q1, v_q2 = split(v)

e = TrialFunction(V)
e_p, e_sk, e_q1, e_q2 = split(e)

v_q = as_tensor([[v_q1[0], v_q1[1]],
                    [v_q2[0], v_q2[1]]
                   ])

e_q = as_tensor([[e_q1[0], e_q1[1]],
                    [e_q2[0], e_q2[1]]
                   ])

al_p = rho * h * e_p
al_q = bending_curv(e_q)

v_sk = as_tensor([[0, v_sk],
                    [-v_sk, 0]])

al_sk = as_tensor([[0, e_sk],
                    [-e_sk, 0]])

# v_skw = skew(v_skw)
# al_skw = skew(e_skw)

dx = Measure('dx')
ds = Measure('ds')

m_form = v_p * al_p * dx \
    + inner(v_q, al_q) * dx + inner(v_q, al_sk) * dx \
    + inner(v_sk, e_q) * dx

n_ver = FacetNormal(mesh)
s_ver = as_vector([-n_ver[1], n_ver[0]])

e_mnn = inner(e_q, outer(n_ver, n_ver))
v_mnn = inner(v_q, outer(n_ver, n_ver))

e_mns = inner(e_q, outer(n_ver, s_ver))
v_mns = inner(v_q, outer(n_ver, s_ver))

j_graddiv = dot(grad(v_p),  div(e_q)) * dx + v_p * dot(grad(e_mns), s_ver) * ds
j_divgrad = - dot(div(v_q), grad(e_p)) * dx - dot(grad(v_mns), s_ver) * e_p * ds

j_form = j_graddiv + j_divgrad

# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1

bc_1, bc_2, bc_3, bc_4 = bc_input

bc_dict = {1: bc_1, 3: bc_2, 2: bc_3, 4: bc_4}

V_qn = FunctionSpace(mesh, "CG", 1)
V_omn = FunctionSpace(mesh, "CG", 1)

Vu = MixedFunctionSpace([V_qn, V_omn])

q_n, om_n = TrialFunction(Vu)

b_vec = []
for key,val in bc_dict.items():
    if val == 'C':
        b_vec.append(v_p * q_n * ds(key))
    elif val == 'S':
        b_vec.append(v_p * q_n * ds(key) + v_mnn * om_n * ds(key))
    elif val == 'F':
        b_vec.append(v_mnn * om_n * ds(key))

b_u = sum(b_vec)

M = assemble(m_form, mat_type='aij')
J = assemble(j_form, mat_type='aij')


petsc_j = J.M.handle
petsc_m = M.M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())

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

Z_u = np.zeros((n_u, n_u))

J_aug = np.vstack([np.hstack([JJ, B_in]),
                   np.hstack([-B_in.T, Z_u])
                ])

M_aug = la.block_diag(MM, Z_u)
tol = 10**(-6)

# plt.spy(M_aug); plt.show()
# plt.spy(J_aug); plt.show()

eigenvalues, eigvectors = la.eig(J_aug, M_aug)
omega_all = np.imag(eigenvalues)

index = omega_all >= tol

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()
norm_coeff = L ** 2 * np.sqrt(rho*h/D)

omega_tilde = omega * norm_coeff

for i in range(nreq):
    print(omega_tilde[i])

n_fig = nreq


n_Vpw = V_p.dim()
fntsize = 15
for i in range(n_fig):
    print("Eigenvalue num " + str(i + 1) + ":" + str(omega_tilde[i]))
    eig_real_w = Function(V_p)
    eig_imag_w = Function(V_p)

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