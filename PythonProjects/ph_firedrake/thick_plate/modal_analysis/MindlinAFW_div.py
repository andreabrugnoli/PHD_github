# Mindlin plate written with the port Hamiltonian approach
# with weak symmetry
from firedrake import *
import numpy as np
import scipy.linalg as la
from modules_ph.classes_phsystem import SysPhdaeRig, check_positive_matrix
import warnings
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from firedrake.plot import _two_dimension_triangle_func_val
from mpl_toolkits.mplot3d import Axes3D
from math import pi
plt.rc('text', usetex=True)


n_el = 7  #int(input("Number of elements for side: "))
deg = 2  #int(input('Degree for FE: '))
nreq = 5

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
    # return sym(nabla_grad(u))

def bending_moment(kappa):
    momenta = D * ((1-nu) * kappa + nu * Identity(2) * tr(kappa))
    return momenta

def bending_curv(momenta):
    kappa = fl_rot * ((1+nu)*momenta - nu * Identity(2) * tr(momenta))
    return kappa

# Finite element defition


V_pw = FunctionSpace(mesh, "DG", deg-1)
V_skw = FunctionSpace(mesh, "DG", deg-1)
V_pth = VectorFunctionSpace(mesh, "DG", deg-1)

# V_qth1 = FunctionSpace(mesh, "BDM", deg)
# V_qth2 = FunctionSpace(mesh, "BDM", deg)
# V_qw = FunctionSpace(mesh, "BDM", deg)

V_qth1 = FunctionSpace(mesh, "BDM", deg)
V_qth2 = FunctionSpace(mesh, "BDM", deg)
V_qw = FunctionSpace(mesh, "RT", deg)

# V_qth1 = FunctionSpace(mesh, "RT", deg+1)
# V_qth2 = FunctionSpace(mesh, "RT", deg+1)
# V_qw = FunctionSpace(mesh, "RT", deg+1)

V = MixedFunctionSpace([V_pw, V_skw, V_pth, V_qth1, V_qth2, V_qw])

n_V = V.dim()
print(n_V)

v = TestFunction(V)
v_pw, v_skw, v_pth, v_qth1, v_qth2, v_qw = split(v)

e = TrialFunction(V)
e_pw, e_skw, e_pth, e_qth1, e_qth2, e_qw = split(e)

v_qth = as_tensor([[v_qth1[0], v_qth1[1]],
                    [v_qth2[0], v_qth2[1]]
                   ])

e_qth = as_tensor([[e_qth1[0], e_qth1[1]],
                    [e_qth2[0], e_qth2[1]]
                   ])

# v_qth = as_tensor([[v_qth1[0], v_qth2[0]],
#                    [v_qth1[1], v_qth2[1]]
#                    ])
#
# e_qth = as_tensor([[e_qth1[0], e_qth2[0]],
#                    [e_qth1[1], e_qth2[1]]
#                    ])

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
ds = Measure('ds')

m_form = v_pw * al_pw * dx \
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

j_form = j_alldiv

# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1

bc_1, bc_2, bc_3, bc_4 = bc_input

bc_dict = {1: bc_1, 3: bc_2, 2: bc_3, 4: bc_4}


n_ver = FacetNormal(mesh)
s_ver = as_vector([-n_ver[1], n_ver[0]])

v_qn = dot(v_qw, n_ver)
v_Mnn = inner(v_qth, outer(n_ver, n_ver))
v_Mns = inner(v_qth, outer(n_ver, s_ver))

V_wt = FunctionSpace(mesh, "DG", deg-1)
V_omn = FunctionSpace(mesh, "DG", deg-1)
V_oms = FunctionSpace(mesh, "DG", deg-1)

# V_wt = FunctionSpace(mesh, "CG", deg)
# V_omn = FunctionSpace(mesh, "CG", deg)
# V_oms = FunctionSpace(mesh, "CG", deg)

Vu = MixedFunctionSpace([V_wt, V_omn, V_oms])

w_t, om_n, om_s = TrialFunction(Vu)
#
# w_t = e_pw
# om_n = dot(e_pth, n_ver)
# om_s = dot(e_pth, s_ver)

b_vec = []
for key,val in bc_dict.items():
    if val == 'F':
        b_vec.append(v_qn * w_t * ds(key) + v_Mnn * om_n * ds(key) + v_Mns * om_s * ds(key))
    elif val == 'S':
        b_vec.append(v_Mnn * om_n * ds(key))

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

n_fig = nreq


n_Vpw = V_pw.dim()
fntsize = 15
for i in range(n_fig):
    # print("Eigenvalue num " + str(i + 1) + ":" + str(omega_tilde[i]))
    eig_real_w = Function(V_pw)
    eig_imag_w = Function(V_pw)

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