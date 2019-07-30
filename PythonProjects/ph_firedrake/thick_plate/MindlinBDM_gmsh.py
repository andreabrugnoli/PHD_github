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


deg = 1 #int(input('Degree for FE: '))

rho = 3000
E = 7e10
nu = 0.3
h = 0.003
k = 5/6

ray = 0.1

plot_eigenvector = 'n'

G = E / 2 / (1 + nu)
F = G * h * k


# Useful Matrices
D = E * h ** 3 / (1 - nu ** 2) / 12.
fl_rot = 12 / (E * h ** 3)

# The unit square mesh is divided in :math:`N\times N` quadrilaterals::

mesh = Mesh("attachment.msh")

plot(mesh)
plt.show()

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


V_pw = FunctionSpace(mesh, "DG", deg)
V_skw = FunctionSpace(mesh, "DG", deg)
V_pth = VectorFunctionSpace(mesh, "DG", deg)

V_qth1 = FunctionSpace(mesh, "BDM", deg+1)
V_qth2 = FunctionSpace(mesh, "BDM", deg+1)
V_qw = FunctionSpace(mesh, "BDM", deg+1)

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

j_Id = dot(v_pth, e_qw) * dx
j_IdIP = -dot(v_qw, e_pth) * dx

j_alldiv = j_div + j_divIP + j_divSym + j_divSymIP + j_Id + j_IdIP

j_form = j_alldiv

# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1

n_ver = FacetNormal(mesh)
s_ver = as_vector([-n_ver[1], n_ver[0]])

v_qn = dot(v_qw, n_ver)
v_Mnn = inner(v_qth, outer(n_ver, n_ver))
v_Mns = inner(v_qth, outer(n_ver, s_ver))

V_wt = FunctionSpace(mesh, "DG", deg)
V_omn = FunctionSpace(mesh, "DG", deg)
V_oms = FunctionSpace(mesh, "DG", deg)
#
# V_wt = FunctionSpace(mesh, "CG", deg+1)
# V_omn = FunctionSpace(mesh, "CG", deg+1)
# V_oms = FunctionSpace(mesh, "CG", deg+1)

Vu = MixedFunctionSpace([V_wt, V_omn, V_oms])

w_t, om_n, om_s = TrialFunction(Vu)

x, y = SpatialCoordinate(mesh)
free_boundary = conditional(Not(And(le(x, ray), le(y, ray))), 1.0, 0)

# b_u = v_qn * w_t * free_boundary * ds + v_Mnn * om_n * free_boundary * ds + v_Mns * om_s * free_boundary * ds
b_u = v_qn * w_t * ds(1) + v_Mnn * om_n * ds(1) + v_Mns * om_s * ds(1)


M = assemble(m_form, mat_type='aij')
J = assemble(j_form, mat_type='aij')


petsc_j = J.M.handle
petsc_m = M.M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())

B = assemble(b_u, mat_type="aij")
petsc_b = B.M.handle
B_in = np.array(petsc_b.convert("dense").getDenseArray())
boundary_dofs = np.where(B_in.any(axis=0))[0]
B_in = B_in[:, boundary_dofs]
n_u = len(boundary_dofs)


Z_u = np.zeros((n_u, n_u))

J_aug = np.vstack([np.hstack([JJ, B_in]),
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

omega_fr = omega/(2*pi)

for i in range(5):
    print(omega_fr[i])

n_fig = 5


n_Vpw = V_pw.dim()
fntsize = 15
for i in range(n_fig):
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