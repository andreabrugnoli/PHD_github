# Mindlin plate written with the port Hamiltonian approach

from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt

import scipy.linalg as la

import matplotlib
import matplotlib.pyplot as plt
from firedrake.plot import _two_dimension_triangle_func_val
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

matplotlib.rcParams['text.usetex'] = True

from firedrake.petsc import PETSc
try:
    from slepc4py import SLEPc
except ImportError:
    import sys
    warning("Unable to import SLEPc, eigenvalue computation not possible (try firedrake-update --slepc)")
    sys.exit(0)


n = 50
r = 1 #int(input('Degree for FE: '))

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

def j_operator(v_p, v_q, e_p, e_q):

    j_form = - inner(grad(grad(v_p)), e_q) * dx \
    + jump(grad(v_p), n_ver) * dot(dot(e_q('+'), n_ver('+')), n_ver('+')) * dS \
    + dot(grad(v_p), n_ver) * dot(dot(e_q, n_ver), n_ver) * ds \
    + inner(v_q, grad(grad(e_p))) * dx \
    - dot(dot(v_q('+'), n_ver('+')), n_ver('+')) * jump(grad(e_p), n_ver) * dS \
    - dot(dot(v_q, n_ver), n_ver) * dot(grad(e_p), n_ver) * ds

    return j_form


# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
n_x, n_y = n, n
mesh = UnitSquareMesh(n_x, n_y, quadrilateral=False)

# domain = mshr.Rectangle(Point(0, 0), Point(l_x, l_y))
# mesh = mshr.generate_mesh(domain, n, "cgal")

# plot(mesh)
# plt.show()


# Finite element defition

Vp = FunctionSpace(mesh, 'CG', r)
Vq = FunctionSpace(mesh, 'HHJ', r-1)
V = Vp * Vq

n_Vp = V.sub(0).dim()
n_Vq = V.sub(1).dim()
n_V = V.dim()
print(n_V)

v = TestFunction(V)
v_p, v_q = split(v)

e = TrialFunction(V)
e_p, e_q = split(e)

al_p = rho * h * e_p
al_q = bending_curv(e_q)

dx = Measure('dx')
ds = Measure('ds')
dS = Measure("dS")

m_form = inner(v_p, al_p) * dx + inner(v_q, al_q) * dx

n_ver = FacetNormal(mesh)
s_ver = as_vector([-n_ver[1], n_ver[0]])

j_form = j_operator(v_p, v_q, e_p, e_q)

bc_1, bc_2, bc_3, bc_4 = bc_input

bc_dict = {1: bc_1, 3: bc_2, 2: bc_3, 4: bc_4}

# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1

bcs = []

for key, val in bc_dict.items():

    if val == 'C':
        bcs.append(DirichletBC(V.sub(0), Constant(0.0), key))

    elif val == 'S':
        bcs.append(DirichletBC(V.sub(0), Constant(0.0), key))
        bcs.append( DirichletBC(V.sub(1), Constant(((0.0, 0.0), (0.0, 0.0))), key))

    elif val == 'F':
        bcs.append(DirichletBC(V.sub(1), Constant(((0.0, 0.0), (0.0, 0.0))), key))

M = assemble(m_form, bcs=bcs, mat_type='aij')
J = assemble(j_form, bcs=bcs, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle

petsc_j = J.M.handle
petsc_m = M.M.handle

tol = 10**(-9)

# plt.spy(J_aug); plt.show()

num_eigenvalues = 100

target = 10/norm_coeff
opts = PETSc.Options()
opts.setValue("pos_gen_non_hermitian", None)
opts.setValue("st_pc_factor_shift_type", "NONZERO")
opts.setValue("eps_type", "krylovschur")
opts.setValue("eps_tol", 1e-10)
opts.setValue("st_type", "sinvert")
# opts.setValue("eps_target_imaginary", None)
opts.setValue("st_shift", target)
opts.setValue("eps_target", target)


es = SLEPc.EPS().create(comm=COMM_WORLD)
# st = es.getST()
# st.setShift(0)
# st.setType("sinvert")
# es.setST(st)
# es.setWhichEigenpairs(1)
es.setDimensions(num_eigenvalues)
es.setOperators(petsc_j, petsc_m)
es.setFromOptions()
es.solve()

nconv = es.getConverged()
if nconv == 0:
    import sys
    warning("Did not converge any eigenvalues")
    sys.exit(0)

vr, vi = petsc_j.getVecs()

tol = 1e-6
lamda_vec = np.zeros((nconv))

n_stocked_eig = 0

omega_tilde = []
eig_real_w_vec = []
eig_imag_w_vec = []

eig_real_w = Function(Vp)
eig_imag_w = Function(Vp)
fntsize = 15

n_pw = Vp.dim()
for i in range(nconv):
    lam = es.getEigenpair(i, vr, vi)

    lam_r = np.real(lam)
    lam_c = np.imag(lam)

    if lam_c > tol:

        print(n_stocked_eig)

        omega_tilde.append(lam_c*norm_coeff)

        vr_w = vr.getArray().copy()[:n_pw]
        vi_w = vi.getArray().copy()[:n_pw]

        eig_real_w_vec.append(vr_w)
        eig_imag_w_vec.append(vi_w)

        n_stocked_eig += 1

n_req = min(4, n_stocked_eig)

for i in range(n_req):
    print("Eigenvalue num " + str(i + 1) + ":" + str(omega_tilde[i]))

for i in range(n_req):
    norm_real_eig = np.linalg.norm(eig_real_w_vec[i])
    norm_imag_eig = np.linalg.norm(eig_imag_w_vec[i])

    eig_real_w.vector()[:] = eig_real_w_vec[i]
    eig_imag_w.vector()[:] = eig_imag_w_vec[i]

    figure = plt.figure()
    ax = figure.add_subplot(111, projection="3d")

    ax.set_xlabel('$x [m]$', fontsize=fntsize)
    ax.set_ylabel('$y [m]$', fontsize=fntsize)
    ax.set_title('Eigenvector num ' + str(i + 1), fontsize=fntsize)

    ax.w_zaxis.set_major_locator(LinearLocator(10))
    ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

    if norm_imag_eig > norm_real_eig:
        triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_imag_w, 10)
        # plot(eig_imag_w, axes=ax, plot3d=True)
    else:
        triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_real_w, 10)
        # plot(eig_real_w, axes=ax, plot3d=True)

    ax.plot_trisurf(triangulation, z_goodeig, cmap=cm.jet)

plt.show()