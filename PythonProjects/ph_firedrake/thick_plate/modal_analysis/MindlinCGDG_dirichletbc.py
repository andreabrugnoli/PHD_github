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

n = 15
deg = 1

rho = 1
E = 1
nu = 0.3
thick = 'y'
if thick == 'y':
    h = 0.1
else:
    h = 0.01

plot_eigenvector = 'y'

bc_input = input('Select Boundary Condition: ')

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

Vp_w = FunctionSpace(mesh, "CG", deg)
Vp_th = VectorFunctionSpace(mesh, "CG", deg)
# Vq_th = VectorFunctionSpace(mesh, "CG", deg, dim=3)
# Vq_w = VectorFunctionSpace(mesh, "CG", deg)
Vq_th = VectorFunctionSpace(mesh, "DG", deg-1, dim=3)
Vq_w = VectorFunctionSpace(mesh, "DG", deg-1)

V = Vp_w * Vp_th * Vq_th * Vq_w
print(V.dim())

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

# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1

bc_1, bc_2, bc_3, bc_4 = bc_input

bc_dict = {1: bc_1, 2: bc_3, 3: bc_2, 4: bc_4}


bcs = []
for key,val in bc_dict.items():
    if val == 'C':
        bcs.append(DirichletBC(V.sub(0), Constant(0.0), key))
        bcs.append(DirichletBC(V.sub(1), Constant((0.0, 0.0)), key))

    elif val == 'S':
        bcs.append(DirichletBC(V.sub(0), Constant(0.0), key))
        if key == 1 or key ==2:
            bcs.append(DirichletBC(V.sub(1).sub(1), Constant(0.0), key))
        else:
            bcs.append(DirichletBC(V.sub(1).sub(0), Constant(0.0), key))

J = assemble(j_form, bcs=bcs, mat_type='aij')
M = assemble(m_form, bcs=bcs, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())

tol = 10**(-9)

eigenvalues, eigvectors = la.eig(JJ, MM)
omega_all = np.imag(eigenvalues)

index = omega_all > tol

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()


omega_tilde = omega*L*((2*(1+nu)*rho)/E)**0.5

for i in range(5):
    print("Eigenvalue num " + str(i + 1) + ":" + str(omega_tilde[i]))

n_fig = 5

n_Vpw = Vp_w.dim()
fntsize = 15
for i in range(n_fig):
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