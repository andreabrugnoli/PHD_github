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
nreq = 10

E = 1
nu = 0.3

rho = 1
L = 1
h = 0.01

plot_eigenvector = 'y'

bc_input = input('Select Boundary Condition: ')   #'SSSS'

# Useful Matrices
D = E * h ** 3 / (1 - nu ** 2) / 12.
fl_rot = 12 / (E * h ** 3)

D_b = as_tensor([
  [D, D * nu, 0],
  [D * nu, D, 0],
  [0, 0, D * (1 - nu) / 2]
])

C_b = as_tensor([
    [fl_rot, -nu * fl_rot, 0],
    [-nu * fl_rot, fl_rot, 0],
    [0, 0, fl_rot * (1 + nu) / 2]
])

# The unit square mesh is divided in :math:`N\times N` quadrilaterals::

L = 1

n_x, n_y = n_el, n_el
L_x, L_y = L, L
# mesh = RectangleMesh(n_x, n_y, L_x, L_y, quadrilateral=True)

mesh_int = IntervalMesh(n_el, L)
mesh = ExtrudedMesh(mesh_int, n_el)

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

CG_deg1 = FiniteElement("CG", interval, deg)
DG_deg = FiniteElement("DG", interval, deg-1)
DG_deg1 = FiniteElement("DG", interval, deg)

P_CG1_DG = TensorProductElement(CG_deg1, DG_deg)
P_DG_CG1 = TensorProductElement(DG_deg, CG_deg1)

RT_horiz = HDivElement(P_CG1_DG)
RT_vert = HDivElement(P_DG_CG1)
RT_quad = RT_horiz + RT_vert

P_CG1_DG1 = TensorProductElement(CG_deg1, DG_deg1)
P_DG1_CG1 = TensorProductElement(DG_deg1, CG_deg1)

BDM_horiz = HDivElement(P_CG1_DG1)
BDM_vert = HDivElement(P_DG1_CG1)
BDM_quad = BDM_horiz + BDM_vert

V_pw = FunctionSpace(mesh, "CG", deg)
V_qthD = FunctionSpace(mesh, BDM_quad)
V_qth12 = FunctionSpace(mesh, "CG", deg)

V = MixedFunctionSpace([V_pw, V_qthD, V_qth12])
n_Vp = V.sub(0).dim()
n_Vq = V.sub(1).dim() + V.sub(2).dim()
n_V = V.dim()

print(n_V)
v = TestFunction(V)
v_p, v_qD, v_q12 = split(v)

e = TrialFunction(V)
e_p, e_qD, e_q12 = split(e)

v_q = as_tensor([[v_qD[0], v_q12],
                    [v_q12, v_qD[1]]
                   ])

e_q = as_tensor([[e_qD[0], e_q12],
                    [e_q12, e_qD[1]]
                   ])

al_p = rho * h * e_p
al_q = bending_curv(e_q)

# v_skw = skew(v_skw)
# al_skw = skew(e_skw)

dx = Measure('dx')
ds = Measure('ds')

m_form = v_p * al_p * dx + inner(v_q, al_q) * dx

n_ver = FacetNormal(mesh)
s_ver = as_vector([-n_ver[1], n_ver[0]])

e_mnn = inner(e_q, outer(n_ver, n_ver))
v_mnn = inner(v_q, outer(n_ver, n_ver))

e_mns = inner(e_q, outer(n_ver, s_ver))
v_mns = inner(v_q, outer(n_ver, s_ver))

j_graddiv = dot(grad(v_p),  div(e_q)) * dx \
            + v_p * dot(grad(e_mns), s_ver) * ds_v \
            + v_p * dot(grad(e_mns), s_ver) * ds_b \
            + v_p * dot(grad(e_mns), s_ver) * ds_t
j_divgrad = - dot(div(v_q), grad(e_p)) * dx \
            - dot(grad(v_mns), s_ver) * e_p * ds_v \
            - dot(grad(v_mns), s_ver) * e_p * ds_b \
            - dot(grad(v_mns), s_ver) * e_p * ds_t

j_form = j_graddiv + j_divgrad

# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1

bc_1, bc_2, bc_3, bc_4 = bc_input

bc_dict = {1: bc_1, 3: bc_2, 2: bc_3, 4: bc_4}

V_qn = FunctionSpace(mesh, "CG", deg)
V_om = FunctionSpace(mesh, BDM_quad)

V_u = MixedFunctionSpace([V_qn, V_om])
v_u = TrialFunction(V_u)
q_n, om_vec = split(v_u)

om_n = dot(om_vec, n_ver)

b_vec = []

for key,val in bc_dict.items():
    if key == 1 or key == 2:
        if val == 'C':
            b_vec.append(v_p * q_n * ds_v(key))
        elif val == 'S':
            b_vec.append(v_p * q_n * ds_v(key) + v_mnn * om_n * ds_v(key))
        elif val == 'F':
            b_vec.append(v_mnn * om_n * ds_v(key))
    elif key == 3:
        if val == 'C':
            b_vec.append(v_p * q_n * ds_b)
        elif val == 'S':
            b_vec.append(v_p * q_n * ds_b + v_mnn * om_n * ds_b)
        elif val == 'F':
            b_vec.append(v_mnn * om_n * ds_b)
    else:
        if val == 'C':
            b_vec.append(v_p * q_n * ds_t)
        elif val == 'S':
            b_vec.append(v_p * q_n * ds_t + v_mnn * om_n * ds_t)
        elif val == 'F':
            b_vec.append(v_mnn * om_n * ds_t)

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

x, y = SpatialCoordinate(mesh)
con_dom1 = And(And(gt(x, L / 4), lt(x, 3 * L / 4)), And(gt(y, L / 4), lt(y, 3 * L / 4)))
Dom_f = conditional(con_dom1, 1., 0.)
B_f = assemble(v_p * Dom_f * dx).vector().get_local()

Z_u = np.zeros((n_u, n_u))

J_aug = np.vstack([np.hstack([JJ, B_in]),
                   np.hstack([-B_in.T, Z_u])
                   ])

Z_al_u = np.zeros((n_V, n_u))
Z_u_al = np.zeros((n_u, n_V))

E_aug = np.vstack([np.hstack([MM, Z_al_u]),
                   np.hstack([Z_u_al,    Z_u])
                   ])


B_aug = np.concatenate((B_f, np.zeros(n_u, )), axis=0).reshape((-1, 1))

# plate_full = SysPhdaeRig(n_V, 0, 0, n_Vp, n_Vq, MM, JJ, B_in)
plate_full = SysPhdaeRig(n_V+n_u, n_u, 0, n_Vp, n_Vq, E_aug, J_aug, B_aug)

s0 = 0.01
n_red = 50
plate_red, V_red = plate_full.reduce_system(s0, n_red)
Vall_red = la.block_diag(V_red, np.eye(n_u))

E_red = plate_red.E
J_red = plate_red.J
B_red = plate_red.B

# J_red = np.vstack([np.hstack([J_red, B_red]),
#                    np.hstack([-B_red.T, Z_u])
#                    ])
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

print(Vall_red.shape, eigvec_red.shape)

plt.plot(np.real(eigenvaluesF), np.imag(eigenvaluesF), 'r+', np.real(eigenvaluesR), np.imag(eigenvaluesR), 'bo')
plt.legend(("Eigenvalues full", "Eigenvalues reduced"))
plt.show()

# NonDimensional China Paper

n_om = 5

omegaF_tilde = L**2*sqrt(rho*h/D)*omega_full
omegaR_tilde = L**2*sqrt(rho*h/D)*omega_red

for i in range(n_om):
    print(omegaF_tilde[i])

for i in range(n_om):
    print(omegaF_tilde[i], omegaR_tilde[i])
