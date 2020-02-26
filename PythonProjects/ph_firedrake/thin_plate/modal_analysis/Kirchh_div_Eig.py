# Mindlin plate written with the port Hamiltonian approach

from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)
from math import floor, sqrt, pi
import scipy.linalg as la

from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from firedrake.plot import _two_dimension_triangle_func_val
from mpl_toolkits.mplot3d import Axes3D
from math import pi

E = 2e11
nu = 0.3
h = 0.01
rho = 8000  # kg/m^3
D = E * h ** 3 / (1 - nu ** 2) / 12.

L = 1
l_x = L
l_y = L

n = 4 #int(input("N element on each side: "))


# Plate bending stiffness :math:`D=\dfrac{Eh^3}{12(1-\nu^2)}` and shear stiffness :math:`F = \kappa Gh`
# with a shear correction factor :math:`\kappa = 5/6` for a homogeneous plate
# of thickness :math:`h`::
# E = Constant(7e10)
# nu = Constant(0.3)
# h = Constant(0.2)
# rho = Constant(2000)  # kg/m^3
# k = Constant(5. / 6.)

 # Useful Matrices

D_b = as_tensor([
  [D, D * nu, 0],
  [D * nu, D, 0],
  [0, 0, D * (1 - nu) / 2]
])

fl_rot = 12. / (E * h ** 3)

C_b_vec = as_tensor([
    [fl_rot, -nu*fl_rot, 0],
    [-nu*fl_rot, fl_rot, 0],
    [0, 0, fl_rot * 2 * (1 + nu)]
])

# Vectorial Formulation possible only
def bending_moment_vec(kk):
    return dot(D_b, kk)

def bending_curv_vec(MM):
    return dot(C_b_vec, MM)

def tensor_divDiv_vec(MM):
    return MM[0].dx(0).dx(0) + MM[1].dx(1).dx(1) + 2 * MM[2].dx(0).dx(1)

def Gradgrad_vec(u):
    return as_vector([ u.dx(0).dx(0), u.dx(1).dx(1), 2 * u.dx(0).dx(1) ])

def tensor_Div_vec(MM):
    return as_vector([ MM[0].dx(0) + MM[2].dx(1), MM[2].dx(0) + MM[1].dx(1) ])

# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
L = 1
l_x = L
l_y = L
n_x, n_y = n, n
mesh = UnitSquareMesh(n_x, n_y, quadrilateral=False)

# plot(mesh)
# plt.show()

nameFE = 'Bell'
name_FEp = nameFE
name_FEq = nameFE

if name_FEp == 'Morley':
    deg_p = 2
elif name_FEp == 'Hermite':
    deg_p = 3
elif name_FEp == 'Argyris' or name_FEp == 'Bell':
    deg_p = 5

if name_FEq == 'Morley':
    deg_q = 2
elif name_FEq == 'Hermite':
    deg_q = 3
elif name_FEq == 'Argyris' or name_FEq == 'Bell':
    deg_q = 5

V_p = FunctionSpace(mesh, name_FEp, deg_p)
V_q = VectorFunctionSpace(mesh, name_FEq, deg_q, dim=3)
V = V_p * V_q

n_Vp = V.sub(0).dim()
n_Vq = V.sub(1).dim()
n_V  = V.dim()
print(n_V)

v = TestFunction(V)
v_p, v_q = split(v)

e_v = TrialFunction(V)
e_p, e_q = split(e_v)

al_p = rho * h * e_p
al_q = bending_curv_vec(e_q)

# alpha = TrialFunction(V)
# al_pw, al_pth, al_qth, al_qw = split(alpha)

# e_p = 1. / (rho * h) * al_p
# e_q = bending_moment_vec(al_q)

dx = Measure('dx')
ds = Measure('ds')
m_p = dot(v_p, al_p) * dx
m_q = inner(v_q, al_q) * dx
m = m_p + m_q

j_divDiv = -v_p * tensor_divDiv_vec(e_q) * dx
j_divDivIP = tensor_divDiv_vec(v_q) * e_p * dx

j_gradgrad = inner(v_q, Gradgrad_vec(e_p) ) * dx
j_gradgradIP = -inner(Gradgrad_vec(v_p), e_q) * dx


j = j_divDiv + j_divDivIP


# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1

bc_input = input('Select Boundary Condition: ')   #'SSSS'

bc_1, bc_3, bc_2, bc_4 = bc_input

bc_dict = {1: bc_1, 2: bc_2, 3: bc_3, 4: bc_4}

n = FacetNormal(mesh)
s = as_vector([ -n[1], n[0] ])

V_wt = FunctionSpace(mesh, 'CG', 1)
V_omn = FunctionSpace(mesh, 'CG', 1)
# V_om = FunctionSpace(mesh, "RT", 1)

V_u = MixedFunctionSpace([V_wt, V_omn])
v_u = TrialFunction(V_u)

wt, om_n = split(v_u)

# om_n = dot(om_vec, n)


v_Mnn = v_q[0]*n[0]**2 + v_q[1]*n[1]**2 + 2 * v_q[2]*n[0]*n[1]
v_Mns = v_q[0]*s[0]*n[0] + v_q[1]*s[1]*n[1] + v_q[2]*(s[0]*n[1] + s[1]*n[0])

v_qn = - dot(tensor_Div_vec(v_q), n) - dot(grad(v_Mns), s)

b_vec = []
for key,val in bc_dict.items():
    if val == 'F':
        b_vec.append( v_qn * wt * ds(key) + v_Mnn * om_n * ds(key))
    elif val == 'S':
        b_vec.append(v_Mnn * om_n * ds(key) )

b_u = sum(b_vec)

# Assemble the stiffness matrix and the mass matrix.
J = assemble(j, mat_type='aij')
M = assemble(m, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())

B_u = assemble(b_u, mat_type='aij')
# B_y = assemble(b_y_omn, mat_type='aij')

petsc_b_u =  B_u.M.handle
# petsc_b_y =  B_y.M.handle

# print(B_u.array().shape)
B_in  = np.array(petsc_b_u.convert("dense").getDenseArray())
# B_out = np.array(petsc_b_y.convert("dense").getDenseArray())


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
tol = 10**(-6)

eigenvalues, eigvectors = la.eig(J_aug, M_aug)
omega_all = np.imag(eigenvalues)

index = omega_all>=tol

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]


omega.sort()

# NonDimensional China Paper
n_om = 3

omega_tilde = L**2*sqrt(rho*h/D)*omega
for i in range(n_om):
    print(omega_tilde[i])

plot_eigenvectors = True
if plot_eigenvectors:

    fntsize = 15
    n_fig = 3
    n_Vpw = V_p.dim()

    for i in range(n_om):
        eig_real_w = Function(V_p)
        eig_imag_w = Function(V_p)

        eig_real_p = np.real(eigvec_omega[:n_Vp, i])
        eig_imag_p = np.imag(eigvec_omega[:n_Vp, i])
        eig_real_w.vector()[:] = eig_real_p
        eig_imag_w.vector()[:] = eig_imag_p

        Vp_CG = FunctionSpace(mesh, 'Lagrange', 3)
        eig_real_wCG = project(eig_real_w, Vp_CG)
        eig_imag_wCG = project(eig_imag_w, Vp_CG)

        norm_real_eig = np.linalg.norm(eig_real_wCG.vector().get_local())
        norm_imag_eig = np.linalg.norm(eig_imag_wCG.vector().get_local())

        if norm_imag_eig > norm_real_eig:
            triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_imag_wCG, 10)
        else:
            triangulation, z_goodeig = _two_dimension_triangle_func_val(eig_real_wCG, 10)

        figure = plt.figure(i)
        ax = figure.add_subplot(111, projection="3d")

        ax.plot_trisurf(triangulation, z_goodeig, cmap=cm.jet)

        ax.set_xbound(-tol, l_x + tol)
        ax.set_xlabel('$x [m]$', fontsize=fntsize)

        ax.set_ybound(-tol, l_y + tol)
        ax.set_ylabel('$y [m]$', fontsize=fntsize)

        ax.set_title('$v_{e_{w}}$', fontsize=fntsize)

        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

        path_out2 = "/home/a.brugnoli/PycharmProjects/firedrake/Kirchhoff_PHs/Eig_Kirchh/Imag_Eig/"
        # plt.savefig(path_out2 + "Case" + bc_input + "_el" + str(n) + "_FE_" + name_FEp + "_eig_" + str(i+1) + ".eps", format="eps")

plt.show()


