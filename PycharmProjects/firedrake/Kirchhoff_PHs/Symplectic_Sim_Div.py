# Mindlin plate written with the port Hamiltonian approach

from firedrake import *
import numpy as np

np.set_printoptions(threshold=np.inf)
from math import floor, sqrt, pi
import matplotlib.pyplot as plt

from scipy import linalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


E = 7e10
nu = 0.35
h = 0.1
rho = 2700  # kg/m^3
D = E * h ** 3 / (1 - nu ** 2) / 12.

L = 1
l_x = L
l_y = L

n = 5 #int(input("N element on each side: "))

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
    [fl_rot, -nu * fl_rot, 0],
    [-nu * fl_rot, fl_rot, 0],
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

Vp = FunctionSpace(mesh, name_FEp, deg_p)
Vq = VectorFunctionSpace(mesh, name_FEq, deg_q, dim=3)

n_Vp = Vp.dim()
n_Vq = Vq.dim()

v_p = TestFunction(Vp)
v_q = TestFunction(Vq)

e_p = TrialFunction(Vp)
e_q = TrialFunction(Vq)

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
# m = m_p + m_q

j_divDiv = -v_p * tensor_divDiv_vec(e_q) * dx
j_divDivIP = tensor_divDiv_vec(v_q) * e_p * dx

j_gradgrad = inner(v_q, Gradgrad_vec(e_p)) * dx
j_gradgradIP = -inner(Gradgrad_vec(v_p), e_q) * dx

# j = j_gradgrad + j_gradgradIP  #
j_p = j_divDivIP
j_q = j_divDiv

Jp = assemble(j_p, mat_type='aij')
Mp = assemble(m_p, mat_type='aij')

Mq = assemble(m_q, mat_type='aij')
Jq = assemble(j_q, mat_type='aij')


petsc_j_p = Jp.M.handle
petsc_m_p = Mp.M.handle

petsc_j_q = Jq.M.handle
petsc_m_q = Mq.M.handle

D_p = np.array(petsc_j_p.convert("dense").getDenseArray())
M_p = np.array(petsc_m_p.convert("dense").getDenseArray())

D_q = np.array(petsc_j_q.convert("dense").getDenseArray())
M_q = np.array(petsc_m_q.convert("dense").getDenseArray())

# Dirichlet Boundary Conditions and related constraints
# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1
bc_input = input('Select Boundary Condition:')   #'SSSS'

bc_1, bc_3, bc_2, bc_4 = bc_input

bc_dict = {1: bc_1, 2: bc_2, 3: bc_3, 4: bc_4}

V_wt = FunctionSpace(mesh, 'Lagrange', 2)
V_omn = FunctionSpace(mesh, 'Lagrange', 2)

Vu = V_wt * V_omn

wt, om_n = TrialFunction(Vu)
v_wt, v_omn = TestFunction(Vu)

n = FacetNormal(mesh)
s = as_vector([-n[1], n[0]])

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

# Inhomogeneous Neumann Condition
f_w =  Constant(1e6) # project(f_qn, Vp) #
b_y = v_wt * f_w * ds(3) - v_wt * f_w * ds(4)

# Assemble the stiffness matrix and the mass matrix.

B_u = assemble(b_u, mat_type='aij')
petsc_b_u =  B_u.M.handle

B_y = assemble(b_y, mat_type='aij')

B_in  = np.array(petsc_b_u.convert("dense").getDenseArray())

boundary_dofs = np.where(B_in.any(axis=0))[0]
B_in = B_in[:,boundary_dofs]
b_lambda  = B_y.vector().get_local()[boundary_dofs]


# Splitting of matrices

# Force applied at the right boundary
x, y = SpatialCoordinate(mesh)
g = Constant(10)
A = Constant(10**5)
f_w = project(A*x, Vp) # project(1000000*sin(6*pi/l_y*x), Vp) #
# f_w = Expression("1000000*sin(2*pi*x[0])", degree=4)
# f_qn = Expression("1000000*sin(2*pi*x[1])", degree= 4) # Constant(1e5) #
b_p =  v_p * f_w * ds(3) + v_p * f_w * ds(4) + v_p * A * ds(2)
                                 # v_p * f_qn * ds(3) #
                                 # -v_p * rho * h * g * dx #
B_p = assemble(b_p, mat_type='aij').vector().get_local()

# Final Assemble
Mp_sp = csr_matrix(M_p)
Mq_sp = csr_matrix(M_q)

G_eq = B_in.T
G_eqT = np.transpose(G_eq)
G_ep = np.zeros((0, 0))
G_epT = np.transpose(G_eq)

invMp = la.inv(M_p)
invMq = la.inv(M_q)

if G_ep.size != 0:
    invGMGT_p = la.inv(G_ep @ invMp @ G_epT)

if G_eq.size != 0:
    invGMGT_q = la.inv(G_eq @ invMq @ G_eqT)

t_0 = 0
dt = 1e-6
fac = 5
t_f = 0.001
n_ev = 100
t_ev = np.linspace(t_0, t_f, n_ev)

n_t = int(t_f / dt)

ep_sol = np.zeros((n_Vp, n_ev))
eq_sol = np.zeros((n_Vq, n_ev))

init_p = Expression('sin(2*pi*x[0])*sin(2*pi*(x[0]-lx))*sin(2*pi*x[1])*sin(2*pi*(x[1]-ly))', degree=4,
                    lx=l_x, ly=l_y)
init_q = Expression(('0', '0', '0'), degree=4, lx=l_x, ly=l_y)

e_pw_in = project(init_p, Vp)
e_pw_0 = Function(Vp)
e_pw_0.assign(e_pw_in)
ep_old = np.zeros((n_Vp))  # e_pw_0.vector().get_local() #
eq_old = np.zeros((n_Vq))

ep_sol[:, 0] = ep_old
eq_sol[:, 0] = eq_old

k = 1
f = 1
f_lambda = 1
for i in range(1, n_t + 1):

    t = i * dt
    if t < t_f / fac:
        f = 1
        f_lambda = 1
    else:
        f = 0
        f_lambda = 0
    # Intergation for p (n+1/2)

    w_De_q = D_q @ eq_old

    if G_ep.size == 0:
        bp = M_p @ ep_old + 0.5 * dt * (w_De_q + B_p * f)
    else:
        lmbda_p = - invGMGT_p @ G_ep @ invMp @ (w_De_q + B_p * f)
        bp = M_p @ ep_old + 0.5 * dt * (w_De_q + G_epT @ lmbda_p + B_p * f)

    bp_sp = csr_matrix(bp).reshape((n_Vp, 1))
    ep_new = spsolve(Mp_sp, bp_sp)

    ep_old = ep_new
    # Integration of q (n+1)

    w_De_p = D_p @ ep_new

    if G_eq.size == 0:
        bq = M_q @ eq_old + dt * w_De_p
    else:
        # NOOOO b_lmabda has to be derived. Look at time-dependent bcs
        lmbda_q = - invGMGT_q @ (G_eq @ invMq @ w_De_p - b_lambda *f_lambda)
        bq = M_q @ eq_old + dt * (w_De_p + G_eqT @ lmbda_q)

    bq_sp = csr_matrix(bq).reshape((n_Vq, 1))
    eq_new = spsolve(Mq_sp, bq_sp)

    eq_old = eq_new

    # Intergation for p (n+1)

    w_De_q = D_q @ eq_old

    if G_ep.size == 0:
        bp = M_p @ ep_old + 0.5 * dt * (w_De_q + B_p * f)
    else:
        lmbda_p = - invGMGT_p @ G_ep @ invMp @ (w_De_q + B_p * f)
        bp = M_p @ ep_old + 0.5 * dt * (w_De_q + G_epT @ lmbda_p + B_p * f)

    bp_sp = csr_matrix(bp).reshape((n_Vp, 1))
    ep_new = spsolve(Mp_sp, bp_sp)

    ep_old = ep_new

    if t >= t_ev[k]:
        ep_sol[:, k] = ep_new
        eq_sol[:, k] = eq_new
        k = k + 1
        print('Solution number ' + str(k) + ' computed')

w0 = np.zeros((n_Vp,))
w = np.zeros(ep_sol.shape)
w[:, 0] = w0
w_old = w[:, 0]

Deltat = t_f / (n_ev - 1)
for i in range(1, n_ev):
    w[:, i] = w_old + 0.5 * (ep_sol[:, i - 1] + ep_sol[:, i]) * Deltat
    w_old = w[:, i]

w_mm = w * 1000

wmm_CGvec = []
w_fun = Function(Vp)
Vp_CG = FunctionSpace(mesh, 'Lagrange', 3)
n_VpCG = Vp_CG.dim()
print(n_Vp, n_VpCG)

maxZvec = np.zeros(n_ev)
minZvec = np.zeros(n_ev)
for i in range(n_ev):
    w_fun.vector()[:] = w_mm[:, i]
    wmm_CG = project(w_fun, Vp_CG)
    wmm_CGvec.append(wmm_CG)

    maxZvec[i] = max(wmm_CG.vector())
    minZvec[i] = min(wmm_CG.vector())

maxZ = max(maxZvec)
minZ = min(minZvec)

print(maxZ)
print(minZ)

import matplotlib, drawNow

matplotlib.rcParams['text.usetex'] = True
matplotlib.interactive(True)

plotter = drawNow.plot3dClass(wmm_CGvec[0], minZ= minZ, maxZ = maxZ,  \
                         xlabel = '$x[m]$', ylabel = '$y [m]$', \
                         zlabel = '$w [mm]$', title = 'Vertical Displacement')

for i in range(n_ev):
    # w_fun.vector()[:] = w_mm[:, i]
    # wmm_CG = project(w_fun, Vp_CG)
    plotter.drawNow(wmm_CGvec[i])

if matplotlib.is_interactive():
    plt.ioff()
plt.close('all')
#
# H_vec = np.zeros((n_ev,))
# fntsize = 15
#
# for i in range(n_ev):
#     H_vec[i] = 0.5 * (np.transpose(ep_sol[:, i]) @ M_p @ ep_sol[:, i] + np.transpose(eq_sol[:, i]) @ M_q @ eq_sol[:, i])
#
# t_ev = np.linspace(t_0, t_f, n_ev)
# fig = plt.figure(0)
# plt.plot(t_ev, H_vec, 'b-', label='Hamiltonian (J)')
# # plt.plot(t_ev, Ep, 'r-', label = 'Potential Energy (J)')
# # plt.plot(t_ev, H_vec + Ep, 'g-', label = 'Total Energy (J)')
# plt.xlabel(r'{Time} (s)', fontsize=fntsize)
# plt.ylabel(r'{Hamiltonian} (J)',fontsize=fntsize)
# plt.title(r"Hamiltonian trend",
#           fontsize=fntsize)
# plt.legend(loc='upper left')
#
# plt.show()
# path_out = "/home/a.brugnoli/PycharmProjects/firedrake/Kirchhoff_PHs/Simulations/"


# plt.savefig(path_out + "Sim1_Hamiltonian.eps", format="eps")
#
# plot_solutions = True
# if plot_solutions:
#
#     from matplotlib.ticker import LinearLocator, FormatStrFormatter
#     from matplotlib import cm
#     from mpl_toolkits.mplot3d import Axes3D
#
#     plt.close('all')
#     matplotlib.rcParams['text.usetex'] = True
#
#     n_fig = 4
#     tol = 1e-6
#
#     for i in range(n_fig):
#         index = int(n_ev/n_fig*(i+1)-1)
#         fig = plt.figure(i+1)
#         w_fun = Function(Vp)
#         w_fun.vector()[:] = w_mm[:, index]
#
#         Vp_CG = FunctionSpace(mesh, 'Lagrange', 3)
#         wmm_wCG = project(w_fun, Vp_CG)
#
#         from firedrake.plot import _two_dimension_triangle_func_val
#
#         triangulation, Z = _two_dimension_triangle_func_val(wmm_wCG, 10)
#
#         figure = plt.figure(i)
#         ax = figure.add_subplot(111, projection="3d")
#
#         ax.plot_trisurf(triangulation, Z, cmap=cm.jet)
#
#         ax.set_xbound(-tol, l_x + tol)
#         ax.set_xlabel('$x [m]$', fontsize=fntsize)
#
#         ax.set_ybound(-tol, l_y + tol)
#         ax.set_ylabel('$y [m]$', fontsize=fntsize)
#         ax.set_title('$w[mm]$', fontsize=fntsize)
#
#         ax.w_zaxis.set_major_locator(LinearLocator(10))
#         ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))
#
#         ax.set_zlabel('$w [mm]$', fontsize=fntsize)
#         ax.set_title('Vertical displacement', fontsize=fntsize)
#
#         ax.set_zlim3d(minZ - 0.01 * abs(minZ), maxZ + 0.01 * abs(maxZ))
#
#         plt.savefig(path_out + "Sim1_t_" + str(index + 1) + ".eps", format="eps")
#
#         plt.show()

