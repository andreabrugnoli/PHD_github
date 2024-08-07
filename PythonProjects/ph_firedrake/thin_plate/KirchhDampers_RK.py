# Mindlin plate written with the port Hamiltonian approach

from firedrake import *
import numpy as np

np.set_printoptions(threshold=np.inf)
from math import floor, sqrt, pi
import matplotlib.pyplot as plt

from scipy import linalg as la
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as inv_sp
from scipy import integrate

E = 7e10
nu = 0.35
h = 0.05 # 0.01
rho = 2700  # kg/m^3
D = E * h ** 3 / (1 - nu ** 2) / 12.

L = 1
l_x = L
l_y = L

n = 5 #int(input("N element on each side: "))

m_rod = 10
Jxx_rod = 1. / 12 * m_rod * L ** 2

k_sp1 = 10
k_sp2 = 10

r_sp1 = 0
r_sp2 = 0

# Plate bending stiffness :math:`D=\dfrac{Eh^3}{12(1-\nu^2)}` and shear stiffness :math:`F = \kappa Gh`
# with a shear correction factor :math:`\kappa = 5/6` for a homogeneous plate
# of thickness :math:`h`::


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

V = Vp * Vq
n_pl = V.dim()
n_Vp = Vp.dim()
n_Vq = Vq.dim()

v = TestFunction(V)
v_p, v_q = split(v)

e_v = TrialFunction(V)
e_p, e_q = split(e_v)


al_p = rho * h * e_p
al_q = bending_curv_vec(e_q)

dx = Measure('dx')
ds = Measure('ds')
m_p = dot(v_p, al_p) * dx
m_q = inner(v_q, al_q) * dx
m = m_p + m_q

j_divDiv = -v_p * tensor_divDiv_vec(e_q) * dx
j_divDivIP = tensor_divDiv_vec(v_q) * e_p * dx

j_gradgrad = inner(v_q, Gradgrad_vec(e_p)) * dx
j_gradgradIP = -inner(Gradgrad_vec(v_p), e_q) * dx

j_grad = j_gradgrad + j_gradgradIP  #
j = j_grad


J = assemble(j, mat_type='aij')
M = assemble(m, mat_type='aij')


petsc_j = J.M.handle
petsc_m = M.M.handle


J_pl = np.array(petsc_j.convert("dense").getDenseArray())
M_pl = np.array(petsc_m.convert("dense").getDenseArray())
R_pl = np.zeros((n_pl, n_pl))

# Dirichlet Boundary Conditions and related constraints
# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1

n = FacetNormal(mesh)
# s = as_vector([-n[1], n[0]])

V_qn = FunctionSpace(mesh, 'Lagrange', 2)
V_Mnn = FunctionSpace(mesh, 'Lagrange', 2)

Vu = V_qn * V_Mnn

q_n, M_nn = TrialFunction(Vu)

v_omn = dot(grad(v_p), n)
b_l = v_p * q_n * ds(1) + v_omn * M_nn * ds(1)
b_r = v_p * q_n * ds(2)

# Assemble the stiffness matrix and the mass matrix.

B = assemble(b_l + b_r,  mat_type='aij')
petsc_b = B.M.handle

G_pl = np.array(petsc_b.convert("dense").getDenseArray())

boundary_dofs = np.where(G_pl.any(axis=0))[0]
G_pl = G_pl[:,boundary_dofs]

n_mul = len(boundary_dofs)
# Splitting of matrices

# Force applied at the right boundary
x, y = SpatialCoordinate(mesh)
g = Constant(10)
A = Constant(10**5)
f = Expression('A*(x[1] + pow(x[1] - ly/2,2))', A = A, ly = l_y, degree = 4)
f_w = project(A*((y-l_y/2)**2), Vp) # project(A*sin(2*pi/l_y*y), Vp) #
bp_pl = v_p * f_w * dx                  # v_p * f_w * ds(3) - v_p * f_w * ds(4)
#                                       # -v_p * rho * h * g * dx #

Bf_pl = assemble(bp_pl, mat_type='aij').vector().get_local()

# Final Assemble
r_v = r_sp1 + r_sp2
r_th = l_y**2/4*(r_sp1 - r_sp2)
r_vth = l_y/2*(r_sp1 - r_sp2)


Mp_rod = np.diag([m_rod, Jxx_rod])
Mq_rod = np.diag([1./k_sp1, 1./k_sp2])

M_rod = la.block_diag(Mp_rod, Mq_rod)

Dp_rod = np.array([[1, l_y/2], [1, -l_y/2]])
# Dq_rod = np.array([[-1, -1], [-l_y/2, l_y/2]])
Dq_rod = - Dp_rod.T
n_rod = 4
J_rod = np.zeros((n_rod, n_rod))
J_rod[:2, 2:4] = Dq_rod
J_rod[2:4, :2] = Dp_rod

R_rod = np.zeros((n_rod, n_rod))
Rp_rod = np.array([[r_v, r_vth], [r_vth, r_th]])
R_rod[:n_rod, :n_rod] = la.block_diag(Rp_rod, np.zeros((2, 2)))

Bf_rod = np.zeros((n_rod,))
#
C = np.zeros((n_mul, n_rod))

v_wt, v_omn = TestFunction(Vu)
C[:, 0] = assemble(v_wt * ds(2)).vector().get_local()[boundary_dofs]
C[:, 1] = assemble(v_wt * (y - l_y/2) * ds(2)).vector().get_local()[boundary_dofs]

G_rod = -C.T


Mint = la.block_diag(M_pl, M_rod)
Jint = la.block_diag(J_pl, J_rod)
Rint = la.block_diag(R_pl, R_rod)
Gint = np.concatenate((G_pl, G_rod), axis=0)
Bf_int = np.concatenate((Bf_pl, Bf_rod), axis=0)

Mint_sp = csc_matrix(Mint)
invMint = inv_sp(Mint_sp)
invMint = invMint.toarray()

S = Gint @ la.inv(Gint.T @ invMint @ Gint) @ Gint.T @ invMint

n_tot = n_pl + n_rod
I = np.eye(n_tot)

P = I - S

Jsys = P @ Jint
Rsys = P @ Rint
Fsys = P @ Bf_int


t0 = 0.0
fac = 1
t_base = 0.001
t_fin = t_base *fac
n_t = 100
t_span = [t0, t_fin]

def sys(t,y):
    if t< 0.2 * t_base:
        bool_f = 1
    else: bool_f = 0
    dydt = invMint @ ( (Jsys - Rsys) @ y + Fsys *bool_f)
    return dydt


init_con = Expression(('sin(2*pi*x[0])*sin(2*pi*(x[0]-lx))*sin(2*pi*x[1])*sin(2*pi*(x[1]-ly))', \
                      '0', '0', '0'), degree=4, lx=l_x, ly=l_y)

e_pl0 = Function(V)
e_pl0.assign(project(init_con, V))
y0 = np.zeros(n_tot,)
# y0[:n_pl] = e_pl0.vector().get_local()

t_ev = np.linspace(t0, t_fin, num = n_t)

sol = integrate.solve_ivp(sys, t_span, y0, method='RK45', vectorized=False, t_eval = t_ev)

t_ev = sol.t
n_ev = len(t_ev)
dt_vec = np.diff(t_ev)

e_pl = sol.y[:n_pl, :]
e_rod = sol.y[n_pl: n_tot, :]

e_pw = e_pl[:n_Vp, :]


w0_pl = np.zeros((n_Vp,))
w_pl = np.zeros(e_pw.shape)
w_pl[:, 0] = w0_pl
w_pl_old = w_pl[:, 0]


for i in range(1, n_ev):
    w_pl[:, i] = w_pl_old + 0.5 * (e_pw[:, i - 1] + e_pw[:, i]) * dt_vec[i-1]
    w_pl_old = w_pl[:, i]


y_rod = np.array([0, 1])*l_y
x_rod = np.array([1, 1])*l_x

v_rod = np.zeros((len(x_rod), n_ev))
w_rod = np.zeros((len(x_rod), n_ev))
w_rod_old = w_rod[:, 0]

for i in range(n_ev):

    v_rod[:, i] = x_rod * e_rod[0, i] + (y_rod - l_y / 2) * e_rod[1, i]
    if i >= 1:
        w_rod[:, i] = w_rod_old + 0.5 * (v_rod[:, i - 1] + v_rod[:, i]) * dt_vec[i-1]
        w_rod_old = w_rod[:, i]

w_pl_mm = w_pl * 1000
w_rod_mm = w_rod * 1000

wmm_pl_CGvec = []
w_fun = Function(Vp)
Vp_CG = FunctionSpace(mesh, 'Lagrange', 3)
n_VpCG = Vp_CG.dim()
print(n_Vp, n_VpCG)

maxZvec = np.zeros(n_ev)
minZvec = np.zeros(n_ev)
for i in range(n_ev):
    w_fun.vector()[:] = w_pl_mm[:, i]
    wmm_pl_CG = project(w_fun, Vp_CG)
    wmm_pl_CGvec.append(wmm_pl_CG)

    maxZvec[i] = max(wmm_pl_CG.vector())
    minZvec[i] = min(wmm_pl_CG.vector())

maxZ = max(maxZvec)
minZ = min(minZvec)

import drawNow2, matplotlib
# import tkinter as tk

matplotlib.interactive(True)

matplotlib.rcParams['text.usetex'] = True

plotter = drawNow2.plot3dClass(wmm_pl_CGvec[0], minZ=minZ, maxZ=maxZ, X2=x_rod, Y2=y_rod, \
                               xlabel='$x[m]$', ylabel='$y [m]$', \
                               zlabel='$w [mm]$', title='Vertical Displacement')

for i in range(n_ev):
    wpl_t = w_pl_mm[:, i]
    wrod_t = w_rod_mm[:, i]
    plotter.drawNow2(wmm_pl_CGvec[i], wrod_t, z2label='Rod $w[mm]$')

# tk.mainloop()

if matplotlib.is_interactive():
    plt.ioff()
# plt.close("all")

Hpl_vec = np.zeros((n_ev,))
Hrod_vec = np.zeros((n_ev,))

for i in range(n_ev):
    Hpl_vec[i] = 0.5 * (e_pl[:, i].T @ M_pl @ e_pl[:, i])
    Hrod_vec[i] = 0.5 * (e_rod[:, i].T @ M_rod @ e_rod[:, i])

fig = plt.figure(0)
plt.plot(t_ev, Hpl_vec, 'b-', label='Hamiltonian Plate (J)')
plt.plot(t_ev, Hrod_vec, 'r-', label='Hamiltonian Rod (J)')
plt.plot(t_ev, Hpl_vec + Hrod_vec, 'g-', label='Total Energy (J)')
plt.xlabel(r'{Time} (s)', fontsize=16)
plt.ylabel(r'{Hamiltonian} (J)', fontsize=16)
plt.title(r"Hamiltonian trend",
          fontsize=16)
plt.legend(loc='upper left')

plt.show()
#
