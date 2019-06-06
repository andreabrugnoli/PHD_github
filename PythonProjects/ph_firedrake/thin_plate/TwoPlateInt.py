from firedrake import *
import numpy as np

np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

from scipy import linalg as la
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as inv_sp
from scipy import integrate

E = 7e10
nu = 0.35
h = 0.05 # 0.01
rho = 2700  # kg/m^3
D = E * h ** 3 / (1 - nu ** 2) / 12.

n = 4


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

V_qn = FunctionSpace(mesh, 'Lagrange', 1)
V_Mnn = FunctionSpace(mesh, 'Lagrange', 1)

Vu = V_qn * V_Mnn

q_n, M_nn = TrialFunction(Vu)
v_omn = dot(grad(v_p), n)
g_l = v_p * q_n * ds(1) + v_omn * M_nn * ds(1)
g = v_p * q_n * ds + v_omn * M_nn * ds

# Assemble the stiffness matrix and the mass matrix.

petsc_g_l = assemble(g_l,  mat_type='aij').M.handle
petsc_g = assemble(g,  mat_type='aij').M.handle

G_l = np.array(petsc_g_l.convert("dense").getDenseArray())

bd_l = np.where(G_l.any(axis=0))[0]

G_clamp = G_l[:, bd_l]

n_mul = len(bd_l)

G_all = np.array(petsc_g.convert("dense").getDenseArray())

tab_coord = mesh.coordinates.dat.data

tab_x = tab_coord[:, 0]
tab_y = tab_coord[:, 1]


dof_int1_l = ((tab_x == l_x) & (tab_y == l_y/4)).nonzero()[0][0]
dof_int1_r = ((tab_x == 0.0) & (tab_y == l_y/4)).nonzero()[0][0]

dof_int2_l = ((tab_x == l_x) & (tab_y == 3*l_y/4)).nonzero()[0][0]
dof_int2_r = ((tab_x == 0.0) & (tab_y == 3*l_y/4)).nonzero()[0][0]

n_qn = V_qn.dim()
Gint_pl1 = G_all[:, (dof_int1_l, dof_int1_l + n_qn, dof_int2_l, dof_int2_l + n_qn)]
Gint_pl2 = G_all[:, (dof_int1_r, dof_int1_r + n_qn, dof_int2_r, dof_int2_r + n_qn)]

Gint = np.concatenate((Gint_pl1, -Gint_pl2), axis=0)

G_D = np.concatenate((G_clamp, np.zeros((n_pl, n_mul)) ), axis=0)

G = np.concatenate((G_D, Gint), axis=1)

x, y = SpatialCoordinate(mesh)
A = Constant(10**5)
f_w = project(A*(y-l_y/2)**2, Vp)
b_f1 = v_p * f_w * dx  # ds(3) - v_pw * f_w *  ds(4)
Bf_pl1 = assemble(b_f1, mat_type='aij').vector().get_local()

b_f2 = - v_p * f_w * dx  # ds(3) - v_pw * f_w *  ds(4)
Bf_pl2 = assemble(b_f2, mat_type='aij').vector().get_local()

Mint = la.block_diag(M_pl, M_pl)
Jint = la.block_diag(J_pl, J_pl)
Rint = la.block_diag(R_pl, R_pl)

Bf_int = np.concatenate((Bf_pl1, Bf_pl2), axis=0)

Mint_sp = csc_matrix(Mint)
invMint = inv_sp(Mint_sp)
invMint = invMint.toarray()

S = G @ la.inv(G.T @ invMint @ G) @ G.T @ invMint

n_tot = 2 * n_pl
I = np.eye(n_tot)

P = I - S

Jsys = P @ Jint
Rsys = P @ Rint
Fsys = P @ Bf_int


t0 = 0.0
t_fin = 0.001
n_t = 100
t_span = [t0, t_fin]

def sys(t,y):
    if t< 0.2 * t_fin:
        bool_f = 1
    else: bool_f = 0
    dydt = invMint @ ( (Jsys - Rsys) @ y + Fsys *bool_f)
    return dydt


y0 = np.zeros(n_tot,)
# y0[:n_pl] = e_pl0.vector().get_local()

t_ev = np.linspace(t0, t_fin, num = n_t)

sol = integrate.solve_ivp(sys, t_span, y0, method='RK45', vectorized=False, t_eval = t_ev)

t_ev = sol.t
n_ev = len(t_ev)
dt_vec = np.diff(t_ev)

e_pl1 = sol.y[:n_pl, :]
e_rod = sol.y[n_pl: n_tot, :]


e_pl1 = sol.y[:n_pl, :]
e_pl2 = sol.y[n_pl:n_tot, :]

e_pw1 = e_pl1[:n_Vp, :]  # np.zeros((n_pw,n_t))
e_pw2 = e_pl2[:n_Vp, :]

w0_pl1 = np.zeros((n_Vp,))
w_pl1 = np.zeros(e_pw1.shape)
w_pl1[:, 0] = w0_pl1
w_pl1_old = w_pl1[:, 0]

w0_pl2 = np.zeros((n_Vp,))
w_pl2 = np.zeros(e_pw2.shape)
w_pl2[:, 0] = w0_pl2
w_pl2_old = w_pl2[:, 0]

dt_vec = np.diff(t_ev)
for i in range(1, n_ev):
    w_pl1[:, i] = w_pl1_old + 0.5 * (e_pw1[:, i - 1] + e_pw1[:, i]) * dt_vec[i-1]
    w_pl1_old = w_pl1[:, i]

    w_pl2[:, i] = w_pl2_old + 0.5 * (e_pw2[:, i - 1] + e_pw2[:, i]) * dt_vec[i-1]
    w_pl2_old = w_pl2[:, i]


w_pl1_mm = w_pl1 * 1000
w_pl2_mm = w_pl2 * 1000

wmm_pl1_CGvec = []
wmm_pl2_CGvec = []

w_fun = Function(Vp)
Vp_CG = FunctionSpace(mesh, 'Lagrange', 3)
n_VpCG = Vp_CG.dim()
print(n_Vp, n_VpCG)

maxZvec = np.zeros(n_ev)
minZvec = np.zeros(n_ev)
for i in range(n_ev):
    w_fun.vector()[:] = w_pl1_mm[:, i]
    wmm_pl1_CG = project(w_fun, Vp_CG)
    wmm_pl1_CGvec.append(wmm_pl1_CG)

    w_fun.vector()[:] = w_pl2_mm[:, i]
    wmm_pl2_CG = project(w_fun, Vp_CG)
    wmm_pl2_CGvec.append(wmm_pl2_CG)

    maxZvec[i] = max(max(wmm_pl1_CG.vector()), max(wmm_pl2_CG.vector()))
    minZvec[i] = min(min(wmm_pl1_CG.vector()), min(wmm_pl2_CG.vector()))

maxZ = max(maxZvec)
minZ = min(minZvec)


Hpl1_vec = np.zeros((n_ev,))
Hpl2_vec = np.zeros((n_ev,))

for i in range(n_ev):
    Hpl1_vec[i] = 0.5 * (e_pl1[:, i].T @ M_pl @ e_pl1[:, i])
    Hpl2_vec[i] = 0.5 * (e_pl2[:, i].T @ M_pl @ e_pl2[:, i])

fig = plt.figure()
plt.plot(t_ev, Hpl1_vec, 'b-', label='Hamiltonian Plate 1 (J)')
plt.plot(t_ev, Hpl2_vec, 'r-', label='Hamiltonian Plate 2 (J)')
plt.plot(t_ev, Hpl1_vec + Hpl2_vec, 'g-', label='Total Energy (J)')
plt.xlabel(r'{Time} (s)', fontsize=16)
plt.ylabel(r'{Hamiltonian} (J)', fontsize=16)
plt.title(r"Hamiltonian trend",
          fontsize=16)
plt.legend(loc='upper left')

from tools_plotting.animate_2surf import animate2D
import matplotlib.animation as animation
sol = np.concatenate( (w_pl1_mm, w_pl2_mm), axis=0)
anim = animate2D(minZ, maxZ, wmm_pl1_CGvec, wmm_pl2_CGvec, t_ev, xlabel = '$x[m]$', ylabel = '$y [m]$', \
                         zlabel = '$w [mm]$', title = 'Vertical Displacement')

Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
#
# pathout = './'
# anim.save(pathout + 'IntKirchoffPlates.mp4', writer=writer)


plt.show()
