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

n = 4
deg_bd = 2
z_imp = 5

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

# j_gradgrad = inner(v_q, Gradgrad_vec(e_p)) * dx
# j_gradgradIP = -inner(Gradgrad_vec(v_p), e_q) * dx

j_div = j_divDiv + j_divDivIP
# j_grad = j_gradgrad + j_gradgradIP  #
j = j_div

J = assemble(j, mat_type='aij')
M = assemble(m, mat_type='aij')
petsc_j = J.M.handle
petsc_m = M.M.handle

J_pl = np.array(petsc_j.convert("dense").getDenseArray())
M_pl = np.array(petsc_m.convert("dense").getDenseArray())


# Dirichlet Boundary Conditions and related constraints
# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1
n = FacetNormal(mesh)
s = as_vector([ -n[1], n[0] ])

V_wt = FunctionSpace(mesh, 'Lagrange', 2)
V_omn = FunctionSpace(mesh, 'Lagrange', 2)

Vu = V_wt * V_omn

wt, om_n = TrialFunction(Vu)

v_Mnn = v_q[0]*n[0]**2 + v_q[1]*n[1]**2 + 2 * v_q[2]*n[0]*n[1]
v_Mns = v_q[0]*s[0]*n[0] + v_q[1]*s[1]*n[1] + v_q[2]*(s[0]*n[1] + s[1]*n[0])

v_qn = - dot(tensor_Div_vec(v_q), n) - dot(grad(v_Mns), s)

b_bd = v_Mnn * om_n * ds + v_qn * wt * ds

# Assemble the stiffness matrix and the mass matrix.
B = assemble(b_bd,  mat_type='aij')
petsc_b = B.M.handle

Bbd_pl = np.array(petsc_b.convert("dense").getDenseArray())

boundary_dofs = np.where(Bbd_pl.any(axis=0))[0]
Bbd_pl = Bbd_pl[:,boundary_dofs]

n_bdofs = len(boundary_dofs)

# Final Assemble

Z = np.eye(n_bdofs) * z_imp

R_pl = Bbd_pl @ Z @ Bbd_pl.T

# Splitting of matrices
M_sp = csc_matrix(M_pl)
invM_pl = inv_sp(M_sp)
invM_pl = invM_pl.toarray()

n_tot = n_pl

Jsys = J_pl
Rsys = R_pl


t0 = 0.0
fac = 1
t_base = 0.001
t_fin = t_base *fac
n_t = 100
t_span = [t0, t_fin]

def sys(t,y):
    if t< 0.1 * t_base:
        dydt = invM_pl @ (Jsys @ y)
    else: dydt = invM_pl @ ( (Jsys - Rsys) @ y )
    return dydt

x, y = SpatialCoordinate(mesh)
init_con = x**2

e_p0 = Function(Vp)
e_p0.assign(project(init_con, Vp))
y0 = np.zeros(n_tot,)
y0[:n_Vp] = e_p0.vector().get_local()

t_ev = np.linspace(t0, t_fin, num = n_t)

sol = integrate.solve_ivp(sys, t_span, y0, method='RK45', vectorized=True, t_eval = t_ev)

t_ev = sol.t

e_sol = sol.y
ep_sol = e_sol[:n_Vp, :]
n_ev = len(t_ev)
dt_vec = np.diff(t_ev)

w0 = np.zeros((n_Vp,))
w = np.zeros(ep_sol.shape)
w[:, 0] = w0
w_old = w[:, 0]

for i in range(1, n_ev):
    w[:, i] = w_old + 0.5 * (ep_sol[:, i - 1] + ep_sol[:, i]) * dt_vec[i-1]
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

Hpl_vec = np.zeros((n_ev,))
for i in range(n_ev):
    Hpl_vec[i] = 0.5 * (e_sol[:, i].T @ M_pl @ e_sol[:, i])

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
fig = plt.figure(0)
plt.plot(t_ev, Hpl_vec, 'b-', label='Hamiltonian Plate (J)')
plt.xlabel(r'{Time} (s)', fontsize=16)
plt.ylabel(r'{Hamiltonian} (J)', fontsize=16)
plt.title(r"Hamiltonian trend",
          fontsize=16)
plt.legend(loc='upper left')

plt.show()



from AnimateSurfFiredrake import animate2D
import matplotlib.animation as animation
anim = animate2D(minZ, maxZ, wmm_CGvec, t_ev, xlabel = '$x[m]$', ylabel = '$y [m]$', \
                         zlabel = '$w [mm]$', title = 'Vertical Displacement')

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
# anim.save('Kirchh_NoRod.mp4', writer=writer)

plt.show()