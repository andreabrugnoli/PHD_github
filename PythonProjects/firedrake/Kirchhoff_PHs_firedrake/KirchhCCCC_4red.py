from firedrake import *
import numpy as np

np.set_printoptions(threshold=np.inf)
from math import floor, sqrt, pi
import matplotlib.pyplot as plt

from scipy import linalg as la
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as inv_sp
from scipy import integrate

import matplotlib.pyplot as plt
import matplotlib
from AnimateSurfFiredrake import animate2D
import matplotlib.animation as animation

matplotlib.rcParams['text.usetex'] = True


E = 7e10
nu = 0.35
h = 0.05 # 0.01
rho = 2700  # kg/m^3
D = E * h ** 3 / (1 - nu ** 2) / 12.


L = 1
l_x = L
l_y = 3*L

n = 8
deg = 5


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

def tensor_divDiv_vec(u):
    return u[0].dx(0).dx(0) + u[1].dx(1).dx(1) + 2 * u[2].dx(0).dx(1)

def Gradgrad_vec(u):
    return as_vector([ u.dx(0).dx(0), u.dx(1).dx(1), 2 * u.dx(0).dx(1) ])


# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
L = 1
l_x = L
l_y = L
n_x, n_y = n, n
mesh = UnitSquareMesh(n_x, n_y, quadrilateral=False)

# plot(mesh)
# plt.show()


# Finite element defition

name_FEp = 'Bell'
name_FEq = 'Bell'

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
m_p = dot(v_p, al_p) * dx
m_q = inner(v_q, al_q) * dx
m = m_p + m_q

j_divDiv = -v_p * tensor_divDiv_vec(e_q) * dx
j_divDivIP = tensor_divDiv_vec(v_q) * e_p * dx

j_gradgrad = inner(v_q, Gradgrad_vec(e_p) ) * dx
j_gradgradIP = -inner(Gradgrad_vec(v_p), e_q) * dx


j = j_divDiv + j_divDivIP #

# Assemble the stiffness matrix and the mass matrix.
J = assemble(j, mat_type='aij')
M = assemble(m, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle

J_pl = np.array(petsc_j.convert("dense").getDenseArray())
M_pl = np.array(petsc_m.convert("dense").getDenseArray())

M_sp = csc_matrix(M_pl)
invM_pl = inv_sp(M_sp)
invM_pl = invM_pl.toarray()

bp_pl = v_p * dx

B_pl = assemble(bp_pl, mat_type='aij').vector().get_local()

f_hz = 1
om = 2*pi*f_hz

t0 = 0.0
t_fin = 0.01
n_t = 300
t_span = [t0, t_fin]

def sys(t,y):
    # u = 0*np.cos(om*t)
    # if t< 0.01 * t_fin:
    #     dydt = invM_pl @ (J_pl @ y + B_pl*u)
    # else: dydt = invM_pl @ ( J_pl @ y )

    dydt = la.solve(M_pl, J_pl @ y, sym_pos=True)
    return dydt

x1, x2 = SpatialCoordinate(mesh)
A_v0 = 1e-3
ic_p = A_v0*(1 - cos(4*pi*x1/l_x))*(1 - cos(4*pi*x2/l_y))

e_p0 = Function(Vp)
e_q0 = Function(Vq)
e_p0.assign(project(ic_p, Vp))
y0 = np.zeros(n_V,)
y0[:n_Vp] = e_p0.vector().get_local()
y0[n_Vp:n_V] = e_q0.vector().get_local()

t_ev = np.linspace(t0, t_fin, num = n_t)

sol = integrate.solve_ivp(sys, t_span, y0, method='LSODA', t_eval=t_ev)

t_ev = sol.t
e_sol = sol.y
n_ev = len(t_ev)

al_sol = np.zeros(e_sol.shape)
for i in range(n_ev):
    al_sol[:, i] = M_pl @ e_sol[:, i]

J_file = 'J_file.npy'; M_file = 'M_file';  Q_file = 'Q_file.npy'; B_file = 'B_file.npy'
X_file = 'X_file.npy'; F_file = 'F_file.npy'
np.save(J_file, J_pl), np.save(Q_file, invM_pl), np.save(B_file, B_pl), np.save(M_file, M_pl)
np.save(X_file, al_sol), np.save(F_file, e_sol)

ep_sol = e_sol[:n_Vp, :]


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

fig = plt.figure()
plt.plot(t_ev, Hpl_vec, 'b-', label='Hamiltonian Plate (J)')
plt.xlabel(r'{Time} (s)', fontsize=16)
plt.ylabel(r'{Hamiltonian} (J)', fontsize=16)
plt.title(r"Hamiltonian trend",
          fontsize=16)
plt.legend(loc='upper left')

anim = animate2D(minZ, maxZ, wmm_CGvec, t_ev, xlabel = '$x[m]$', ylabel = '$y [m]$', \
                         zlabel = '$w [mm]$', title = 'Vertical Displacement')

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
# anim.save('Kirchh_NoRod.mp4', writer=writer)

plt.show()