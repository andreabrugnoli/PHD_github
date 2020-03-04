from firedrake import *
import numpy as np
import scipy.linalg as la
import scipy.sparse as spa
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from firedrake.plot import calculate_one_dim_points

fntsize = 15

n_el = 100
L = 1

mesh = IntervalMesh(n_el, L)
x = SpatialCoordinate(mesh)[0]

rho = 1
T = 1
w0 = -5*sin(2*pi*x)*cos(2*pi*x)
v0 = 2*sin(pi*x/2)*sin(pi*x/2)

sig0 = T*w0.dx(0)

# Finite element defition
Vp_D = FunctionSpace(mesh, "DG", 0)
Vq_D = FunctionSpace(mesh, "Lagrange", 1)

V_D = Vp_D * Vq_D
nVp_D = Vp_D.dim()
nVq_D = Vq_D.dim()
nV_D = V_D.dim()


v_D = TestFunction(V_D)
vp_D, vq_D = split(v_D)

e_D = TrialFunction(V_D)
ep_D, eq_D = split(e_D)

alp_D = rho * ep_D
alq_D = 1. / T * eq_D

dx = Measure('dx')
ds = Measure('ds')


m_formD = vp_D * alp_D * dx + vq_D * alq_D * dx

petsc_mD = assemble(m_formD, mat_type='aij').M.handle
M_D = np.array(petsc_mD.convert("dense").getDenseArray())

invM_D = np.linalg.inv(M_D)

j_div = vp_D * eq_D.dx(0) * dx
j_divIP = - vq_D.dx(0) * ep_D * dx

j_D = j_div + j_divIP

petcs_jD = assemble(j_D, mat_type='aij').M.handle
J_D = np.array(petcs_jD.convert("dense").getDenseArray())

b0_D = -vq_D * ds(1)
B0_D = assemble(b0_D).vector().get_local().reshape((-1, 1))

bL_D = vq_D * ds(2)
BL_D = assemble(bL_D).vector().get_local().reshape((-1, 1))

B_D = np.hstack((B0_D, BL_D))

print(B_D[B_D!=0])

exp_w0_D = interpolate(w0, Vp_D)
exp_v0_D = interpolate(v0, Vp_D)
exp_sig0_D = interpolate(sig0, Vq_D)

ep0_D = Function(Vp_D).assign(exp_v0_D).vector().get_local()
eq0_D = Function(Vq_D).assign(exp_sig0_D).vector().get_local()

e0_D = np.concatenate((ep0_D, eq0_D))

v0_0 = Function(Vp_D).assign(exp_v0_D).at(0)
v0_L = Function(Vp_D).assign(exp_v0_D).at(L)

# Controls: boundary velocities (Dirichlet)

uD_0 = lambda t: np.cos(pi*t) * v0_0
uD_L = lambda t: np.cos(4*pi*t) * v0_L

def func_D(t, y):

    uD = np.array([uD_0(t), uD_L(t)])
    dydt = invM_D @ (J_D @ y + B_D @ uD)

    return dydt


t0 = 0
tfin = 1
n_t = 200

t_ev = np.linspace(t0, tfin, num=n_t)

sol_D = integrate.solve_ivp(func_D, [0, tfin], e0_D, method='BDF', vectorized=False, t_eval=t_ev, atol=1e-8, rtol=1e-6)

t_evD = sol_D.t
e_solD = sol_D.y
ep_solD = e_solD[:nVp_D, :]
eq_solD = e_solD[nVp_D:, :]
n_evD = len(t_evD)
dt_vecD = np.diff(t_evD)

w_D = np.zeros(ep_solD.shape)
w_oldD = Function(Vp_D).assign(exp_w0_D).vector().get_local()
w_D[:, 0] = w_oldD

pnt_elem = 5
n_x = pnt_elem*n_el
w_plotD = np.zeros((n_x, n_evD))

w_funcD = Function(Vp_D)

H_D = np.zeros(n_evD, )

for i in range(1, n_evD):
    w_D[:, i] = w_oldD + 0.5 * (ep_solD[:, i - 1] + ep_solD[:, i]) * dt_vecD[i - 1]
    w_oldD = w_D[:, i]

for i in range(n_evD):
    H_D[i] = 0.5 * (e_solD[:, i].T @ M_D @ e_solD[:, i])

    w_funcD.vector().set_local(w_D[:, i])
    x_plotD, w_plotD[:, i] = calculate_one_dim_points(w_funcD, pnt_elem)

# Hamiltonians
plt.figure()
plt.plot(t_evD, H_D, 'b', label='Dirichlet')
plt.ylabel('Hamiltonian (Joules)')
plt.title('Energies')
plt.legend()

X_plotD, T_plotD = np.meshgrid(x_plotD, t_evD)

# Customize the z axis.

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Space coordinate $[m]$', fontsize=fntsize)
ax.set_ylabel('Time $[s]$', fontsize=fntsize)
ax.set_zlabel('Vertical deflection $[m]$', fontsize=fntsize)
ax.set_title('Dirichlet')
W_plotD = np.transpose(w_plotD)
surf_D = ax.plot_surface(X_plotD, T_plotD, W_plotD, cmap=cm.jet, linewidth=0, antialiased=False, label='Dirichlet $w$')

#
# plt.figure()
plt.figure()
plt.plot(t_evD, ep_solD[0, :], label='Left velocity (Dirichlet)')
plt.plot(t_evD, uD_0(t_evD), label='Left velocity (exact)')
# plt.legend()
# plt.title('Boundary controls Dirichlet')
# plt.xlabel('t (seconds)')
#
# plt.figure()
plt.plot(t_evD, ep_solD[-1, :], label='Right velocity (Dirichlet)')
plt.plot(t_evD, uD_L(t_evD), label='Right velocity (exact)')

plt.legend()
plt.title('Boundary controls Dirichlet')
plt.xlabel('t (seconds)')

plt.show()
