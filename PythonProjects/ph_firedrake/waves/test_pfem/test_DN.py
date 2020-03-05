from firedrake import *
import numpy as np
import scipy.linalg as la
import scipy.sparse as spa
import scipy.sparse.linalg as sp_la

from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from firedrake.plot import calculate_one_dim_points

fntsize = 15

n_el = 1000
L = 1

mesh = IntervalMesh(n_el, L)
x = SpatialCoordinate(mesh)[0]

tfin = 10

# Physical parameters
rho = 10-cos(pi*x) #  Space-dependent mass density
T = 1.25*(-x*(1-x)+2/3) # Space-dependent Young's modulus
w0 = 20*exp(-10*(x-1/2)**2) + 5*x**2 - 8*x + 2 #  Space-dependent initial deflection
v0 = 2*x*(1-x) + 2  # Space-dependent initial deflection velocity


# rho = 1
# T = 1
# w0 = -5*sin(2*pi*x)*cos(2*pi*x)
# v0 = 2*sin(pi*x/2)*sin(pi*x/2)

sig0 = T*w0.dx(0)

# Finite element defition
Vp_N = FunctionSpace(mesh, "Lagrange", 1)
Vq_N = FunctionSpace(mesh, "DG", 0)

Vp_D = FunctionSpace(mesh, "DG", 0)
Vq_D = FunctionSpace(mesh, "Lagrange", 1)

V_N = Vp_N * Vq_N
nVp_N = Vp_N.dim()
nVq_N = Vq_N.dim()
nV_N = V_N.dim()

V_D = Vp_D * Vq_D
nVp_D = Vp_D.dim()
nVq_D = Vq_D.dim()
nV_D = V_D.dim()

v_N = TestFunction(V_N)
vp_N, vq_N = split(v_N)

e_N = TrialFunction(V_N)
ep_N, eq_N = split(e_N)

alp_N = rho * ep_N
alq_N = 1. / T * eq_N

v_D = TestFunction(V_D)
vp_D, vq_D = split(v_D)

e_D = TrialFunction(V_D)
ep_D, eq_D = split(e_D)

alp_D = rho * ep_D
alq_D = 1. / T * eq_D

dx = Measure('dx')
ds = Measure('ds')

m_formN = vp_N * alp_N * dx + vq_N * alq_N * dx

petsc_mN = assemble(m_formN, mat_type='aij').M.handle
# M_N = np.array(petsc_mN.convert("dense").getDenseArray())
M_N = spa.csr_matrix(petsc_mN.getValuesCSR()[::-1])

# M = spa.csr_matrix(M)
# invM_N = np.linalg.inv(M_N)

m_formD = vp_D * alp_D * dx + vq_D * alq_D * dx

petsc_mD = assemble(m_formD, mat_type='aij').M.handle
# M_D = np.array(petsc_mD.convert("dense").getDenseArray())
M_D = spa.csr_matrix(petsc_mD.getValuesCSR()[::-1])

# invM_D = np.linalg.inv(M_D)

j_grad = vq_N * ep_N.dx(0) * dx
j_gradIP = - vp_N.dx(0) * eq_N * dx

j_N = j_grad + j_gradIP

petcs_jN = assemble(j_N, mat_type='aij').M.handle
# J_N = np.array(petcs_jN.convert("dense").getDenseArray())
J_N = spa.csr_matrix(petcs_jN.getValuesCSR()[::-1])

b0_N = vp_N * ds(1)
bL_N = vp_N * ds(2)

B0_N = assemble(b0_N).vector().get_local().reshape((-1, 1))
BL_N = assemble(bL_N).vector().get_local().reshape((-1, 1))

B_N = np.hstack((B0_N, BL_N))
B_N = spa.csr_matrix(B_N)

j_div = vp_D * eq_D.dx(0) * dx
j_divIP = - vq_D.dx(0) * ep_D * dx

j_D = j_div + j_divIP

petcs_jD = assemble(j_D, mat_type='aij').M.handle
# J_D = np.array(petcs_jD.convert("dense").getDenseArray())
J_D = spa.csr_matrix(petcs_jD.getValuesCSR()[::-1])

b0_D = -vq_D * ds(1)
bL_D = vq_D * ds(2)

B0_D = assemble(b0_D).vector().get_local().reshape((-1, 1))
BL_D = assemble(bL_D).vector().get_local().reshape((-1, 1))

B_D = np.hstack((B0_D, BL_D))
B_D = spa.csr_matrix(B_D)

exp_w0_N = interpolate(w0, Vp_N)
exp_v0_N = interpolate(v0, Vp_N)
exp_sig0_N = interpolate(sig0, Vq_N)

ep0_N = Function(Vp_N).assign(exp_v0_N).vector().get_local()
eq0_N = Function(Vq_N).assign(exp_sig0_N).vector().get_local()

e0_N = np.concatenate((ep0_N, eq0_N))

sig0_0 = Function(Vq_N).assign(exp_sig0_N).at(0)
sig0_L = Function(Vq_N).assign(exp_sig0_N).at(L)

exp_w0_D = interpolate(w0, Vp_D)
exp_v0_D = interpolate(v0, Vp_D)
exp_sig0_D = interpolate(sig0, Vq_D)

ep0_D = Function(Vp_D).assign(exp_v0_D).vector().get_local()
eq0_D = Function(Vq_D).assign(exp_sig0_D).vector().get_local()

e0_D = np.concatenate((ep0_D, eq0_D))

v0_0 = Function(Vp_D).assign(exp_v0_D).at(0)
v0_L = Function(Vp_D).assign(exp_v0_D).at(L)

# Controls: boundary sigma (Neumann)

uN_0 = lambda t: -np.cos(pi*t) * (t+1) * sig0_0
uN_L = lambda t: np.cos(4*pi*t) * np.exp(-t) * sig0_L

# Controls: boundary velocities (Dirichlet)

uD_0 = lambda t: np.cos(pi*t) * 1/(t+1) * v0_0
uD_L = lambda t: np.cos(4*pi*t) * np.exp(-t) * v0_L


def func_N(t, y):

    uN = np.array([uN_0(t), uN_L(t)])
    dydt = sp_la.spsolve(M_N, J_N @ y + B_N @ uN)

    return dydt

def func_D(t, y):

    uD = np.array([uD_0(t), uD_L(t)])
    dydt = sp_la.spsolve(M_D, J_D @ y + B_D @ uD)

    return dydt

# def func_N(t, y):
#
#     uN = np.array([uN_0(t), uN_L(t)])
#     dydt = invM_N @ (J_N @ y + B_N @ uN)
#
#     return dydt
#
# def func_D(t, y):
#
#     uD = np.array([uD_0(t), uD_L(t)])
#     dydt = invM_D @ (J_D @ y + B_D @ uD)
#
#     return dydt


t0 = 0
n_t = 200

t_ev = np.linspace(t0, tfin, num=n_t)

sol_N = integrate.solve_ivp(func_N, [0, tfin], e0_N, method='BDF', vectorized=False, t_eval=t_ev, atol=1e-8, rtol=1e-6)
sol_D = integrate.solve_ivp(func_D, [0, tfin], e0_D, method='BDF', vectorized=False, t_eval=t_ev, atol=1e-8, rtol=1e-6)

t_evN = sol_N.t
e_solN = sol_N.y
ep_solN = e_solN[:nVp_N, :]
eq_solN = e_solN[nVp_N:, :]
n_evN = len(t_evN)
dt_vecN = np.diff(t_evN)

t_evD = sol_D.t
e_solD = sol_D.y
ep_solD = e_solD[:nVp_D, :]
eq_solD = e_solD[nVp_D:, :]
n_evD = len(t_evD)
dt_vecD = np.diff(t_evD)

w_N = np.zeros(ep_solN.shape)
w_oldN = Function(Vp_N).assign(exp_w0_N).vector().get_local()
w_N[:, 0] = w_oldN

w_D = np.zeros(ep_solD.shape)
w_oldD = Function(Vp_D).assign(exp_w0_D).vector().get_local()
w_D[:, 0] = w_oldD

pnt_elem = 5
n_x = pnt_elem*n_el
w_plotN = np.zeros((n_x, n_evN))
w_plotD = np.zeros((n_x, n_evD))

w_funcN = Function(Vp_N)
w_funcD = Function(Vp_D)

H_N = np.zeros(n_evN, )
H_D = np.zeros(n_evD, )

for i in range(1, n_evN):
    w_N[:, i] = w_oldN + 0.5 * (ep_solN[:, i - 1] + ep_solN[:, i]) * dt_vecN[i-1]
    w_oldN = w_N[:, i]

for i in range(1, n_evD):
    w_D[:, i] = w_oldD + 0.5 * (ep_solD[:, i - 1] + ep_solD[:, i]) * dt_vecD[i - 1]
    w_oldD = w_D[:, i]

for i in range(n_evN):
    H_N[i] = 0.5 * (e_solN[:, i].T @ M_N @ e_solN[:, i])

    w_funcN.vector().set_local(w_N[:, i])

    x_plotN, w_plotN[:, i] = calculate_one_dim_points(w_funcN, pnt_elem)

for i in range(n_evD):
    H_D[i] = 0.5 * (e_solD[:, i].T @ M_D @ e_solD[:, i])

    w_funcD.vector().set_local(w_D[:, i])
    x_plotD, w_plotD[:, i] = calculate_one_dim_points(w_funcD, pnt_elem)

yN = B_N.T @ e_solN
yD = B_D.T @ e_solD

# Hamiltonians
plt.figure()
plt.plot(t_evN, H_N, 'r', label='Neumann')
plt.plot(t_evD, H_D, 'b', label='Dirichlet')
plt.ylabel('Hamiltonian (Joules)')
plt.title('Energies')
plt.legend()


X_plotN, T_plotN = np.meshgrid(x_plotN, t_evN)
X_plotD, T_plotD = np.meshgrid(x_plotD, t_evD)

# Customize the z axis.


# Plot the surface.
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Space coordinate $[m]$', fontsize=fntsize)
ax.set_ylabel('Time $[s]$', fontsize=fntsize)
ax.set_zlabel('Vertical deflection $[m]$', fontsize=fntsize)
ax.set_title('Neumann')
W_plotN = np.transpose(w_plotN)
surf_N = ax.plot_surface(X_plotN, T_plotN, W_plotN, cmap=cm.jet, linewidth=0, antialiased=False, label='Neumann $w$')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('Space coordinate $[m]$', fontsize=fntsize)
ax.set_ylabel('Time $[s]$', fontsize=fntsize)
ax.set_zlabel('Vertical deflection $[m]$', fontsize=fntsize)
ax.set_title('Dirichlet')
W_plotD = np.transpose(w_plotD)
surf_D = ax.plot_surface(X_plotD, T_plotD, W_plotD, cmap=cm.jet, linewidth=0, antialiased=False, label='Dirichlet $w$')

# surf_D._facecolors2d = surf_D._facecolors3d
# surf_D._edgecolors2d = surf_D._edgecolors3d
#
# surf_N._facecolors2d = surf_N._facecolors3d
# surf_N._edgecolors2d = surf_N._edgecolors3d
#
# plt.legend()

# Boundary controls
plt.figure()
plt.plot(t_evN, -eq_solN[0, :], label='Left force (Neumann)')
plt.plot(t_evN, uN_0(t_evN), label='Left force (exact)')
# plt.legend()ep_solD[0, :]
# plt.title('Boundary controls Neumann')
# plt.xlabel('t (seconds)')
#
# plt.figure()

plt.plot(t_evN, eq_solN[-1, :], label='Right force (Neumann)')
plt.plot(t_evN, uN_L(t_evN), label='Right force (exact)')
plt.legend()
plt.title('Boundary controls Neumann')
plt.xlabel('t (seconds)')

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

# Boundary observations
plt.figure()
plt.plot(t_evN, yN[0, :], label='Left velocity (Neumann)')
plt.plot(t_evN, yN[1, :], label='Right velocity (Neumann)')
plt.legend()
plt.title('Boundary Observations Neumann')
plt.xlabel('t (seconds)')

plt.figure()
plt.plot(t_evD, yD[0, :], label='Left force (Dirichlet)')
plt.plot(t_evD, yD[1, :], label='Right force (Dirichlet)')

plt.legend()
plt.title('Boundary Observations Dirichlet')
plt.xlabel('t (seconds)')

diffH_N = np.diff(H_N)
diffH_D = np.diff(H_D)

dotHNint = np.divide(diffH_N, dt_vecN)
# dotHNext = np.multiply(eq_solN[0, :], yN[0, :]) + np.multiply(eq_solN[-1, :], yN[1, :])
dotHNext = np.multiply(uN_0(t_evN), yN[0, :]) + np.multiply(uN_L(t_evN), yN[1, :])

plt.figure()
plt.plot(t_evN[:-1], dotHNint, label='H int (Neumann)')
plt.plot(t_evN, dotHNext, label='Hext (Neumann)')
plt.legend()
plt.title('Hamitonian Neumann')
plt.xlabel('t (seconds)')

dotHDint = np.divide(diffH_D, dt_vecD)
# dotHDext = np.multiply(ep_solD[0, :], yD[0, :]) + np.multiply(ep_solD[-1, :], yD[1, :])
dotHDext = np.multiply(uD_0(t_evD), yD[0, :]) + np.multiply(uD_L(t_evD), yD[1, :])

plt.figure()
plt.plot(t_evD[:-1], dotHDint, label='H int (Dirichlet)')
plt.plot(t_evD, dotHDext, label='Hext (Dirichlet)')

plt.legend()
plt.title('Hamitonian Dirichlet')
plt.xlabel('t (seconds)')

plt.show()
