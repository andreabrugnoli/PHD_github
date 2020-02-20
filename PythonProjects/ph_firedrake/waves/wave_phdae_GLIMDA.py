# Dissipation is not considered carefully
# BEWARE


from firedrake import *
import numpy as np
import scipy.linalg as la
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt
from tools_plotting.animate_surf import animate2D
from assimulo.solvers import GLIMDA
from assimulo.implicit_ode import Implicit_Problem

plt.rc('text', usetex=True)
# Finite element defition

# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
path_mesh = "/home/a.brugnoli/GitProjects/PythonProjects/ph_firedrake/waves/meshes/"
mesh = Mesh(path_mesh + "circle_5.msh")

# figure = plt.figure()
# ax = figure.add_subplot(111)
# plot(mesh, axes=ax)
# plt.show()

rho = 1
T = as_tensor([[1, 0], [0, 1]])

deg_p = 1
deg_q = 2
Vp = FunctionSpace(mesh, "CG", deg_p)
Vq = FunctionSpace(mesh, "RT", deg_q)
# Vq = VectorFunctionSpace(mesh, "Lagrange", deg_q)

V = Vp * Vq

n_V = V.dim()
print(n_V)

v = TestFunction(V)
v_p, v_q = split(v)

e_v = TrialFunction(V)
e_p, e_q = split(e_v)

al_p = rho * e_p
al_q = dot(inv(T), e_q)

dx = Measure('dx')
ds = Measure('ds')
m_p = v_p * al_p * dx
m_q = dot(v_q, al_q) * dx
m_form = m_p + m_q

j_div = dot(v_p, div(e_q)) * dx
j_divIP = -dot(div(v_q), e_p) * dx

j_form = j_div + j_divIP
petsc_j = assemble(j_form, mat_type='aij').M.handle
petsc_m = assemble(m_form, mat_type='aij').M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())

n_p = Vp.dim()
n_q = Vq.dim()

n_e = V.dim()

n_ver = FacetNormal(mesh)
x, y = SpatialCoordinate(mesh)

u_Dxy = pow(x, 2) - pow(y, 2)

ray = sqrt(pow(x, 2) + pow(y, 2))
is_D = conditional(lt(ray, 1.5), 1, 0)
b_Dxy = dot(v_q, n_ver) * u_Dxy * is_D * ds
b_Dt = dot(v_q, n_ver) * 100 * is_D * ds

B_Dxy = assemble(b_Dxy).vector().get_local()
B_Dt = assemble(b_Dt).vector().get_local()

# B matrices based on Lagrange
V_lmb = FunctionSpace(mesh, 'CG', 1)
lmb_N = TrialFunction(V_lmb)
v_N = TestFunction(V_lmb)

is_N = conditional(gt(ray, 1.5), 1, 0)
g_N = dot(v_q, n_ver) * lmb_N * is_N * ds

u_N = pow(x, 2) + pow(y, 2)
b_N = v_N * u_N * is_N * ds

petsc_gN = assemble(g_N, mat_type='aij').M.handle
G_N = np.array(petsc_gN.convert("dense").getDenseArray())

neumann_dofs = np.where(G_N.any(axis=0))[0]
G_N = G_N[:, neumann_dofs]

B_N = assemble(b_N).vector().get_local()[neumann_dofs]

n_lmb = G_N.shape[1]


def dae_closed_phs(t, y, yd):

    e_var = y[:n_e]
    lmb_var = y[n_e:]
    ed_var = yd[:n_e]

    res_e = MM @ ed_var - JJ @ e_var - G_N @ lmb_var - B_Dxy - B_Dt * t
    # res_lmb = - G_N.T @ e_var + B_N
    res_lmb = G_N.T @ invMM @ (JJ @ e_var + G_N @ lmb_var + B_Dxy + B_Dt * t)
    # res_lmb = G_N.T @ ed_var

    return np.concatenate((res_e, res_lmb))


invMM = la.inv(MM)
ep_0 = project(u_Dxy, Vp).vector().get_local()
eq_0 = project(as_vector([u_N*x/ray, u_N*y/ray]), Vq).vector().get_local()

e_0 = np.concatenate((ep_0, eq_0))
lmb_0 = la.solve(- G_N.T @ invMM @ G_N, G_N.T @ invMM @ (JJ @ e_0 + B_Dxy))

y0 = np.zeros(n_e + n_lmb)  # Initial conditions

de_0 = invMM @ (JJ @ e_0 + G_N @ lmb_0 + B_Dxy)
dlmb_0 = la.solve(- G_N.T @ invMM @ G_N, G_N.T @ invMM @ (JJ @ de_0 + B_Dt))

y0[:n_p] = ep_0
y0[n_p:n_V] = eq_0
y0[n_V:] = lmb_0

yd0 = np.zeros(n_e + n_lmb)  # Initial conditions
yd0[:n_V] = de_0
yd0[n_V:] = dlmb_0

# Create an Assimulo implicit problem
imp_mod = Implicit_Problem(dae_closed_phs, y0, yd0, name='dae_closed_pHs')

# Set the algebraic components
imp_mod.algvar = list(np.concatenate((np.ones(n_e), np.zeros(n_lmb))))

# Create an Assimulo implicit solver (IDA)
imp_sim = GLIMDA(imp_mod)  # Create a IDA solver

# Sets the paramters
imp_sim.atol = 1e-6  # Default 1e-6
imp_sim.rtol = 1e-6  # Default 1e-6
imp_sim.report_continuously = True
# imp_sim.maxh = 1e-6

# Simulate
t_final = 0.1
t_sol, y_sol, yd_sol = imp_sim.simulate(t_final)

t_ev = t_sol
n_ev = len(t_ev)
e_sol = y_sol[:, :n_e].T
lmb_sol = y_sol[:, n_e:].T

ep_sol = e_sol[:n_p, :]

w0 = np.zeros((n_p,))  # Should be integrated from e_q = grad(w)
w_sol = np.zeros(ep_sol.shape)
w_sol[:, 0] = w0
w_old = w_sol[:, 0]
n_ev = len(t_sol)
dt_vec = np.diff(t_sol)

for i in range(1, n_ev):
    w_sol[:, i] = w_old + 0.5 * (ep_sol[:, i - 1] + ep_sol[:, i]) * dt_vec[i-1]
    w_old = w_sol[:, i]


# maxZ = np.max(w_sol)
# minZ = np.min(w_sol)

maxZ = np.max(ep_sol)
minZ = np.min(ep_sol)
wfun_vec = []
w_fun = Function(Vp)

for i in range(n_ev):
    w_fun.vector()[:] = ep_sol[:, i]
    wfun_vec.append(interpolate(w_fun, Vp))

H_vec = np.zeros((n_ev,))

for i in range(n_ev):
    H_vec[i] = 0.5 * (e_sol[:, i].T @ MM @ e_sol[:, i])
fntsize = 16

fig = plt.figure()
plt.plot(t_ev, H_vec, 'b-')
plt.xlabel(r'{Time} (s)', fontsize=fntsize)
plt.ylabel(r'{Hamiltonian} (J)', fontsize=fntsize)
plt.title(r"Hamiltonian trend",
          fontsize=fntsize)
# plt.legend(loc='upper left')

# path_out = "./"

anim = animate2D(minZ, maxZ, wfun_vec, t_ev, xlabel = '$x[m]$', ylabel = '$y [m]$', \
                         zlabel='$p$', title='pressure')

plt.show()

# rallenty = 10
# fps = 20
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps= fps, metadata=dict(artist='Me'), bitrate=1800)
#
# anim.save(path_out + 'Kirchh_Rod.mp4', writer=writer)
