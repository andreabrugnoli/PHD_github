from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy import linalg as la
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as inv_sp
from scipy import integrate

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from tools_plotting.animate_surf import animate2D
import matplotlib.animation as animation
from matplotlib import cm

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams['text.usetex'] = True

path_out = "/home/a.brugnoli/Plots/Python/Plots/Kirchhoff_plots/Simulations/Article_CDC/DampingInjection2/"


E = 7e10
nu = 0.35
h = 0.05 # 0.01
rho = 2700  # kg/m^3
D = E * h ** 3 / (1 - nu ** 2) / 12.

n = 4
deg_bd = 2
z_imp = 75

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

name_FEp = 'Bell'
name_FEq = 'DG'
deg_p = 5
deg_q = 3

# if name_FEp == 'Morley':
#     deg_p = 2
# elif name_FEp == 'Hermite':
#     deg_p = 3
# elif name_FEp == 'Argyris' or name_FEp == 'Bell':
#     deg_p = 5
#
# if name_FEq == 'Morley':
#     deg_q = 2
# elif name_FEq == 'Hermite':
#     deg_q = 3
# elif name_FEq == 'Argyris' or name_FEq == 'Bell':
#     deg_q = 5

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

b_bd = v_p * q_n * ds(2) + v_omn * M_nn * ds(2) \
       + v_p * q_n * ds(3) + v_omn * M_nn * ds(3) + v_p * q_n * ds(4) + v_omn * M_nn * ds(4)
b_mul = v_p * q_n * ds(1) + v_omn * M_nn * ds(1)

# Assemble the stiffness matrix and the mass matrix.
B = assemble(b_bd,  mat_type='aij')
petsc_b = B.M.handle

Bbd_pl = np.array(petsc_b.convert("dense").getDenseArray())

bd_dofs_ctrl = np.where(Bbd_pl.any(axis=0))[0]
Bbd_pl = Bbd_pl[:, bd_dofs_ctrl]

n_ctrl = len(bd_dofs_ctrl)

Z = np.eye(n_ctrl) * z_imp

R_pl = Bbd_pl @ Z @ Bbd_pl.T

G = assemble(b_mul,  mat_type='aij')
petsc_g = G.M.handle
G_pl = np.array(petsc_g.convert("dense").getDenseArray())

bd_dofs_mul = np.where(G_pl.any(axis=0))[0]
G_pl = G_pl[:, bd_dofs_mul]

n_mul = len(bd_dofs_mul)
# Force applied
x, y = SpatialCoordinate(mesh)
A = Constant(10**5)
f_w = project(A*x, Vp)
bp_pl = v_p * f_w * dx
#

Bf_pl = assemble(bp_pl, mat_type='aij').vector().get_local()

# Final Assemble

M_sp = csc_matrix(M_pl)
invM_pl = inv_sp(M_sp)
invM_pl = invM_pl.toarray()

S = G_pl @ la.inv(G_pl.T @ invM_pl @ G_pl) @ G_pl.T @ invM_pl

n_tot = n_pl
I = np.eye(n_tot)
P = I - S

Jsys = P @ J_pl
Rsys = P @ R_pl
Fsys = P @ Bf_pl

t0 = 0.0
t_fin = 0.001
n_t = 100
t_span = [t0, t_fin]

def sys(t,y):

    print(t/t_fin*100)

    if t < 0.2 * t_fin:
        dydt = invM_pl @ (Jsys @ y)

    else: dydt = invM_pl @ ((Jsys - Rsys) @ y )

    return dydt

Aw = 0.001
e_p0 = Function(Vp)
e_p0.assign(project(Aw*x**2, Vp))
y0 = np.zeros(n_tot,)
y0[:n_Vp] = e_p0.vector().get_local()

t_ev = np.linspace(t0, t_fin, num = n_t)

sol = integrate.solve_ivp(sys, t_span, y0, method='RK45', vectorized=False, t_eval = t_ev)

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


fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2g'))
plt.plot(t_ev, Hpl_vec, 'b-')
plt.xlabel(r'{Time} $\mathrm{[s]}$')
plt.ylabel(r'{Hamiltonian} $\mathrm{[J]}$')
plt.title(r"Hamiltonian")
# plt.legend(loc='upper left')

plt.savefig(path_out + "Hamiltonian.eps", format="eps")

plot_solutions = False
if plot_solutions:


    n_fig = 8
    tol = 1e-6

    for i in range(n_fig):
        index = int(n_ev/n_fig*(i+1)-1)
        w_fun = Function(Vp)
        w_fun.vector()[:] = w_mm[:, index]

        Vp_CG = FunctionSpace(mesh, 'Lagrange', 3)
        wmm_wCG = project(w_fun, Vp_CG)

        from firedrake.plot import _two_dimension_triangle_func_val

        triangulation, Z = _two_dimension_triangle_func_val(wmm_wCG, 10)
        fig = plt.figure()

        ax = fig.add_subplot(111, projection="3d")
        ax.collections.clear()

        surf_opts = {'cmap': cm.jet, 'linewidth': 0, 'antialiased': False} #, 'vmin': minZ, 'vmax': maxZ}
        lab = 'Time =' + '{0:.2f}'.format(t_ev[index])
        surf = ax.plot_trisurf(triangulation, Z, **surf_opts)
        # fig.colorbar(surf)

        ax.set_xbound(-tol, l_x + tol)
        ax.set_xlabel('$x \;  \mathrm{[m]}$')

        ax.set_ybound(-tol, l_y + tol)
        ax.set_ylabel('$y \;  \mathrm{[m]}$')

        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2f'))

        ax.set_zlabel('$w \;  \mathrm{[\mu m]}$')
        ax.set_title('Vertical displacement ' +'$(t=$' + '{0:.2f}'.format(t_ev[index]) + '$\mathrm{[s]})$')
        # ax.set_title('Vertical displacement ' +'$(t=$' + str(t_ev[index]) + '$\mathrm{[s]})$')

        ax.set_zlim3d(minZ - 0.01 * abs(minZ), maxZ + 0.01 * abs(maxZ))

        plt.savefig(path_out + "Snapshot_t" + str(index + 1) + ".eps", format="eps")



anim = animate2D(minZ, maxZ, wmm_CGvec, t_ev, xlabel = '$x[m]$', ylabel = '$y [m]$', \
                         zlabel = '$w [mm]$', title = 'Vertical Displacement')

Writer = animation.writers['ffmpeg']
writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)
# anim.save('Kirchh_NoRod.mp4', writer=writer)

plt.show()




# eigenvalues, eigvectors = la.eig(J_pl - R_pl, M_pl)
# real_eig = np.real(eigenvalues)
# imag_eig = np.imag(eigenvalues)
#
# from matplotlib.ticker import FormatStrFormatter
# fig, ax = plt.subplots()
# # Rewrite the y labels
# y_labels = ax.get_yticks()
# ax.xaxis.set_major_formatter(FormatStrFormatter('%0.0g'))
# ax.yaxis.set_major_formatter(FormatStrFormatter('%0.0g'))
#
# plt.plot(real_eig, imag_eig, 'b+')
# bottom, top = plt.ylim()
# plt.plot([0, 0], [bottom, top], 'r')  # plot x and y using blue circle markers
#
# fontsize = 15
# plt.xlabel(r'$\Re{(\lambda)}$', fontsize=fontsize)
# plt.ylabel(r'$\Im{(\lambda)}$', fontsize=fontsize)
#
# plt.show()

