from firedrake import *
import numpy as np

np.set_printoptions(threshold=np.inf)

from scipy import linalg as la
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as inv_sp

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from tools_plotting.animate_platerod import animateInt2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from thin_plate.symplectic_integrators import StormerVerletGrad

matplotlib.rcParams['text.usetex'] = True


E = 7e10
nu = 0.35
h = 0.05 # 0.01
rho = 2700  # kg/m^3
D = E * h ** 3 / (1 - nu ** 2) / 12.

L = 1
l_x = L
l_y = L

n = 4 #int(input("N element on each side: "))

m_rod = 50
Jxx_rod = 1. / 12 * m_rod * L ** 2

k_sp1 = 10
k_sp2 = 10

r_sp1 = 0
r_sp2 = 0

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

# e_p = 1./(rho * h) * al_p
# e_q = bending_moment_vec(al_q)

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

j_gradgrad = inner(v_q, Gradgrad_vec(e_p)) * dx
j_gradgradIP = -inner(Gradgrad_vec(v_p), e_q) * dx

# j = j_gradgrad + j_gradgradIP  #
j_p = j_gradgrad
j_q = j_gradgradIP

Jp = assemble(j_p, mat_type='aij')
Mp = assemble(m_p, mat_type='aij')

Mq = assemble(m_q, mat_type='aij')
Jq = assemble(j_q, mat_type='aij')


petsc_j_p = Jp.M.handle
petsc_m_p = Mp.M.handle

petsc_j_q = Jq.M.handle
petsc_m_q = Mq.M.handle

Dp_pl = np.array(petsc_j_p.convert("dense").getDenseArray())
Mp_pl = np.array(petsc_m_p.convert("dense").getDenseArray())

Dq_pl = np.array(petsc_j_q.convert("dense").getDenseArray())
Mq_pl = np.array(petsc_m_q.convert("dense").getDenseArray())

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

v_wt, v_omn = TestFunction(Vu)

v_omn = dot(grad(v_p), n)
g = v_p * q_n * ds(1) + v_omn * M_nn * ds(1) + v_p * q_n * ds(2)

# Assemble the stiffness matrix and the mass matrix.
petsc_g = assemble(g, mat_type='aij').M.handle

# print(B_u.array().shape)
G_pl = np.array(petsc_g.convert("dense").getDenseArray())

bd_dofs = np.where(G_pl.any(axis=0))[0]# np.where(~np.all(B_in == 0, axis=0) == True) #

G_pl = G_pl[:, bd_dofs]
n_bd = len(bd_dofs)

# Splitting of matrices

# Force applied at the right boundary
x, y = SpatialCoordinate(mesh)
g = Constant(10)
A = Constant(10**6)
f_w = project(A*(y/10 + (y-l_y/2)**2), Vp) # project(A*sin(2*pi/l_y*y), Vp) #
# f_w = Expression("1000000*sin(2*pi*x[0])", degree=4)
b_p = v_p * f_w * dx             # v_p * f_w * ds(3) - v_p * f_w * ds(4)
#                                # -v_p * rho * h * g * dx #

Bp_pl = assemble(b_p, mat_type='aij').vector().get_local()

# Final Assemble

np_rod = 2
Mp_rod = np.diag([m_rod, Jxx_rod])
Mq_rod = np.diag([1./k_sp1, 1./k_sp2])

Dp_rod = np.array([[1, l_y/2], [1, -l_y/2]])
Dq_rod = - Dp_rod.T

r_v = r_sp1 + r_sp2
r_th = l_y**2/4*(r_sp1 - r_sp2)
r_vth = l_y/2*(r_sp1 - r_sp2)
R_rod = np.array([[r_v, r_vth], [r_vth, r_th]])


G_rodT = np.zeros((n_bd, 2))


G_rodT[:, 0] = assemble(- v_wt * ds(2)).vector().get_local()[bd_dofs]
G_rodT[:, 1] = assemble(- v_wt * (y - l_y/2) * ds(2)).vector().get_local()[bd_dofs]

G_rod = G_rodT.T

M_p = la.block_diag(Mp_pl, Mp_rod)
M_q = Mq_pl
D_p = np.concatenate((Dp_pl, np.zeros((n_Vq, 2))), axis = 1)
D_q = np.concatenate((Dq_pl, np.zeros((2, n_Vq))), axis = 0)

Mp_sp = csc_matrix(M_p)
invMp = inv_sp(Mp_sp)
invM_pl = invMp.toarray()
G_p = np.concatenate((G_pl, G_rod), axis = 0)
S_p = G_p @ la.inv(G_p.T @ invMp @ G_p) @ G_p.T @ invMp

np_tot = n_Vp + np_rod
Id_p = np.eye(np_tot)
P_p = Id_p - S_p

R_p = np.zeros((np_tot, np_tot))
F_p = np.zeros((np_tot, )); F_p[:n_Vp] = Bp_pl

solverSym = StormerVerletGrad(M_p, M_q, D_p, D_q, R_p, P_p, F_p)

t_0 = 0
dt = 1e-6
t_f = 1e-3
n_ev = 100

Aw = 0.001

e_pw_0 = Function(Vp)
e_pw_0.assign(project(Aw*x**2, Vp))
ep_pl0 = e_pw_0.vector().get_local() #
eq_pl0 = np.zeros((n_Vq))

ep_0 = np.zeros((np_tot,))
eq_0 = np.zeros((n_Vq,))

sol = solverSym.compute_sol(ep_0, eq_0, t_f, t_0 = t_0, dt = dt, n_ev = n_ev)

t_ev = sol.t_ev
ep_sol = sol.ep_sol
eq_sol = sol.eq_sol

dt_vec = np.diff(t_ev)

ep_pl = sol.ep_sol[:n_Vp, :]
ep_rod = sol.ep_sol[n_Vp: np_tot, :]

eq_pl = sol.eq_sol

w0_pl = np.zeros((n_Vp,))
w_pl = np.zeros(ep_pl.shape)
w_pl[:, 0] = w0_pl
w_pl_old = w_pl[:, 0]


for i in range(1, n_ev):
    w_pl[:, i] = w_pl_old + 0.5 * (ep_pl[:, i - 1] + ep_pl[:, i]) * dt_vec[i-1]
    w_pl_old = w_pl[:, i]

y_rod = np.array([0, 1])*l_y
x_rod = np.array([1, 1])*l_x

v_rod = np.zeros((len(x_rod), n_ev))
w_rod = np.zeros((len(x_rod), n_ev))
w_rod_old = w_rod[:, 0]

for i in range(n_ev):

    v_rod[:, i] = x_rod * ep_rod[0, i] + (y_rod - l_y / 2) * ep_rod[1, i]
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


Hpl_vec = np.zeros((n_ev,))
Hrod_vec = np.zeros((n_ev,))

for i in range(n_ev):
    Hpl_vec[i] = 0.5 * (ep_pl[:, i].T @ Mp_pl @ ep_pl[:, i] +  eq_pl[:, i].T @ Mq_pl @ eq_pl[:, i])
    Hrod_vec[i] = 0.5 * (ep_rod[:, i].T @ Mp_rod @ ep_rod[:, i])

fntsize = 16

fig = plt.figure()
plt.plot(t_ev, Hpl_vec, 'b-', label='Hamiltonian Plate (J)')
plt.plot(t_ev, Hrod_vec, 'r-', label='Hamiltonian Rod (J)')
plt.plot(t_ev, Hpl_vec + Hrod_vec, 'g-', label='Total Energy (J)')
plt.xlabel(r'{Time} (s)', fontsize = fntsize)
# plt.ylabel(r'{Hamiltonian} (J)', fontsize = fntsize)
plt.title(r"Hamiltonian trend",
          fontsize = fntsize)
plt.legend(loc='upper left')

path_out = "/home/a.brugnoli/Plots_Videos/Kirchhoff_plots/Simulations/Article_CDC/InterconnectionRod/"

# plt.savefig(path_out + "HamiltonianRod.eps", format="eps")

anim = animateInt2D(minZ, maxZ, wmm_pl_CGvec, x_rod, y_rod, w_rod_mm, \
             t_ev, xlabel = '$x[m]$', ylabel = '$y [m]$', \
                         zlabel = '$w [mm]$', z2label = '$w [mm]$ Rod', title = 'Vertical Displacement')

rallenty = 10
fps = 20
Writer = animation.writers['ffmpeg']
writer = Writer(fps= fps, metadata=dict(artist='Me'), bitrate=1800)

# anim.save(path_out + 'Kirchh_Rod.mp4', writer=writer)

plt.show()

# save_solutions = True
# if save_solutions:
#
#
#     matplotlib.rcParams['text.usetex'] = True
#
#     n_fig = 4
#     tol = 1e-6
#
#     for i in range(n_fig):
#         index = int(n_ev/n_fig*(i+1)-1)
#         w_fun = Function(Vp)
#         w_fun.vector()[:] = w_pl_mm[:, index]
#
#         Vp_CG = FunctionSpace(mesh, 'Lagrange', 3)
#         wmm_wCG = project(w_fun, Vp_CG)
#
#         from firedrake.plot import _two_dimension_triangle_func_val
#
#         triangulation, Z = _two_dimension_triangle_func_val(wmm_wCG, 10)
#         fig = plt.figure()
#
#         ax = fig.add_subplot(111, projection="3d")
#         ax.collections.clear()
#
#         surf_opts = {'cmap': cm.jet, 'linewidth': 0, 'antialiased': False} #, 'vmin': minZ, 'vmax': maxZ}
#         # lab = 'Time =' + '{0:.2e}'.format(t_ev[index])
#         tri_surf = ax.plot_trisurf(triangulation, Z, **surf_opts)
#         tri_line = ax.plot(x_rod, y_rod, w_rod_mm[:, index], linewidth = 5, label='Rod $w[mm]$', color='black')
#
#         tri_surf._facecolors2d = tri_surf._facecolors3d
#         tri_surf._edgecolors2d = tri_surf._edgecolors3d
#         ax.legend(handles=[tri_line[0]])
#         # fig.colorbar(surf)
#
#         ax.set_xbound(-tol, l_x + tol)
#         ax.set_xlabel('$x [m]$', fontsize=fntsize)
#
#         ax.set_ybound(-tol, l_y + tol)
#         ax.set_ylabel('$y [m]$', fontsize=fntsize)
#
#         ax.w_zaxis.set_major_locator(LinearLocator(10))
#         ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))
#
#         ax.set_zlabel('$w [mm]$' +'$(t=$' + '{0:.4f}'.format(t_ev[index]) + '$s)$', fontsize=fntsize)
#         ax.set_title('Vertical displacement', fontsize=fntsize)
#
#         ax.set_zlim3d(minZ - 0.01 * abs(minZ), maxZ + 0.01 * abs(maxZ))
#
#         plt.savefig(path_out + "SnapRod_t" + str(index + 1) + ".eps", format="eps")
#
#         plt.close()