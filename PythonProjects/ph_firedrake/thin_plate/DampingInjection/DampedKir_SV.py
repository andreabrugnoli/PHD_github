# Kirchhoff plate written with the port Hamiltonian approach

from firedrake import *
import numpy as np

np.set_printoptions(threshold=np.inf)

from scipy import linalg as la
import sys
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from tools_plotting.animate_surf import animate2D
import matplotlib.animation as animation
from matplotlib import cm

SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
rcParams['text.usetex'] = True

from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, csc_matrix


def Integration_DAE_SV2_Augmented(M, J, R, B_Dext, e0, Np, Nq, dt, t_fin, n_ev=1000):
    t_ev = np.linspace(0, t_fin, n_ev)

    M_sparse = csc_matrix(M)
    J_sparse = csc_matrix(J)
    R_sparse = csc_matrix(R)

    B_Dext_sparse = csc_matrix(B_Dext)

    SysJ = spsolve(M_sparse, J_sparse)

    BDMinvBDT = B_Dext_sparse.T @ spsolve(M_sparse, B_Dext_sparse)
    prefix = spsolve(M_sparse, B_Dext_sparse)

    Sys_AugJ = - prefix @ spsolve(BDMinvBDT, B_Dext.T) @ SysJ
    Sys_AUGJ = SysJ + Sys_AugJ

    Sys_AUGJ_pp = csc_matrix(Sys_AUGJ[:Np, :Np])
    Sys_AUGJ_qq = csc_matrix(Sys_AUGJ[Np:, Np:])
    Sys_AUGJ_qp = csc_matrix(Sys_AUGJ[Np:, :Np])
    Sys_AUGJ_pq = csc_matrix(Sys_AUGJ[:Np, Np:])

    SysJR = spsolve(M_sparse, J_sparse - R_sparse)
    Sys_AugJR = - prefix @ spsolve(BDMinvBDT, B_Dext.T) @ SysJR
    Sys_AUGJR = SysJR + Sys_AugJR

    Sys_AUGJR_pp = csc_matrix(Sys_AUGJR[:Np, :Np])
    Sys_AUGJR_qq = csc_matrix(Sys_AUGJR[Np:, Np:])
    Sys_AUGJR_qp = csc_matrix(Sys_AUGJR[Np:, :Np])
    Sys_AUGJR_pq = csc_matrix(Sys_AUGJR[:Np, Np:])


    Nsys = Np + Nq

    Nt = int(t_fin / dt) + 1
    if n_ev > Nt:
        raise ValueError("Choose less evaluation points")

    A_pq = np.zeros((Nsys, n_ev))

    A_pq[:, 0] = e0

    App = np.zeros((Np, ))
    Aqq = np.zeros((Nq, ))

    Ap_old= A_pq[:Np, 0]
    Aq_old = A_pq[Np:, 0]

    Ap_new = np.zeros((Np,))
    Aq_new = np.zeros((Nq,))

    k = 1

    for i in range(Nt):

        t = dt * (i + 1)

        if t < 0.2 * t_fin:
            # Aqq[:] = Aq_old + dt / 2 * (Sys_AUGJ_qp @ Ap_old + Sys_AUGJ_qq @ Aq_old)
            #
            # Ap_new[:] = Ap_old + dt * (Sys_AUGJ_pp @ Ap_old + Sys_AUGJ_pq @ Aqq)
            #
            # Aq_new[:] = Aq_old + dt/2 * (Sys_AUGJ_qq @ Aqq + Sys_AUGJ_qp @ Ap_new)

            App[:] = Ap_old + dt / 2 * (Sys_AUGJ_pq @ Aq_old + Sys_AUGJ_pp @ Ap_old)

            Aq_new[:] = Aq_old + dt * (Sys_AUGJ_qq @ Aq_old + Sys_AUGJ_qp @ App)

            Ap_new[:] = App + dt / 2 * (Sys_AUGJ_pq @ Aq_new + Sys_AUGJ_pp @ App)
        else:
            # Aqq[:] = Aq_old + dt / 2 * (Sys_AUGJR_qp @ Ap_old + Sys_AUGJR_qq @ Aq_old)
            #
            # Ap_new[:] = Ap_old + dt * (Sys_AUGJR_pp @ Ap_old + Sys_AUGJR_pq @ Aqq)
            #
            # Aq_new[:] = Aq_old + dt/2 * (Sys_AUGJR_qq @ Aqq + Sys_AUGJR_qp @ Ap_new)

            App[:] = Ap_old + dt / 2 * (Sys_AUGJR_pq @ Aq_old + Sys_AUGJR_pp @ Ap_old)

            Aq_new[:] = Aq_old + dt * (Sys_AUGJR_qq @ Aq_old + Sys_AUGJR_qp @ App)

            Ap_new[:] = App + dt / 2 * (Sys_AUGJR_pq @ Aq_new + Sys_AUGJR_pp @ App)

        Ap_old = Ap_new
        Aq_old = Aq_new

        if k < n_ev and t >= t_ev[k]:
            A_pq[:Np, k] = Ap_new
            A_pq[Np:, k] = Aq_new
            k = k + 1
        elif k == n_ev:
            break

        # Progress bar
        perct = int(i / (Nt - 1) * 100)
        bar = ('Time-stepping SV2Augmented : |' + '#' * int(perct / 2) + ' ' + str(perct) + '%' + '|')
        sys.stdout.write('\r' + bar)

    print('\n intergation completed \n')

    return A_pq


E = 7e10
nu = 0.35
h = 0.05  # 0.01
rho = 2700  # kg/m^3
D = E * h ** 3 / (1 - nu ** 2) / 12.

L = 1
l_x = L
l_y = L

z_imp = 0.1

n = 6  # int(input("N element on each side: "))

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
    return as_vector([u.dx(0).dx(0), u.dx(1).dx(1), 2 * u.dx(0).dx(1)])


def tensor_Div_vec(MM):
    return as_vector([MM[0].dx(0) + MM[2].dx(1), MM[2].dx(0) + MM[1].dx(1)])


# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
L = 1
l_x = L
l_y = L
n_x, n_y = n, n
mesh = UnitSquareMesh(n_x, n_y, quadrilateral=False)

# plot(mesh)
# plt.show()
name_FEp = 'Argyris'
name_FEq = 'DG'

Vp = FunctionSpace(mesh, name_FEp, 5)
Vq = VectorFunctionSpace(mesh, name_FEq, 3, dim=3)

V =Vp*Vq
n_Vp = Vp.dim()
n_Vq = Vq.dim()

v = TestFunction(V)
v_p, v_q = split(v)

e = TrialFunction(V)
e_p, e_q =split(e)

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

M = assemble(m, mat_type='aij')
petsc_m = M.M.handle
MM = np.array(petsc_m.convert("dense").getDenseArray())

j_gradgrad = inner(v_q, Gradgrad_vec(e_p)) * dx
j_gradgradIP = -inner(Gradgrad_vec(v_p), e_q) * dx

j = j_gradgrad + j_gradgradIP  #

J = assemble(j, mat_type='aij')
petsc_j = J.M.handle
JJ = np.array(petsc_j.convert("dense").getDenseArray())

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

# b_bd = v_p * q_n * ds(2) + v_omn * M_nn * ds(2)
b_bd = v_p * q_n * ds(2) + v_omn * M_nn * ds(2) \
       + v_p * q_n * ds(3) + v_omn * M_nn * ds(3) + v_p * q_n * ds(4) + v_omn * M_nn * ds(4)
b_mul = v_p * q_n * ds(1) + v_omn * M_nn * ds(1)

# Assemble the stiffness matrix and the mass matrix.
G = assemble(b_mul, mat_type='aij')
petsc_g = G.M.handle
GG = np.array(petsc_g.convert("dense").getDenseArray())

bd_dofs_mul = np.where(GG.any(axis=0))[0]
GG = GG[:, bd_dofs_mul]

B = assemble(b_bd, mat_type='aij')
petsc_b = B.M.handle

Bbd_pl = np.array(petsc_b.convert("dense").getDenseArray())

bd_dofs_ctrl = np.where(Bbd_pl.any(axis=0))[0]
bd_dofs_ctrl = list(set(bd_dofs_ctrl).difference(set(bd_dofs_ctrl).intersection(bd_dofs_mul)))

Bbd_pl = Bbd_pl[:, bd_dofs_ctrl]

n_ctrl = len(bd_dofs_ctrl)


v_wt, v_omn = TestFunction(Vu)

m_partial = v_wt * q_n * ds(2) + v_omn * M_nn * ds(2) \
            + v_wt * q_n * ds(3) + v_omn * M_nn * ds(3) + v_wt * q_n * ds(4) + v_omn * M_nn * ds(4)

M_part = assemble(m_partial, mat_type='aij')
petsc_mpart = M_part.M.handle

M_partial = np.array(petsc_mpart.convert("dense").getDenseArray())
M_partial = M_partial[:, bd_dofs_ctrl][bd_dofs_ctrl, :]

Z = la.inv(M_partial) * z_imp

RR = Bbd_pl @ Z @ Bbd_pl.T

n_mul = len(bd_dofs_mul)

# Final Assemble

t_0 = 0
dt = 1e-6

t_f = 5
n_ev = 1000

x, y = SpatialCoordinate(mesh)

e_pw_0 = Function(Vp)
e_pw_0.assign(project(x ** 2*cos(2*pi*y), Vp))
ep_0 = e_pw_0.vector().get_local()
eq_0 = np.zeros((n_Vq))

e0 = np.concatenate((ep_0, eq_0))

sol = Integration_DAE_SV2_Augmented(MM, JJ, RR, GG, e0, n_Vp, n_Vq, dt, t_f, n_ev)

path_res = "/home/a.brugnoli/LargeFiles/results_DampKirchh/"

t_ev = np.linspace(0, t_f, n_ev)
ep_sol = sol[:n_Vp]
eq_sol = sol[n_Vp:]

np.save(path_res + "t_ev", t_ev)
np.save(path_res + "ep_sol", ep_sol)
np.save(path_res + "eq_sol", eq_sol)

n_ev = len(t_ev)

n_sol = len(t_ev)
w0 = np.zeros((n_Vp,))
w = np.zeros(ep_sol.shape)
w[:, 0] = w0
w_old = w[:, 0]

dt_ev = np.diff(t_ev)
for i in range(1, n_ev):
    w[:, i] = w_old + 0.5 * (ep_sol[:, i - 1] + ep_sol[:, i]) * dt_ev[i - 1]
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

if matplotlib.is_interactive():
    plt.ioff()
plt.close('all')

H_vec = np.zeros((n_ev,))

for i in range(n_ev):
    H_vec[i] = 0.5 * (np.transpose(ep_sol[:, i]) @ MM[:n_Vp, :n_Vp] @ ep_sol[:, i]\
                      + np.transpose(eq_sol[:, i]) @ MM[n_Vp:, n_Vp:] @ eq_sol[:, i])

matplotlib.rcParams['text.usetex'] = True
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%1.3f'))
plt.plot(t_ev, H_vec, 'b-')
plt.xlabel(r'{Time} $\mathrm{[s]}$')
plt.ylabel(r'{Hamiltonian} $\mathrm{[J]}$')
plt.title(r"Hamiltonian")
# plt.legend(loc='upper left')
path_out = "/home/a.brugnoli/Plots/Python/Plots/Kirchhoff_plots/Simulations/Article_CDC/DampingInjection2/"

plt.savefig(path_out + "Hamiltonian.eps", format="eps")

anim = animate2D(minZ, maxZ, wmm_CGvec, t_ev, xlabel='$x \;  \mathrm{[m]}$', ylabel='$y \;  \mathrm{[m]}$', \
                 zlabel='$w \;  \mathrm{[mm]}$', title='Vertical Displacement')

rallenty = 10
Writer = animation.writers['ffmpeg']
# writer = Writer(fps=int(n_ev/(t_f*rallenty)), metadata=dict(artist='Me'), bitrate=1800)
writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=1800)

anim.save(path_out + 'Kirchh_Damped.mp4', writer=writer)

plot_solutions = True
if plot_solutions:

    n_fig = 100
    tol = 1e-6

    for i in range(n_fig):
        index = int(n_ev / n_fig * (i + 1) - 1)
        w_fun = Function(Vp)
        w_fun.vector()[:] = w_mm[:, index]

        Vp_CG = FunctionSpace(mesh, 'Lagrange', 3)
        wmm_wCG = project(w_fun, Vp_CG)

        from firedrake.plot import _two_dimension_triangle_func_val

        triangulation, Z = _two_dimension_triangle_func_val(wmm_wCG, 10)
        fig = plt.figure()

        ax = fig.add_subplot(111, projection="3d")
        ax.collections.clear()

        surf_opts = {'cmap': cm.jet, 'linewidth': 0, 'antialiased': False}  # , 'vmin': minZ, 'vmax': maxZ}
        lab = 'Time =' + '{0:.2f}'.format(t_ev[index])
        surf = ax.plot_trisurf(triangulation, Z, **surf_opts)
        # fig.colorbar(surf)

        ax.set_xbound(-tol, l_x + tol)
        ax.set_xlabel('$x \;  \mathrm{[m]}$')

        ax.set_ybound(-tol, l_y + tol)
        ax.set_ylabel('$y \;  \mathrm{[m]}$')

        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2f'))

        ax.set_zlabel('$w \;  \mathrm{[mm]}$')
        ax.set_title('Vertical displacement ' + '$(t=$' + '{0:.2f}'.format(t_ev[index]) + '$\mathrm{[s]})$')
        # ax.set_title('Vertical displacement ' +'$(t=$' + str(t_ev[index]) + '$\mathrm{[s]})$')

        ax.set_zlim3d(minZ - 0.01 * abs(minZ), maxZ + 0.01 * abs(maxZ))

        plt.savefig(path_out + "Snapshot_t" + str(index + 1) + ".eps", format="eps")



