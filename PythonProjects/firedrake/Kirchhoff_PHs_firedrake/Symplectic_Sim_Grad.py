# Kirchhoff plate written with the port Hamiltonian approach

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
from AnimateSurfFiredrake import animate2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

matplotlib.rcParams['text.usetex'] = True


E = 7e10
nu = 0.35
h = 0.05
rho = 2700  # kg/m^3
D = E * h ** 3 / (1 - nu ** 2) / 12.

L = 1
l_x = L
l_y = L

n_sim = 1

n = 5 #int(input("N element on each side: "))

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
# m = m_p + m_q

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

D_p = np.array(petsc_j_p.convert("dense").getDenseArray())
M_p = np.array(petsc_m_p.convert("dense").getDenseArray())

D_q = np.array(petsc_j_q.convert("dense").getDenseArray())
M_q = np.array(petsc_m_q.convert("dense").getDenseArray())

# Dirichlet Boundary Conditions and related constraints
# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1
bc_input = input('Select Boundary Condition:')   #'SSSS'

bc_1, bc_3, bc_2, bc_4 = bc_input

bc_dict = {1: bc_1, 2: bc_2, 3: bc_3, 4: bc_4}

n = FacetNormal(mesh)
# s = as_vector([-n[1], n[0]])

V_qn = FunctionSpace(mesh, 'Lagrange', 2)
V_Mnn = FunctionSpace(mesh, 'Lagrange', 2)

Vu = V_qn * V_Mnn

q_n, M_nn = TrialFunction(Vu)

v_omn = dot(grad(v_p), n)

g_vec = []
for key,val in bc_dict.items():
    if val == 'C':
        g_vec.append( v_p * q_n * ds(key) + v_omn * M_nn * ds(key))
    elif val == 'S':
        g_vec.append(v_p * q_n * ds(key))

g = sum(g_vec)
# Assemble the stiffness matrix and the mass matrix.

petsc_g = assemble(g, mat_type='aij').M.handle
G_p = np.array(petsc_g.convert("dense").getDenseArray())
# B_out = np.array(petsc_b_y.convert("dense").getDenseArray())

boundary_dofs = np.where(G_p.any(axis=0))[0]  # np.where(~np.all(B_in == 0, axis=0) == True) #
G_p = G_p[:, boundary_dofs]

# Splitting of matrices

# Force applied at the right boundary
x, y = SpatialCoordinate(mesh)
g = Constant(10)
A = Constant(10**5)
f_w = project(A*x, Vp) # project(1000000*sin(6*pi/l_y*x), Vp) #
b_p1 = -v_p * rho * h * g * dx #
b_p2 = v_p * f_w * ds(3) + v_p * f_w * ds(4) + v_p * f_w * ds(2)


if n_sim == 1:
    b_p = b_p1
else: b_p = b_p2
F_p = assemble(b_p, mat_type='aij').vector().get_local()

# Final Assemble
Mp_sp = csc_matrix(M_p)

invMp = inv_sp(Mp_sp)
invM_pl = invMp.toarray()

S_p = G_p @ la.inv(G_p.T @ invMp @ G_p) @ G_p.T @ invMp

Id_p = np.eye(n_Vp)
P_p = Id_p - S_p


x, y = SpatialCoordinate(mesh)
Aw = 0.001

e_pw_0 = Function(Vp)
e_pw_0.assign(project(A*x**2, Vp))
ep_0 = np.zeros((n_Vp)) # e_pw_0.vector().get_local() #
eq_0 = np.zeros((n_Vq))

from symplectic_integrators import StormerVerletGrad

R_p = np.zeros((n_Vp, n_Vp))

solverSym = StormerVerletGrad(M_p, M_q, D_p, D_q, R_p, P_p, F_p)


t_0 = 0
dt = 1e-7
t_f = 1e-3
n_ev = 100

sol = solverSym.compute_sol(ep_0, eq_0, t_f, t_0 = t_0, dt = dt, n_ev = n_ev)

t_ev = sol.t_ev
ep_sol = sol.ep_sol
eq_sol = sol.eq_sol

n_ev = len(t_ev)

n_sol = len(t_ev)
w0 = np.zeros((n_Vp,))
w = np.zeros(ep_sol.shape)
w[:, 0] = w0
w_old = w[:, 0]

dt_ev = np.diff(t_ev)

h_Ep = Function(Vp)
Ep = np.zeros((n_ev,))
for i in range(1, n_ev):
    w[:, i] = w_old + 0.5 * (ep_sol[:, i - 1] + ep_sol[:, i]) * dt_ev[i-1]
    w_old = w[:, i]
    h_Ep.vector()[:] = np.ascontiguousarray(w[:, i], dtype='float')
    Ep[i] = assemble(rho * g * h * h_Ep * dx)

if n_sim ==1:
    w_mm = w * 1000000
else: w_mm = w * 1000

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

# if matplotlib.is_interactive():
#     plt.ioff()
# plt.close('all')

fntsize = 16
H_vec = np.zeros((n_ev,))

for i in range(n_ev):
    H_vec[i] = 0.5 * (np.transpose(ep_sol[:, i]) @ M_p @ ep_sol[:, i] \
        + np.transpose(eq_sol[:, i]) @ M_q @ eq_sol[:, i])

fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2g'))
plt.plot(t_ev, H_vec, 'b-', label='Hamiltonian Plate (J)')
# plt.plot(t_ev, Ep, 'r-', label = 'Potential Energy (J)')
# plt.plot(t_ev, H_vec + Ep, 'g-', label = 'Total Energy (J)')
plt.xlabel(r'{Time} (s)', fontsize=fntsize)
# plt.ylabel(r'{Hamiltonian} (J)', fontsize=fntsize)
plt.title(r"Hamiltonian trend",  fontsize=fntsize)
# plt.legend(loc='upper left')

path_out = "/home/a.brugnoli/Plots_Videos/Videos/Kirchhoff_Plate/"
# plt.savefig(path_out + "Sim" +str(n_sim) + "Hamiltonian.eps", format="eps")

anim = animate2D(minZ, maxZ, wmm_CGvec, t_ev, xlabel = '$x[m]$', ylabel = '$y [m]$', \
                         zlabel = '$w [\mu m]$', title = 'Vertical Displacement')


# rallenty = 10
# fps = int(n_ev/(t_f*rallenty))
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
# anim.save(path_out + "Kirchh_Gravity.mp4", writer=writer)

plt.show()

plot_solutions = False
if plot_solutions:


    matplotlib.rcParams['text.usetex'] = True

    n_fig = 4
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
        # lab = 'Time =' + '{0:.2e}'.format(t_ev[index])
        surf = ax.plot_trisurf(triangulation, Z, **surf_opts)
        # fig.colorbar(surf)

        ax.set_xbound(-tol, l_x + tol)
        ax.set_xlabel('$x [m]$', fontsize=fntsize)

        ax.set_ybound(-tol, l_y + tol)
        ax.set_ylabel('$y [m]$', fontsize=fntsize)

        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

        ax.set_zlabel('$w [mm]$', fontsize=fntsize)
        ax.set_title('Vertical displacement ', fontsize=fntsize)

        # ax.set_title('Vertical displacement ' +'$(t=$' + '{0:.2e}'.format(t_ev[index]) + '$s)$', fontsize=fntsize)

        ax.set_zlim3d(minZ - 0.01 * abs(minZ), maxZ + 0.01 * abs(maxZ))

        plt.savefig(path_out + "Sim" + str(n_sim) + "t" + str(index + 1) + ".eps", format="eps")
