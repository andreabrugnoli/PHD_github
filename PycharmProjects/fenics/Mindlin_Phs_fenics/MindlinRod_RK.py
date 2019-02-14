# Mindlin plate written with the port Hamiltonian approach

from fenics import *
import numpy as np

np.set_printoptions(threshold=np.inf)
# import mshr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy import linalg as la
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as inv_sp
from scipy import integrate

n = 5
deg = 2

E = (7e10)
nu = (0.35)
h = (0.1)
rho = (2700)  # kg/m^3
k = 0.8601  # 5./6. #
L = 1

m_rod = 100
Jxx_rod = 1. / 12 * m_rod * L ** 2

k_sp1 = 10
k_sp2 = 10

r_sp1 = 10
r_sp2 = 10
# Plate bending stiffness :math:`D=\dfrac{Eh^3}{12(1-\nu^2)}` and shear stiffness :math:`F = \kappa Gh`
# with a shear correction factor :math:`\kappa = 5/6` for a homogeneous plate
# of thickness :math:`h`::
# E = Constant(7e10)
# nu = Constant(0.3)
# h = Constant(0.2)
# rho = Constant(2000)  # kg/m^3
# k = Constant(5. / 6.)

D = E * h ** 3 / (1 - nu ** 2) / 12.
G = E / 2 / (1 + nu)
F = G * h * k

phi_rot = (rho * h ** 3) / 12.

# Useful Matrices

D_b = as_tensor([
    [D, D * nu, 0],
    [D * nu, D, 0],
    [0, 0, D * (1 - nu) / 2]
])

fl_rot = 12. / (E * h ** 3)
C_b = as_tensor([
    [fl_rot, -nu * fl_rot, 0],
    [-nu * fl_rot, fl_rot, 0],
    [0, 0, fl_rot * (1 + nu) / 2]
])


# Operators and functions
def gradSym(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
    # return sym(nabla_grad(u))


def strain2voigt(eps):
    return as_vector([eps[0, 0], eps[1, 1], 2 * eps[0, 1]])


def voigt2stress(S):
    return as_tensor([[S[0], S[2]], [S[2], S[1]]])


def bending_moment(u):
    return voigt2stress(dot(D_b, strain2voigt(u)))


def bending_curv(u):
    return voigt2stress(dot(C_b, strain2voigt(u)))


# The unit square mesh is divided in :math:`N\times N` quadrilaterals::

L = 1
l_x = L
l_y = L

n_x, n_y = n, n
mesh = RectangleMesh(Point(0, 0), Point(L, L), n_x, n_y, "right/left")

# domain = mshr.Rectangle(Point(0, 0), Point(l_x, l_y))
# mesh = mshr.generate_mesh(domain, n, "cgal")

d = mesh.geometry().dim()


# plot(mesh)
# plt.show()


class Left(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 0.0) < DOLFIN_EPS and on_boundary


class Right(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - l_x) < DOLFIN_EPS and on_boundary


class Lower(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1] - 0.0) < DOLFIN_EPS and on_boundary


class Upper(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1] - l_y) < DOLFIN_EPS and on_boundary


class AllBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


# Boundary conditions on displacement
all_boundary = AllBoundary()
# Boundary conditions on rotations
left = Left()
right = Right()
lower = Lower()
upper = Upper()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)
lower.mark(boundaries, 3)
upper.mark(boundaries, 4)

dx = Measure('dx')
ds = Measure('ds', subdomain_data=boundaries)

# Finite element defition

P_pw = FiniteElement('CG', triangle, deg)
P_pth = VectorElement('CG', triangle, deg)
P_qth = TensorElement('CG', triangle, deg, symmetry=True)
P_qw = VectorElement('CG', triangle, deg)

elem = MixedElement([P_pw, P_pth, P_qth, P_qw])

V = FunctionSpace(mesh, elem)


n_pl = V.dim()
dofV_x = V.tabulate_dof_coordinates().reshape((-1, d))

dofs_Vpw = V.sub(0).dofmap().dofs()
dofs_Vpth = V.sub(1).dofmap().dofs()

dofs_Vqth = V.sub(2).dofmap().dofs()
dofs_Vqw = V.sub(3).dofmap().dofs()

v = TestFunction(V)
v_pw, v_pth, v_qth, v_qw = split(v)


e = TrialFunction(V)
e_pw, e_pth, e_qth, e_qw = split(e)


al_pw = rho * h * e_pw
al_pth = (rho * h ** 3) / 12. * e_pth
al_qth = bending_curv(e_qth)
al_qw = 1. / F * e_qw

m_p = inner(v_pw, al_pw) * dx + inner(v_pth, al_pth) * dx
m_q = inner(v_qth, al_qth) * dx + inner(v_qw, al_qw) * dx

m = m_p + m_q
# j_div = v_pw * div(e_qw) * dx
# j_divIP = -div(v_qw) * e_pw * dx
#
# j_divSym = dot(v_pth, div(e_qth)) * dx
# j_divSymIP = -dot(div(v_qth), e_pth) * dx

j_grad = dot(v_qw, grad(e_pw)) * dx
j_gradIP = -dot(grad(v_pw), e_qw) * dx

j_gradSym = inner(v_qth, gradSym(e_pth)) * dx
j_gradSymIP = -inner(gradSym(v_pth), e_qth) * dx

j_Id = dot(v_pth, e_qw) * dx
j_IdIP = -dot(v_qw, e_pth) * dx

# j_alldiv_q = j_div + j_divSym + j_Id
# j_alldiv_p = j_IdIP + j_divIP + j_divSymIP

j_allgrad = j_grad + j_gradSym + j_IdIP + j_gradIP + j_gradSymIP + j_Id


j = j_allgrad


# Dirichlet Boundary Conditions and related constraints

n = FacetNormal(mesh)
s = as_vector([ -n[1], n[0] ])

P_qn = FiniteElement('CG', triangle, deg)
P_Mnn = FiniteElement('CG', triangle, deg)
P_Mns = FiniteElement('CG', triangle, deg)

element_u = MixedElement([P_qn, P_Mnn, P_Mns])

Vu = FunctionSpace(mesh, element_u)

q_n, M_nn, M_ns = TrialFunction(Vu)


v_omn = dot(v_pth, n)
v_oms = dot(v_pth, s)

b_l = v_pw * q_n * ds(1) + v_oms * M_ns * ds(1) + v_omn * M_nn * ds(1)
b_r = v_pw * q_n * ds(2) + v_oms * M_ns * ds(2)

J, M, B = PETScMatrix(), PETScMatrix(), PETScMatrix()

J = assemble(j)
M = assemble(m)
B = assemble(b_l + b_r)

M_pl = M.array()
J_pl = J.array()
R_pl = np.zeros((n_pl, n_pl))

G_pl  = B.array()

boundary_dofs = np.where(G_pl.any(axis=0))[0]
G_pl = G_pl[:,boundary_dofs]

n_mul = len(boundary_dofs)
# Force applied at the right boundary
A = Constant(10**3)
f_w = Expression("A*( pow((x[1]-l_y/2), 2) + x[1]/10)", degree=4, l_y=l_y, A = A)  #
b_f = v_pw * f_w * dx  # ds(3) - v_pw * f_w *  ds(4)
Bf_pl = assemble(b_f).get_local()
# ind_f = np.where(dofV_x[:, 0].any == 1)  # and (dofVp_x[:,1].any == 0 or dofVp_x[:,1].any == 1))
#
# for index in ind_f:
#     B_p[index] = 0

# Final Assemble
r_v = r_sp1 + r_sp2
r_th = l_y**2/4*(r_sp1 - r_sp2)
r_vth = l_y/2*(r_sp1 - r_sp2)


Mp_rod = np.diag([m_rod, Jxx_rod])
Mq_rod = np.diag([1./k_sp1, 1./k_sp2])

M_rod = Mp_rod #la.block_diag(Mp_rod, Mq_rod)

#Dp_rod = np.array([[1, l_y/2], [1, -l_y/2]])
# Dq_rod = np.array([[-1, -1], [l_y/2, -l_y/2]])
# Dq_rod = - Dp_rod.T
n_rod = 2
J_rod = np.zeros((n_rod, n_rod))
# J_rod[:2, 2:4] = Dq_rod
# J_rod[2:4, :2] = Dp_rod

R_rod = np.zeros((n_rod, n_rod))
# Rp_rod = np.array([[r_v, r_vth], [r_vth, r_th]])
# R_rod[:2, :2] = la.block_diag(Rp_rod, np.zeros((2, 2)))

Bf_rod = np.zeros((n_rod,))

v_wt, v_omn, v_oms = TestFunctions(Vu)
C = np.zeros((n_mul, n_rod))

y_rod = Expression('x[1] - ly/2', degree=2, ly=l_y)
C[:, 0] = assemble(v_wt * ds(2)).get_local()[boundary_dofs]
C[:, 1] = assemble(v_wt * y_rod * ds(2) + v_oms * ds(2)).get_local()[boundary_dofs]

G_rod = -C.T

Mint = la.block_diag(M_pl, M_rod)
Jint = la.block_diag(J_pl, J_rod)
Rint = la.block_diag(R_pl, R_rod)
Gint = np.concatenate((G_pl, G_rod), axis=0)
Bf_int = np.concatenate((Bf_pl, Bf_rod), axis=0)

Mint_sp = csc_matrix(Mint)
invMint = inv_sp(Mint_sp)
invMint = invMint.toarray()

S = Gint @ la.inv(Gint.T @ invMint @ Gint) @ Gint.T @ invMint

n_tot = n_pl + n_rod
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
    dydt = invMint @ (Jsys @ y + Fsys *bool_f)
    return dydt


init_con = Expression(('sin(2*pi*x[0])*sin(2*pi*(x[0]-lx))*sin(2*pi*x[1])*sin(2*pi*(x[1]-ly))', \
                      '0', '0', '0', '0', '0', '0', '0', '0'), degree=4, lx=l_x, ly=l_y)

e_pl0 = Function(V)
e_pl0.assign(interpolate(init_con, V))
y0 = np.zeros(n_tot,)
# y0[:n_pl] = e_pl0.vector().get_local()

t_ev = np.linspace(t0,t_fin, num = n_t)
dt = t_fin/(n_t - 1)

sol = integrate.solve_ivp(sys, t_span, y0, method='RK45', vectorized=False, t_eval = t_ev)

n_ev = len(t_ev)

e_pl = sol.y[:n_pl, :]
e_rod = sol.y[n_pl: n_tot, :]

e_pw = e_pl[dofs_Vpw, :]  # np.zeros((n_pw,n_t))

n_pw = V.sub(0).dim()
w0_pl = np.zeros((n_pw,))
w_pl = np.zeros(e_pw.shape)
w_pl[:, 0] = w0_pl
w_pl_old = w_pl[:, 0]
dt_vec = np.diff(t_ev)
for i in range(1, n_ev):
    w_pl[:, i] = w_pl_old + 0.5 * (e_pw[:, i - 1] + e_pw[:, i]) * dt_vec[i-1]
    w_pl_old = w_pl[:, i]

x_pl = dofV_x[dofs_Vpw, 0]
y_pl = dofV_x[dofs_Vpw, 1]


x_rod = np.ones((2,))*L

y_rod = np.array([l_y, 0])

v_rod = np.zeros((n_rod, n_ev))
w_rod = np.zeros((n_rod, n_ev))
w_rod_old = w_rod[:, 0]

for i in range(n_ev):

    v_rod[:, i] = x_rod * e_rod[0, i] + (y_rod - l_y / 2) * e_rod[1, i]
    if i >= 1:
        w_rod[:, i] = w_rod_old + 0.5 * (v_rod[:, i - 1] + v_rod[:, i]) * dt_vec[i-1]
        w_rod_old = w_rod[:, i]

w_pl_mm = w_pl * 1000
w_rod_mm = w_rod * 1000
minZ = w_pl_mm.min()
maxZ = w_pl_mm.max()

import drawNow2, matplotlib
# import tkinter as tk

matplotlib.interactive(True)

matplotlib.rcParams['text.usetex'] = True

plotter = drawNow2.plot3dClass(x_pl, y_pl, minZ=minZ, maxZ=maxZ, X2=x_rod, Y2=y_rod, \
                               xlabel='$x[m]$', ylabel='$y [m]$', \
                               zlabel='$w [mm]$', title='Vertical Displacement')

for i in range(n_ev):
    wpl_t = w_pl_mm[:, i]
    wrod_t = w_rod_mm[:, i]
    plotter.drawNow2(wpl_t, wrod_t, z2label='Rod $w[mm]$')

# tk.mainloop()

if matplotlib.is_interactive():
    plt.ioff()
# plt.close("all")

Hpl_vec = np.zeros((n_ev,))
Hrod_vec = np.zeros((n_ev,))

for i in range(n_ev):
    Hpl_vec[i] = 0.5 * (e_pl[:, i].T @ M_pl @ e_pl[:, i])
    Hrod_vec[i] = 0.5 * (e_rod[:, i].T @ M_rod @ e_rod[:, i])

fig = plt.figure(0)
plt.plot(t_ev, Hpl_vec, 'b-', label='Hamiltonian Plate (J)')
plt.plot(t_ev, Hrod_vec, 'r-', label='Hamiltonian Rod (J)')
plt.plot(t_ev, Hpl_vec + Hrod_vec, 'g-', label='Total Energy (J)')
plt.xlabel(r'{Time} (s)', fontsize=16)
plt.ylabel(r'{Hamiltonian} (J)', fontsize=16)
plt.title(r"Hamiltonian trend",
          fontsize=16)
plt.legend(loc='upper left')

plt.show()
#
# path_out = "/home/a.brugnoli/PycharmProjects/Mindlin_Phs_fenics/Temp_Simulation/Interconnection/"
# plt.savefig(path_out + "Pl_Rod_Hamiltonian.eps", format="eps")
#
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# n_fig = 4
# tol = 1e-6
# fntsize = 15
# for i in range(n_fig):
#    index = int(n_ev/n_fig*(i+1)-1)
#
#    fig = plt.figure(i+1)
#
#    ax = fig.add_subplot(111, projection='3d')
#
#    ax.set_xbound(min(x_pl) - tol, max(x_pl) + tol)
#    ax.set_xlabel('$x [m]$', fontsize=fntsize)
#    ax.set_ybound(min(y_pl) - tol, max(y_pl) + tol)
#    ax.set_ylabel('$y [m]$', fontsize=fntsize)
#    ax.set_zlabel('$w [mm]$', fontsize=fntsize)
#    ax.set_title('Vertical displacement', fontsize=fntsize)
#    ax.set_zlim3d(minZ - 0.01 * abs(minZ), maxZ + 0.01 * abs(maxZ))
#    ax.w_zaxis.set_major_locator(LinearLocator(10))
#    ax.w_zaxis.set_major_formatter(FormatStrFormatter('%3.2f' ))
#
#    tri_surf = ax.plot_trisurf(x_pl, y_pl, w_pl_mm[:,index], cmap=cm.jet, linewidth=0, antialiased=False)
#    tri_line = ax.plot(x_rod, y_rod, w_rod_mm[:,index], label='Rod $w[mm]$', color='black')
#
#    tri_surf._facecolors2d = tri_surf._facecolors3d
#    tri_surf._edgecolors2d = tri_surf._edgecolors3d
#    ax.legend(handles=[tri_line[0]])
#
#    plt.savefig(path_out + "Pl_Rod_t_" + str(index+1) + ".eps", format="eps")
#
# plt.show()


