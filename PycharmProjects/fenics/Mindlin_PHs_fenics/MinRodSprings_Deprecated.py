# Mindlin plate written with the port Hamiltonian approach

from fenics import *
import numpy as np

np.set_printoptions(threshold=np.inf)
# import mshr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy import linalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

n = 10
deg =1

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

elem_p = MixedElement([P_pw, P_pth])
elem_q = MixedElement([P_qth, P_qw])

Vp = FunctionSpace(mesh, elem_p)
Vq = FunctionSpace(mesh, elem_q)

n_Vp = Vp.dim()
n_Vq = Vq.dim()
n_V = n_Vp + n_Vq

dofVp_x = Vp.tabulate_dof_coordinates().reshape((-1, d))
dofVq_x = Vq.tabulate_dof_coordinates().reshape((-1, d))

vertex_x = mesh.coordinates().reshape((-1, d))

dofs_Vpw = Vp.sub(0).dofmap().dofs()
dofs_Vpth = Vp.sub(1).dofmap().dofs()

dofs_Vqw = Vq.sub(1).dofmap().dofs()
dofs_Vqth = Vq.sub(0).dofmap().dofs()

v_p = TestFunction(Vp)
v_pw, v_pth, = split(v_p)

v_q = TestFunction(Vq)
v_qth, v_qw = split(v_q)

e_p = TrialFunction(Vp)
e_pw, e_pth = split(e_p)

e_q = TrialFunction(Vq)
e_qth, e_qw = split(e_q)

al_pw = rho * h * e_pw
al_pth = (rho * h ** 3) / 12. * e_pth
al_qth = bending_curv(e_qth)
al_qw = 1. / F * e_qw

m_p = inner(v_pw, al_pw) * dx + inner(v_pth, al_pth) * dx
m_q = inner(v_qth, al_qth) * dx + inner(v_qw, al_qw) * dx

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

j_allgrad_p = j_grad + j_gradSym + j_IdIP
j_allgrad_q = j_gradIP + j_gradSymIP + j_Id

j_p = j_allgrad_p
j_q = j_allgrad_q

# Assemble the interconnection matrix and the mass matrix.
J_p, J_q, M_p, M_q = PETScMatrix(), PETScMatrix(), PETScMatrix(), PETScMatrix()

J_p = assemble(j_p)
J_q = assemble(j_q)

M_p = assemble(m_p)
M_q = assemble(m_q)

Mp_pl = M_p.array()
Mq_pl = M_q.array()

Dp_pl = J_p.array()
Dq_pl = J_q.array()

# Dirichlet Boundary Conditions and related constraints
bc_w_l = DirichletBC(Vp.sub(0), Constant(0.0), left)
bc_th_l = DirichletBC(Vp.sub(1), (Constant(0.0), Constant(0.0)), left)
bcs_l = [bc_w_l, bc_th_l]

bc_w_r = DirichletBC(Vp.sub(0), Constant(0.0), right)
bc_th_r = DirichletBC(Vp.sub(1).sub(1), Constant(0.0), right)
bcs_r = [bc_w_r, bc_th_r]

boundary_dofs_l = []
for bc in bcs_l:
    for key in bc.get_boundary_values().keys():
        boundary_dofs_l.append(key)

boundary_dofs_l = sorted(boundary_dofs_l)
n_l = len(boundary_dofs_l)

G_l = np.zeros((n_Vp, n_l))
for (i, j) in enumerate(boundary_dofs_l):
    G_l[j, i] = 1

boundary_dofs_r = []
for bc in bcs_r:
    for key in bc.get_boundary_values().keys():
        boundary_dofs_r.append(key)
boundary_dofs_r = sorted(boundary_dofs_r)
n_r = len(boundary_dofs_r)

G_r = np.zeros((n_Vp, n_r))
for (i, j) in enumerate(boundary_dofs_r):
    G_r[j, i] = 1

# Force applied at the right boundary

f_w = Expression("1000000*( pow((x[1]-l_y/2), 2) + x[1]/10)", degree=4, l_y=l_y)  #
b_p = v_pw * f_w * dx  # ds(3) - v_pw * f_w *  ds(4)
B_p = assemble(b_p).get_local()
ind_f = np.where(dofVp_x[:, 0].any == 1)  # and (dofVp_x[:,1].any == 0 or dofVp_x[:,1].any == 1))

for index in ind_f:
    B_p[index] = 0

# Final Assemble

Mp_rod = np.diag([m_rod, Jxx_rod])
invMp_rod = la.inv(Mp_rod)
Mq_rod = np.diag([1./k_sp1, 1./k_sp2])
invMq_rod = np.diag([k_sp1, k_sp2])

Dp_rod = np.array([[1, l_y/2], [1, -l_y/2]])
# Dq_rod = np.array([[-1, -1], [l_y/2, -l_y/2]])
Dq_rod = - Dp_rod.T


r_v = r_sp1 + r_sp2
r_th = l_y**2/4*(r_sp1 - r_sp2)
r_vth = l_y/2*(r_sp1 - r_sp2)
R_rod = np.array([[r_v, r_vth], [r_vth, r_th]])

Mp_sp = csr_matrix(Mp_pl)
Mq_sp = csr_matrix(Mq_pl)

G_rT = G_r.T
G_lT = G_l.T

G = np.concatenate((G_l, G_r), axis=1)
GT = G.T

invMp_pl = la.inv(Mp_pl)
invMq_pl = la.inv(Mq_pl)

C = np.zeros((2, n_r))
# Dy = l_y/(n_y*deg)
for i in range(0, n_r):
    dof = boundary_dofs_r[i]
    x_dof, y_dof = dofVp_x[dof, :]
    if dof in dofs_Vpw:
        C[1, i] = y_dof - l_y / 2
        C[0, i] = 1
    else:
        C[1, i] = 1

CT = C.T

GMG = GT @ invMp_pl @ G

CMC_rr = CT @ invMp_rod @ C
CMC = np.zeros((n_l + n_r, n_l + n_r))
CMC[n_l: n_l + n_r, n_l:n_l + n_r] = CMC_rr
M_lambda = GT @ invMp_pl @ G + CMC
invM_lambda = la.inv(M_lambda)

invM_lmb_r = invM_lambda[n_l:, :]

Bpl_q = GT @ invMp_pl @ Dq_pl
Brod_q = np.vstack( (np.zeros((n_l, 2)), CT) ) @ invMp_rod @ Dq_rod
Brod_p = np.vstack( (np.zeros((n_l, 2)), CT) ) @ invMp_rod @ R_rod

t_0 = 0
dt = 1e-6
fac = 5
t_fac = 1
t_f = 0.001 * t_fac
n_ev = 100
t_ev = np.linspace(t_0, t_f, n_ev)

n_t = int(t_f / dt)

ep_pl_sol = np.zeros((n_Vp, n_ev))
eq_pl_sol = np.zeros((n_Vq, n_ev))
ep_rod_sol = np.zeros((2, n_ev))
eq_rod_sol = np.zeros((2, n_ev))

p0_pl = Expression(('sin(2*pi*x[0])*sin(2*pi*(x[0]-lx))*sin(2*pi*x[1])*sin(2*pi*(x[1]-ly))', '0', '0'), degree=4,
                   lx=l_x, ly=l_y)
q0_pl = Expression(('0', '0', '0', '0'), degree=4, lx=l_x, ly=l_y)

ep_pl0 = Function(Vp)
ep_pl0.assign(interpolate(p0_pl, Vp))
ep_pl_old = np.zeros((n_Vp))  # ep_pl_old =  np.zeros((n_Vp)) #
eq_pl_old = np.zeros((n_Vq))

ep_rod_old = np.zeros((2,))
eq_rod_old = np.zeros((2,))

ep_pl_sol[:, 0] = ep_pl_old
eq_pl_sol[:, 0] = eq_pl_old

ep_rod_sol[:, 0] = ep_rod_old
eq_rod_sol[:, 0] = eq_rod_old

Arod_p = Mp_rod + 0.5*dt*R_rod - 0.5*dt* C @ invM_lmb_r @ Brod_p

invArod_p = la.inv(Arod_p)

k = 1
f = 0
for i in range(1, n_t + 1):

    t = i * dt
    if t < t_f / (fac * t_fac):
        f = 1
    else:
        f = 0
    # Intergation for p (n+1/2)

    # Bpl_q = GT @ invMp_pl @ Dq_pl
    # Brod_q = np.vstack((np.zeros((n_l, 2)), CT)) @ invMp_rod @ Dq_rod
    # Brod_p = np.vstack((np.zeros((n_l, 2)), CT)) @ invMp_rod @ R_rod

    # lmbda = invM_lambda @ (-GT @ invMp_pl @ (Dq_pl @ eq_pl_old + B_p * f)\
    #                        +np.vstack( (np.zeros((n_l, 2)), CT) ) @ Mp_rod @ Dq_rod @ eq_rod_old)
    #
    # bp_rod = Mp_rod @ ep_rod_old + 0.5 * dt * (- C @ lmbda[n_l:] + Dq_rod @ eq_rod_old)
    #
    # ep_rod_new = invMp_rod @ bp_rod
    #
    # bp_pl = Mp_pl @ ep_pl_old + 0.5 * dt * (Dq_pl @ eq_pl_old + G @ lmbda + B_p * f)
    # bp_pl_sp = csr_matrix(bp_pl).reshape((n_Vp, 1))
    # ep_pl_new = spsolve(Mp_sp, bp_pl_sp)
    #
    # ep_pl_old = ep_pl_new
    # ep_rod_old = ep_rod_new

    bp_rod = Mp_rod @ ep_rod_old +\
             0.5 * dt * (- C @ invM_lmb_r @ (Brod_q @ eq_rod_old - Bpl_q @ eq_pl_old \
                          -GT @ invMp_pl @ B_p * f) + Dq_rod @ eq_rod_old)

    ep_rod_new = invArod_p @ bp_rod

    lmbda = invM_lambda @ ( - Bpl_q @ eq_pl_old \
                            + Brod_q @ eq_rod_old - Brod_p @ ep_rod_new -GT @ invMp_pl @ B_p * f)

    bp_pl = Mp_pl @ ep_pl_old + 0.5 * dt * (Dq_pl @ eq_pl_old + G @ lmbda + B_p * f)
    bp_pl_sp = csr_matrix(bp_pl).reshape((n_Vp, 1))
    ep_pl_new = spsolve(Mp_sp, bp_pl_sp)

    ep_pl_old = ep_pl_new
    ep_rod_old = ep_rod_new

    # Integration of q (n+1)
    bq_rod = Mq_rod @ eq_rod_old + dt * Dp_rod @ ep_rod_new

    eq_rod_new = invMq_rod @ bq_rod

    bq_pl = Mq_pl @ eq_pl_old + dt * Dp_pl @ ep_pl_new

    bq_pl_sp = csr_matrix(bq_pl).reshape((n_Vq, 1))
    eq_pl_new = spsolve(Mq_sp, bq_pl_sp)

    eq_pl_old = eq_pl_new
    eq_rod_old = eq_rod_new

    # Intergation for p (n+1)

    bp_rod = Mp_rod @ ep_rod_old + \
             0.5 * dt * (- C @ invM_lmb_r @ (Brod_q @ eq_rod_old - Bpl_q @ eq_pl_old \
                                             - GT @ invMp_pl @ B_p * f) + Dq_rod @ eq_rod_old)

    ep_rod_new = invArod_p @ bp_rod

    lmbda = invM_lambda @ (- Bpl_q @ eq_pl_old \
                           + Brod_q @ eq_rod_old - Brod_p @ ep_rod_new - GT @ invMp_pl @ B_p * f)

    bp_pl = Mp_pl @ ep_pl_old + 0.5 * dt * (Dq_pl @ eq_pl_old + G @ lmbda + B_p * f)
    bp_pl_sp = csr_matrix(bp_pl).reshape((n_Vp, 1))
    ep_pl_new = spsolve(Mp_sp, bp_pl_sp)

    ep_pl_old = ep_pl_new
    ep_rod_old = ep_rod_new

    # # Verify Constraints
    # assert (abs(G_rT @ ep_pl_new -CT @ ep_rod_new)<1e-10).all

    if t >= t_ev[k]:
        ep_pl_sol[:, k] = ep_pl_new
        eq_pl_sol[:, k] = eq_pl_new

        ep_rod_sol[:, k] = ep_rod_new
        eq_rod_sol[:, k] = eq_rod_new
        k = k + 1
        print('Solution number ' + str(k) + ' computed')

n_pw = Vp.sub(0).dim()

e_pw = ep_pl_sol[dofs_Vpw, :]  # np.zeros((n_pw,n_t))

w0_pl = np.zeros((n_pw,))
w_pl = np.zeros(e_pw.shape)
w_pl[:, 0] = w0_pl
w_pl_old = w_pl[:, 0]
Deltat = t_f / (n_ev - 1)
for i in range(1, n_ev):
    w_pl[:, i] = w_pl_old + 0.5 * (e_pw[:, i - 1] + e_pw[:, i]) * Deltat
    w_pl_old = w_pl[:, i]

x_pl = dofVp_x[dofs_Vpw, 0]
y_pl = dofVp_x[dofs_Vpw, 1]

dofsVpw_r = []
for dof in boundary_dofs_r:

    if dof in dofs_Vpw:
        dofsVpw_r.append(dof)

y_rod = dofVp_x[dofsVpw_r, 1]
n_rod = len(y_rod)
x_rod = np.ones((n_rod,))

v_rod = np.zeros((n_rod, n_ev))
w_rod = np.zeros((n_rod, n_ev))
w_rod_old = w_rod[:, 0]

for i in range(n_ev):

    #    k=0
    #    for j in range(n_r):
    #        if boundary_dofs_r[j] in dofsVpw_r:
    #
    #            v_rod[k,i] = CT[j,:] @ ep_rod_sol[:,i]
    #            k = k +1
    #    if k!= n_rod:
    #        ValueError("Okkio a k")

    v_rod[:, i] = x_rod * ep_rod_sol[0, i] + (y_rod - l_y / 2) * ep_rod_sol[1, i]
    if i >= 1:
        w_rod[:, i] = w_rod_old + 0.5 * (v_rod[:, i - 1] + v_rod[:, i]) * Deltat
        x_rod * eq_rod_sol[1, i] / k_sp2 + y_rod * eq_rod_sol[0, i] / k_sp1  #
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
    Hpl_vec[i] = 0.5 * (ep_pl_sol[:, i].T @ Mp_pl @ ep_pl_sol[:, i]\
                        + eq_pl_sol[:, i].T @ Mq_pl @ eq_pl_sol[:, i])
    Hrod_vec[i] = 0.5 * (ep_rod_sol[:, i].T @ Mp_rod @ ep_rod_sol[:, i]\
                         + eq_rod_sol[:, i].T @ Mq_rod @ eq_rod_sol[:, i])

t_ev = np.linspace(t_0, t_f, n_ev)
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

