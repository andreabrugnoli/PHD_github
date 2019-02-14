# Mindlin plate written with the port Hamiltonian approach

from fenics import *
import numpy as np

np.set_printoptions(threshold=np.inf)
import mshr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy import integrate
from scipy import linalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

n = 10
deg = 1

E = (7e10)
nu = (0.35)
h = (0.1)
rho = (2700)  # kg/m^3
k = 0.8601  # 5./6. #
L = 1

case_study = input("Select the case under study: ")  # 'CFFF' #

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

# I_w = 1. / (rho * h)
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

class ClampedSide(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], 0.0) and on_boundary


class FreeSide(SubDomain):
    def inside(self, x, on_boundary):
        return (abs(x[0] - l_x) < DOLFIN_EPS or abs(x[1]) < DOLFIN_EPS or abs(x[1] - l_y) < DOLFIN_EPS) and on_boundary


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
free_side = FreeSide()
clamped_side = ClampedSide()
all_boundary = AllBoundary()
# Boundary conditions on rotations
left = Left()
right = Right()
lower = Lower()
upper = Upper()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
clamped_side.mark(boundaries, 1)
free_side.mark(boundaries, 2)
right.mark(boundaries, 3)
lower.mark(boundaries, 4)
upper.mark(boundaries, 5)

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

dofVpw_x = dofVp_x[dofs_Vpw]

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

j_div = v_pw * div(e_qw) * dx
j_divIP = -div(v_qw) * e_pw * dx

j_divSym = dot(v_pth, div(e_qth)) * dx
j_divSymIP = -dot(div(v_qth), e_pth) * dx

j_grad = dot(v_qw, grad(e_pw)) * dx
j_gradIP = -dot(grad(v_pw), e_qw) * dx

j_gradSym = inner(v_qth, gradSym(e_pth)) * dx
j_gradSymIP = -inner(gradSym(v_pth), e_qth) * dx

j_Id = dot(v_pth, e_qw) * dx
j_IdIP = -dot(v_qw, e_pth) * dx

# j_alldiv = j_div + j_divIP + j_divSym + j_divSymIP + j_Id + j_IdIP
# j_allgrad = j_grad + j_gradIP + j_gradSym + j_gradSymIP + j_Id + j_IdIP
#
# j = j_alldiv

j_alldiv_q = j_div + j_divSym + j_Id
j_alldiv_p = j_IdIP + j_divIP + j_divSymIP

j_allgrad_p = j_grad + j_gradSym + j_IdIP
j_allgrad_q = j_gradIP + j_gradSymIP + j_Id

j_p = j_allgrad_p
j_q = j_allgrad_q

# j_p = j_alldiv_p
# j_q = j_alldiv_q

# Assemble the interconnection matrix and the mass matrix.
J_p, J_q, M_p, M_q = PETScMatrix(), PETScMatrix(), PETScMatrix(), PETScMatrix()

J_p = assemble(j_p)
J_q = assemble(j_q)

M_p = assemble(m_p).array()
M_q = assemble(m_q).array()

D_p = J_p.array()
D_q = J_q.array()

# Dirichlet Boundary Conditions and related constraints

bcs_p = []
bcs_q = []

if case_study == 'CCCC':
    bc_w = DirichletBC(Vp.sub(0), Constant(0.0), all_boundary)
    bc_th = DirichletBC(Vp.sub(1), (Constant(0.0), Constant(0.0)), all_boundary)

    bcs_p = [bc_w, bc_th]

if case_study == 'SSSS':
    bc_w = DirichletBC(Vp.sub(0), Constant(0.0), all_boundary)

    bc_ths_l = DirichletBC(Vp.sub(1).sub(1), Constant(0.0), left)
    bc_ths_r = DirichletBC(Vp.sub(1).sub(1), Constant(0.0), right)
    bc_ths_d = DirichletBC(Vp.sub(1).sub(0), Constant(0.0), lower)
    bc_ths_u = DirichletBC(Vp.sub(1).sub(0), Constant(0.0), upper)

    # bc_Mnn_l = DirichletBC(Vq.sub(0).sub(0), Constant(0.0), left)
    # bc_Mnn_r = DirichletBC(Vq.sub(0).sub(0), Constant(0.0), right)
    # bc_Mnn_d = DirichletBC(Vq.sub(0).sub(2), Constant(0.0), lower)
    # bc_Mnn_u = DirichletBC(Vq.sub(0).sub(2), Constant(0.0), upper)

    bcs_p = [bc_w, bc_ths_l, bc_ths_r, bc_ths_d, bc_ths_u]
    # bcs_q = [bc_Mnn_l, bc_Mnn_r, bc_Mnn_d, bc_Mnn_u]

if case_study == 'SCSC':
    bc_w = DirichletBC(Vp.sub(0), Constant(0.0), all_boundary)

    bc_ths_l = DirichletBC(Vp.sub(1).sub(1), Constant(0.0), left)
    bc_ths_r = DirichletBC(Vp.sub(1).sub(1), Constant(0.0), right)
    bc_th_d = DirichletBC(Vp.sub(1), (Constant(0.0), Constant(0.0)), lower)
    bc_th_u = DirichletBC(Vp.sub(1), (Constant(0.0), Constant(0.0)), upper)

    # bc_Mnn_l = DirichletBC(Vq.sub(0).sub(0), Constant(0.0), left)
    # bc_Mnn_r = DirichletBC(Vq.sub(0).sub(0), Constant(0.0), right)

    bcs_p = [bc_w, bc_ths_l, bc_ths_r, bc_th_d, bc_th_u]
    # bcs_q = [bc_Mnn_l, bc_Mnn_r]

if case_study == 'CCCF':
    bc_w_l = DirichletBC(Vp.sub(0), Constant(0.0), left)
    bc_w_d = DirichletBC(Vp.sub(0), Constant(0.0), lower)
    bc_w_u = DirichletBC(Vp.sub(0), Constant(0.0), upper)

    bc_th_l = DirichletBC(Vp.sub(1), (Constant(0.0), Constant(0.0)), left)
    bc_th_d = DirichletBC(Vp.sub(1), (Constant(0.0), Constant(0.0)), lower)
    bc_th_u = DirichletBC(Vp.sub(1), (Constant(0.0), Constant(0.0)), upper)

    # bc_Mxx_r = DirichletBC(Vq.sub(0).sub(0), Constant(0.0), right)
    # bc_Mxy_r = DirichletBC(Vq.sub(0).sub(1), Constant(0.0), right)
    # bc_Qx_r  = DirichletBC(Vq.sub(1).sub(0), Constant(0.0) , right)

    bcs_p = [bc_w_l, bc_w_d, bc_w_u, bc_th_l, bc_th_d, bc_th_u]
    # bcs_q = [bc_Mxx_r, bc_Mxy_r, bc_Qx_r]

if case_study == 'CFFF':
    bc_w = DirichletBC(Vp.sub(0), Constant(0.0), clamped_side)
    bc_th = DirichletBC(Vp.sub(1), (Constant(0.0), Constant(0.0)), clamped_side)

    # bc_Mxx_r = DirichletBC(Vq.sub(0).sub(0), Constant(0.0), right)
    # bc_Myy_b = DirichletBC(Vq.sub(0).sub(2), Constant(0.0), lower)
    # bc_Myy_t = DirichletBC(Vq.sub(0).sub(2), Constant(0.0), upper)
    # bc_Mxy   = DirichletBC(Vq.sub(0).sub(1), Constant(0.0), free_side)

    bc_Qx_r = DirichletBC(Vq.sub(1).sub(0), Constant(0.0), right)
    bc_Qy_b = DirichletBC(Vq.sub(1).sub(1), Constant(0.0), lower)
    bc_Qy_t = DirichletBC(Vq.sub(1).sub(1), Constant(0.0), upper)

    bcs_p = [bc_w, bc_th]
    # bcs_q = [bc_Mxx_r, bc_Mxy, bc_Myy_b, bc_Myy_t, bc_Qx_r, bc_Qy_b, bc_Qy_t]

# if ( (not bcs_p) and  (not bcs_q) ):
#     raise ValueError('Empty boundary conditions')

if case_study == 'CFCF':
    bc_w_l = DirichletBC(Vp.sub(0), Constant(0.0), left)
    bc_th_l = DirichletBC(Vp.sub(1), (Constant(0.0), Constant(0.0)), left)

    bc_w_r = DirichletBC(Vp.sub(0), Constant(0.0), right)
    bc_th_r = DirichletBC(Vp.sub(1), (Constant(0.0), Constant(0.0)), right)

    bcs_p = [bc_w_l, bc_w_r, bc_th_l, bc_th_r]

# if ( (not bcs_p) and  (not bcs_q) ):
#     raise ValueError('Empty boundary conditions')


boundary_dofs_p = []
for bc in bcs_p:
    for key in bc.get_boundary_values().keys():
        boundary_dofs_p.append(key)

boundary_dofs_p = sorted(list(set(boundary_dofs_p)))
n_ep = len(boundary_dofs_p)
G_ep = np.zeros((n_ep, n_Vp))

for (i, j) in enumerate(boundary_dofs_p):
    G_ep[i, j] = 1

boundary_dofs_q = []
for bc in bcs_q:
    for key in bc.get_boundary_values().keys():
        boundary_dofs_q.append(key)
boundary_dofs_q = sorted(list(set(boundary_dofs_q)))
n_eq = len(boundary_dofs_q)

G_eq = np.zeros((n_eq, n_Vq))
for (i, j) in enumerate(boundary_dofs_q):
    G_eq[i, j] = 1

## Constraints related to the integration by parts
n = FacetNormal(mesh)
t = as_vector([-n[1], n[0]])

P_u = FiniteElement('P', triangle, deg)

elem_u = MixedElement([P_u, P_u, P_u])
Vu = FunctionSpace(mesh, elem_u)

u = TrialFunction(Vu)
v_u = TestFunction(Vu)

# dwdt, omega_n, omega_s = split(u)
# v_dwdt, v_omn, v_oms = split(v_u)
#
#
# v_Qn  = dot(v_qw, n)
# v_Mnn = inner(v_qth, outer(n, n))
# v_Mns = inner(v_qth, outer(t, n))
#
# Q_n  = dot(e_qw, n)
# M_nn = inner(e_qth, outer(n, n))
# M_ns = inner(e_qth, outer(t, n))
#
# b_u_dwdt = v_Qn * dwdt    * ds
# b_u_omn = v_Mnn * omega_n * ds
# b_u_oms = v_Mns * omega_s * ds
#
# b_y_dwdt = v_dwdt * Q_n  * ds
# b_y_omn  = v_omn  * M_nn * ds
# b_y_oms  = v_oms  * M_ns * ds
#
#
# B_in = assemble(b_u_dwdt + b_u_omn + b_u_oms)
# B_out = assemble(b_y_dwdt + b_y_omn + b_y_oms)
#
# bc_u = DirichletBC(Vu, (Constant(0.0), Constant(0.0), Constant(0.0)) , free_side)
#
# boundary_dofs_u = []
# for key in bc_u.get_boundary_values().keys() :
#     boundary_dofs_u.append(key)
# boundary_dofs_u = sorted(list(set(boundary_dofs_u)))
# n_u = len(boundary_dofs_u)
#
# B_u = B_in.array()[:, boundary_dofs_u]
# B_y = B_out.array()[boundary_dofs_u, :]

Q_n, M_nn, M_ns = split(u)
v_Qn, v_Mnn, v_Mns = split(v_u)

v_dwdt = v_pw
v_omn = dot(v_pth, n)
v_oms = dot(v_pth, t)

dwdt = e_pw
omega_n = dot(e_pth, n)
omega_s = dot(e_pth, t)

b_u_Qn = v_dwdt * Q_n * ds
b_u_Mnn = v_omn * M_nn * ds
b_u_Mns = v_oms * M_ns * ds

b_y_Qn = v_Qn * dwdt * ds
b_y_Mnn = v_Mnn * omega_n * ds
b_y_Mns = v_Mns * omega_s * ds
#
bc_u = DirichletBC(Vu, (Constant(0.0), Constant(0.0), Constant(0.0)), clamped_side)

boundary_dofs_u = []
for key in bc_u.get_boundary_values().keys():
    boundary_dofs_u.append(key)
boundary_dofs_u = sorted(list(set(boundary_dofs_u)))
n_u = len(boundary_dofs_u)

boundary_dofs_u = sorted(bc_u.get_boundary_values().keys())

n_u = len(boundary_dofs_u)

B_in, B_out = PETScMatrix(), PETScMatrix()

B_in = assemble(b_u_Qn + b_u_Mnn + b_u_Mns)
B_out = assemble(b_y_Qn + b_y_Mnn + b_y_Mns)

B_u = B_in.array()[:, boundary_dofs_u]
B_y = B_out.array()[boundary_dofs_u, :]

# Splitting of matrices

# Force applied at the right boundary
g = Constant(10)

f_w = Expression("100000*sin(2*pi*x[0])", degree=4)
# f_qn = Expression("1000000*sin(2*pi*x[1])", degree= 4) # Constant(1e5) #
b_p = v_pw * f_w * ds(4) - v_pw * f_w * ds(5)
#                                # v_pw * f_qn * ds(3) #
#                                # - v_pw * rho * h * g * dx #


# f_w =Expression("1000000*( pow((x[1]-l_y/2), 2) + x[1]/10)", degree= 4, l_y = l_y) #
# b_p = v_pw * f_w * dx #ds(3) - v_pw * f_w *  ds(4)

B_p = assemble(b_p).get_local()
ind_f = np.where(dofVp_x[:, 0].any == 1)  # and (dofVp_x[:,1].any == 0 or dofVp_x[:,1].any == 1))

for index in ind_f:
    B_p[index] = 0
B_p = assemble(b_p).get_local()

# Final Assemble
Mp_sp = csr_matrix(M_p)
Mq_sp = csr_matrix(M_q)

G_epT = np.transpose(G_ep)
G_eq = np.zeros((0, 0))
G_eqT = np.transpose(G_eq)

invMp = la.inv(M_p)
invMq = la.inv(M_q)

if G_ep.size != 0:
    invGMGT_p = la.inv(G_ep @ invMp @ G_epT)

if G_eq.size != 0:
    invGMGT_q = la.inv(G_eq @ invMq @ G_eqT)

t_0 = 0
dt = 1e-6
fac = 4
t_f = 0.001
n_ev = 100
t_ev = np.linspace(t_0, t_f, n_ev)

n_t = int(t_f / dt)

ep_sol = np.zeros((n_Vp, n_ev))
eq_sol = np.zeros((n_Vq, n_ev))

init_p = Expression(('sin(2*pi*x[0])*sin(2*pi*(x[0]-lx))*sin(2*pi*x[1])*sin(2*pi*(x[1]-ly))', '0', '0'), degree=4,
                    lx=l_x, ly=l_y)
init_q = Expression(('0', '0', '0', '0'), degree=4, lx=l_x, ly=l_y)

e_pw_in = interpolate(init_p, Vp)
e_pw_0 = Function(Vp)
e_pw_0.assign(e_pw_in)
ep_old = np.zeros((n_Vp))  # e_pw_0.vector().get_local() #
eq_old = np.zeros((n_Vq))

ep_sol[:, 0] = ep_old
eq_sol[:, 0] = eq_old

k = 1
f = 1
for i in range(1, n_t + 1):

    t = i * dt
    if t < t_f / fac:
        f = 1
    else:
        f = 0
    # Intergation for p (n+1/2)

    w_De_q = D_q @ eq_old

    if G_ep.size == 0:
        bp = M_p @ ep_old + 0.5 * dt * (w_De_q + B_p * f)
    else:
        lmbda_p = - invGMGT_p @ G_ep @ invMp @ (w_De_q + B_p * f)
        bp = M_p @ ep_old + 0.5 * dt * (w_De_q + G_epT @ lmbda_p + B_p * f)

    bp_sp = csr_matrix(bp).reshape((n_Vp, 1))
    ep_new = spsolve(Mp_sp, bp_sp)

    ep_old = ep_new
    # Integration of q (n+1)

    w_De_p = D_p @ ep_new

    if G_eq.size == 0:
        bq = M_q @ eq_old + dt * w_De_p
    else:
        lmbda_q = - invGMGT_q @ G_eq @ invMq @ w_De_p
        bq = M_q @ eq_old + dt * (w_De_p + G_eqT @ lmbda_q)

    bq_sp = csr_matrix(bq).reshape((n_Vq, 1))
    eq_new = spsolve(Mq_sp, bq_sp)

    eq_old = eq_new

    # Intergation for p (n+1)

    w_De_q = D_q @ eq_old

    if G_ep.size == 0:
        bp = M_p @ ep_old + 0.5 * dt * (w_De_q + B_p * f)
    else:
        lmbda_p = - invGMGT_p @ G_ep @ invMp @ (w_De_q + B_p * f)
        bp = M_p @ ep_old + 0.5 * dt * (w_De_q + G_epT @ lmbda_p + B_p * f)

    bp_sp = csr_matrix(bp).reshape((n_Vp, 1))
    ep_new = spsolve(Mp_sp, bp_sp)

    ep_old = ep_new

    if t >= t_ev[k]:
        ep_sol[:, k] = ep_new
        eq_sol[:, k] = eq_new
        k = k + 1
        print('Solution number ' + str(k) + ' computed')

n_pw = Vp.sub(0).dim()

e_pw = ep_sol[dofs_Vpw, :]  # np.zeros((n_pw,n_t))

w0 = np.zeros((n_pw,))
w = np.zeros(e_pw.shape)
w[:, 0] = w0
w_old = w[:, 0]

Vw = FunctionSpace(mesh, P_pw)
h_Ep = Function(Vw)
Ep = np.zeros((n_ev,))

Deltat = t_f / (n_ev - 1)
for i in range(1, n_ev):
    w[:, i] = w_old + 0.5 * (e_pw[:, i - 1] + e_pw[:, i]) * Deltat
    w_old = w[:, i]
    h_Ep.vector()[:] = np.ascontiguousarray(w[:, i], dtype='float')
    Ep[i] = assemble(rho * g * h * h_Ep * dx)

x = dofVpw_x[:, 0]
y = dofVpw_x[:, 1]

w_mm = w * 1000
minZ = w_mm.min()
maxZ = w_mm.max()

from AnimateSurf import animate2D
anim = animate2D(x, y, w_mm, t_ev, xlabel = '$x[m]$', ylabel = '$y [m]$', \
                         zlabel = '$w [mm]$', title = 'Vertical Displacement')

plt.show()