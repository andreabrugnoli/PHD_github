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

n = 3
deg = 2

E = (7e10)
nu = (0.35)
h = (0.1)
rho = (2700)  # kg/m^3
k = 0.8601  # 5./6. #
L = 1

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



# Boundary conditions on displacement
# Boundary conditions on rotations
left = Left()
right = Right()
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
# boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)



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

b_l = v_pw * q_n * ds(1) + v_omn * M_nn * ds(1) + v_oms * M_ns * ds(1)

b_r = v_pw * q_n * ds(2) + v_omn * M_nn * ds(2) + v_oms * M_ns * ds(2)

dofVu_x = Vu.tabulate_dof_coordinates().reshape((-1, d))

dofs_Vqn = Vu.sub(0).dofmap().dofs()
dofs_VMnn = Vu.sub(1).dofmap().dofs()
dofs_VMns = Vu.sub(2).dofmap().dofs()


J, M, B_l, B_r = PETScMatrix(), PETScMatrix(), PETScMatrix(), PETScMatrix()

J = assemble(j)
M = assemble(m)
B_l = assemble(b_l)
B_r = assemble(b_r)
M_pl = M.array()
J_pl = J.array()
R_pl = np.zeros((n_pl, n_pl))

Gl_pl1  = B_l.array()

bd_dofs_l = np.where(Gl_pl1.any(axis=0))[0]
Gl_pl1 = Gl_pl1[:,bd_dofs_l]

n_l = len(bd_dofs_l)

tol_bd = 10**-5

cond1 = np.logical_and( abs(dofVu_x[:, 0]- l_x) < tol_bd,
           np.logical_or( abs(dofVu_x[:, 1] - l_y/3) < tol_bd,
            abs(dofVu_x[:, 1] - 2*l_y/3) < tol_bd) )

cond2 = np.logical_and( abs(dofVu_x[:, 0]) < tol_bd,
           np.logical_or( abs(dofVu_x[:, 1] - l_y/3) < tol_bd,
            abs(dofVu_x[:, 1] - 2*l_y/3) < tol_bd) )

Gint_pl1 = B_r.array()[:, cond1]

Gint_pl2_NoArr = B_l.array()[:, cond2]
Gint_pl2 = np.zeros(Gint_pl2_NoArr.shape)

Gint_pl2[:, 0:3] = Gint_pl2_NoArr[:, 3:6]
Gint_pl2[:, 3:6] = Gint_pl2_NoArr[:, 0:3]

Gint = np.concatenate((Gint_pl1, -Gint_pl2), axis=0)

Gl = np.concatenate(( Gl_pl1, np.zeros((n_pl, n_l)) ), axis=0)

G = np.concatenate((Gl, Gint), axis=1)

A = Constant(10**3)
f_w1 = Expression("A*x[1]", degree=4, l_y=l_y, A = A)  #
b_f1 = v_pw * f_w1 * dx  # ds(3) - v_pw * f_w *  ds(4)
Bf_pl1 = assemble(b_f1).get_local()

f_w2 = Expression("A*(l_y - x[1])", degree=4, l_y=l_y, A = A)  #
b_f2 = v_pw * f_w2 * dx  # ds(3) - v_pw * f_w *  ds(4)
Bf_pl2 = assemble(b_f2).get_local()

Mint = la.block_diag(M_pl, M_pl)
Jint = la.block_diag(J_pl, J_pl)
Rint = la.block_diag(R_pl, R_pl)

Bf_int = np.concatenate((Bf_pl1, Bf_pl2), axis=0)

Mint_sp = csc_matrix(Mint)
invMint = inv_sp(Mint_sp)
invMint = invMint.toarray()

S = G @ la.inv(G.T @ invMint @ G) @ G.T @ invMint

n_tot = 2 * n_pl
I = np.eye(n_tot)

P = I - S

Jsys = P @ Jint
Rsys = P @ Rint
Fsys = P @ Bf_int

t0 = 0.0

t_base = 0.001
t_fin =  0.01
n_t = 200
t_span = [t0, t_fin]

def sys(t,y):
    if t< 0.2 * t_base:
        bool_f = 1
    else: bool_f = 0
    dydt = invMint @ ( (Jsys - Rsys) @ y + Fsys *bool_f)
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

e_pl1 = sol.y[:n_pl, :]
e_pl2 = sol.y[n_pl:n_tot, :]

e_pw1 = e_pl1[dofs_Vpw, :]  # np.zeros((n_pw,n_t))
e_pw2 = e_pl2[dofs_Vpw, :]

n_pw = V.sub(0).dim()

w0_pl1 = np.zeros((n_pw,))
w_pl1 = np.zeros(e_pw1.shape)
w_pl1[:, 0] = w0_pl1
w_pl1_old = w_pl1[:, 0]

w0_pl2 = np.zeros((n_pw,))
w_pl2 = np.zeros(e_pw2.shape)
w_pl2[:, 0] = w0_pl2
w_pl2_old = w_pl2[:, 0]

dt_vec = np.diff(t_ev)
for i in range(1, n_ev):
    w_pl1[:, i] = w_pl1_old + 0.5 * (e_pw1[:, i - 1] + e_pw1[:, i]) * dt_vec[i-1]
    w_pl1_old = w_pl1[:, i]

    w_pl2[:, i] = w_pl2_old + 0.5 * (e_pw2[:, i - 1] + e_pw2[:, i]) * dt_vec[i-1]
    w_pl2_old = w_pl2[:, i]

x_pl = dofV_x[dofs_Vpw, 0]
y_pl = dofV_x[dofs_Vpw, 1]


w_pl1_mm = w_pl1 * 1000
w_pl2_mm = w_pl2 * 1000
minZ = min( w_pl1_mm.min(), w_pl2_mm.min())
maxZ = max( w_pl1_mm.max(), w_pl2_mm.max())

import matplotlib
matplotlib.interactive(True)

matplotlib.rcParams['text.usetex'] = True


if matplotlib.is_interactive():
    plt.ioff()
# plt.close("all")

Hpl1_vec = np.zeros((n_ev,))
Hpl2_vec = np.zeros((n_ev,))

for i in range(n_ev):
    Hpl1_vec[i] = 0.5 * (e_pl1[:, i].T @ M_pl @ e_pl1[:, i])
    Hpl2_vec[i] = 0.5 * (e_pl2[:, i].T @ M_pl @ e_pl2[:, i])

fig = plt.figure()
plt.plot(t_ev, Hpl1_vec, 'b-', label='Hamiltonian Plate 1 (J)')
plt.plot(t_ev, Hpl2_vec, 'r-', label='Hamiltonian Plate 2 (J)')
plt.plot(t_ev, Hpl1_vec + Hpl2_vec, 'g-', label='Total Energy (J)')
plt.xlabel(r'{Time} (s)', fontsize=16)
plt.ylabel(r'{Hamiltonian} (J)', fontsize=16)
plt.title(r"Hamiltonian trend",
          fontsize=16)
plt.legend(loc='upper left')

from AnimateTwoPlates import animate2D
import matplotlib.animation as animation
sol = np.concatenate( (w_pl1_mm, w_pl2_mm), axis=0)
x = np.concatenate( (x_pl, l_x + x_pl), axis=0)
y = np.concatenate( (y_pl, y_pl), axis=0)
anim = animate2D(x_pl, y_pl, w_pl1_mm, w_pl2_mm, t_ev, xlabel = '$x[m]$', ylabel = '$y [m]$', \
                         zlabel = '$w [mm]$', title = 'Vertical Displacement')

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
#
# anim.save('IntPlates.mp4', writer=writer)


plt.show()