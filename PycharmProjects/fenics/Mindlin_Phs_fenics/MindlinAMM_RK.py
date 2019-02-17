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
deg = 3

E = (7e10)
nu = (0.35)
h = (0.1)
rho = (2700)  # kg/m^3
k =  0.8601 # 5./6. #
L = 1

case_study = input("Select the case under study: ")

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
phi_rot = (rho * h ** 3)/12.

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
ds = Measure('ds', subdomain_data= boundaries)

# Finite element defition

P_pw = FiniteElement('P', triangle, deg)
P_pth = VectorElement('P', triangle, deg)
P_qth = TensorElement('P', triangle, deg, symmetry=True)
P_qw = VectorElement('P', triangle, deg)

elem = MixedElement([P_pw, P_pth, P_qth, P_qw])

V = FunctionSpace(mesh, elem)
n_V = V.dim()

dofV_x = V.tabulate_dof_coordinates().reshape((-1, d))

dofs_Vpw = V.sub(0).dofmap().dofs()
dofs_Vpth = V.sub(1).dofmap().dofs()
dofs_Vqth = V.sub(2).dofmap().dofs()
dofs_Vqw = V.sub(3).dofmap().dofs()


dofVpw_x = dofV_x[dofs_Vpw]

v = TestFunction(V)
v_pw, v_pth, v_qth, v_qw = split(v)

e = TrialFunction(V)
e_pw, e_pth, e_qth, e_qw = split(e)


al_pw = rho * h * e_pw
al_pth = (rho * h ** 3) / 12. * e_pth
al_qth = bending_curv(e_qth)
al_qw = 1. / F * e_qw

m = inner(v_pw, al_pw) * dx + inner(v_pth, al_pth) * dx + \
    inner(v_qth, al_qth) * dx + inner(v_qw, al_qw) * dx


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

j_alldiv = j_div + j_divIP + j_divSym + j_divSymIP + j_Id + j_IdIP
j_allgrad = j_grad + j_gradIP + j_gradSym + j_gradSymIP + j_Id + j_IdIP

j = j_allgrad



# Assemble the interconnection matrix and the mass matrix.
J, M = PETScMatrix(), PETScMatrix()

J_pl = assemble(j).array()
M_pl = assemble(m).array()

bc_input = input('Select Boundary Condition:')

bc_1, bc_3, bc_2, bc_4 = bc_input

bc_dict = {1: bc_1, 2: bc_2, 3: bc_3, 4: bc_4}

n = FacetNormal(mesh)
s = as_vector([-n[1], n[0]])
P_qn = FiniteElement('CG', triangle, deg)
P_Mnn = FiniteElement('CG', triangle, deg)
P_Mns = FiniteElement('CG', triangle, deg)

element_u = MixedElement([P_qn, P_Mnn, P_Mns])

Vu = FunctionSpace(mesh, element_u)

q_n, M_nn, M_ns = TrialFunction(Vu)

v_omn = dot(v_pth, n)
v_oms = dot(v_pth, s)


g_vec = []
for key,val in bc_dict.items():
    if val == 'C':
        g_vec.append( v_pw * q_n * ds(key) + v_omn * M_nn * ds(key) + v_oms * M_ns * ds(key))
    elif val == 'S':
        g_vec.append(v_pw * q_n * ds(key) + v_oms * M_ns * ds(key))

g = sum(g_vec)

# Assemble the stiffness matrix and the mass matrix.

G = PETScMatrix

G = assemble(g)
G_pl = G.array()

bd_dofs_mul = np.where(G_pl.any(axis=0))[0]
G_pl = G_pl[:, bd_dofs_mul]

n_mul = len(bd_dofs_mul)

# Force applied at the right boundary

Mint = M_pl
Jint = J_pl
Rint = np.zeros((n_V, n_V))
Gint = G_pl
Bf_int = np.concatenate((Bf_pl, Bf_rod), axis=0)

Mint_sp = csc_matrix(Mint)
invMint = inv_sp(Mint_sp)
invMint = invMint.toarray()

S = Gint @ la.inv(Gint.T @ invMint @ Gint) @ Gint.T @ invMint

I = np.eye(n_V)

P = I - S

Jsys = P @ Jint
Rsys = P @ Rint
Fsys = P @ Bf_int

t0 = 0.0
fac = 100
t_base =
t_fin = 1e-2
t_span = [t0, t_fin]

def sys(t,y):
    if t< 0.25 * t_fin:
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

e_pl = sol.y[:n_pl, :]
e_rod = sol.y[n_pl: n_tot, :]

e_pw = e_pl[dofs_Vpw, :]

w0 = np.zeros((n_pw,))
w = np.zeros(e_pw.shape)
w[:,0] = w0
w_old = w[:,0]
for i in range(1,n_t):
    w[:,i] = w_old + 0.5*(e_pw[:,i-1] + e_pw[:,i])*dt
    w_old  = w[:,i]


x = dofVpw_x[:,0]
y = dofVpw_x[:,1]

minZ = w.min()
maxZ = w.max()

import drawNow, matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.interactive(True)

plotter = drawNow.plot3dClass(x, y, minZ= minZ, maxZ = maxZ,  \
                        xlabel = '$x [m]$', ylabel = '$y [m]$', \
                        zlabel = '$w[m]$', title = 'Vertical Displacement')

for i in range(n_t):
    w_t = w[:,i]
    plotter.drawNow(w_t)


H_vec = np.zeros((n_t,))

for i in range(n_t):
    H_vec[i] = 0.5  * np.transpose(e[:,i]) @ M @ e[:,i]

if matplotlib.is_interactive():
    plt.ioff()
fig = plt.figure()
plt.plot(t_ev, H_vec, 'b-')
plt.xlabel(r'{time} (s)',fontsize=16)
plt.ylabel(r'{Hamiltonian} (J)',fontsize=16)
plt.title(r"Hamiltonian trend",
          fontsize=16)
plt.show()
