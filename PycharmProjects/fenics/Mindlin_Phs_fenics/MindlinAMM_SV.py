# Mindlin plate written with the port Hamiltonian approach


from fenics import *
import numpy as np

from scipy import linalg as la
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as inv_sp

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from AnimateSurf import animate2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

matplotlib.rcParams['text.usetex'] = True

n_sim = 2

n = 10
deg = 2

E = 7e10
nu = 0.35
h = 0.1
rho = 2700  # kg/m^3
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

# Boundary markers
left = Left()
right = Right()
lower = Lower()
upper = Upper()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

left.mark(boundaries, 1)
right.mark(boundaries, 2)
lower.mark(boundaries, 3)
upper.mark(boundaries, 4)

dx = Measure('dx')
ds = Measure('ds', subdomain_data= boundaries)

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
v_pw, v_pth,= split(v_p)

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


## Constraints related to the integration by parts
# bc_input = input('Select Boundary Condition:')

if n_sim == 1:
    bc_input = 'CFFF'
else: bc_input = 'CFCF'
# bc_input = input('Select Boundary Condition:')

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
        g_vec.append(v_p * q_n * ds(key) + v_oms * M_ns * ds(key))

g = sum(g_vec)

# Assemble the stiffness matrix and the mass matrix.

G = PETScMatrix

G = assemble(g)
G_p = G.array()

bd_dofs_mul = np.where(G_p.any(axis=0))[0]
G_p = G_p[:, bd_dofs_mul]

n_mul = len(bd_dofs_mul)

# Splitting of matrices

# Force applied at the right boundary
g = Constant(10)
force = Expression("A*sin(2*pi/ly*x[1])", degree=4, ly = l_y, A = 10**6)
# force = Constant(1e6)
# f_p = v_pw * force * ds(2)
f_p1 = - v_pw * rho * h * g * dx
f_p2 = v_pw * force * ds(3) - v_pw * force * ds(4)

if n_sim == 1:
    f_p = f_p1
else: f_p = f_p2

F_p = assemble(f_p).get_local()

Mp_sp = csc_matrix(M_p)

invMp = inv_sp(Mp_sp)
invM_pl = invMp.toarray()

S_p = G_p @ la.inv(G_p.T @ invMp @ G_p) @ G_p.T @ invMp

n_tot = n_Vp + n_Vq
Id_p = np.eye(n_Vp)
P_p = Id_p - S_p

R_p = np.zeros((n_Vp, n_Vp))

import symplectic_integrators

solverSym = symplectic_integrators.StormerVerletGrad(M_p, M_q, D_p, D_q, R_p, P_p, F_p)

init_p = Expression(('A*sin(2*pi*x[0])*sin(2*pi*(x[0]-lx))*sin(2*pi*x[1])*sin(2*pi*(x[1]-ly))', '0', '0'), degree=4,
                    lx=l_x, ly=l_y, A = 10**(-3))
init_q = Expression(('0', '0', '0', '0'), degree=4, lx=l_x, ly=l_y)
e_p0_fun = Function(Vp)
e_p0_fun.assign(interpolate(init_p, Vp))
ep_0 = np.zeros((n_Vp)) # e_p0_fun.vector().get_local() #
eq_0 = np.zeros((n_Vq))

t_0 = 0
dt = 1e-6
t_f = 1e-2
n_ev = 100

sol = solverSym.compute_sol(ep_0, eq_0, t_f, t_0=t_0, dt=dt, n_ev=n_ev)

t_ev = sol.t_ev
ep_sol = sol.ep_sol
eq_sol = sol.eq_sol


n_pw = Vp.sub(0).dim()

e_pw = ep_sol[dofs_Vpw, :]  # np.zeros((n_pw,n_t))

w0 = np.zeros((n_pw,))
w = np.zeros(e_pw.shape)
w[:, 0] = w0
w_old = w[:, 0]

Vw = FunctionSpace(mesh, P_pw)
h_Ep = Function(Vw)
Ep = np.zeros((n_ev,))

dt_ev = np.diff(t_ev)
for i in range(1, n_ev):
    w[:, i] = w_old + 0.5 * (e_pw[:, i - 1] + e_pw[:, i]) * dt_ev[i-1]
    w_old = w[:, i]
    h_Ep.vector()[:] = np.ascontiguousarray(w[:, i], dtype='float')
    Ep[i] = assemble(rho * g * h * h_Ep * dx)

x = dofVpw_x[:, 0]
y = dofVpw_x[:, 1]

if n_sim == 1:
    w_mm = w * 1000000
else: w_mm = w * 1000
minZ = w_mm.min()
maxZ = w_mm.max()


H_vec = np.zeros((n_ev,))
fntsize = 16

for i in range(n_ev):
    H_vec[i] = 0.5 * (ep_sol[:, i].T @ M_p @ ep_sol[:, i] + eq_sol[:, i].T @ M_q @ eq_sol[:, i])

fig = plt.figure(0)
plt.plot(t_ev, H_vec, 'b-', label='Hamiltonian (J)')
plt.plot(t_ev, Ep, 'r-', label = 'Potential Energy (J)')
plt.plot(t_ev, H_vec + Ep, 'g-', label = 'Total Energy (J)')
plt.xlabel(r'{Time} (s)', fontsize=fntsize)
# plt.ylabel(r'{Hamiltonian} (J)',fontsize=fntsize)
plt.title(r"Hamiltonian trend",
          fontsize=fntsize)
plt.legend(loc='upper left')


path_out = "/home/a.brugnoli/Plots_Videos/Mindlin_plots/Temp_Simulation/Article_Min/"
plt.savefig(path_out + "Sim" +str(n_sim) + "Hamiltonian.eps", format="eps")


anim = animate2D(x, y, w_mm, t_ev, xlabel = '$x[m]$', ylabel = '$y [m]$', \
                         zlabel='$w [mm]$', title = 'Vertical Displacement')

plt.show()


save_figs = True
if save_figs:
    n_fig = 4
    tol = 1e-6
    for i in range(n_fig):
        index = int(n_ev/n_fig*(i+1)-1)
        fig = plt.figure(i+1)
        ax = fig.add_subplot(111, projection='3d')

        surf_opts = {'cmap': cm.jet, 'linewidth': 0, 'antialiased': False}  # , 'vmin': minZ, 'vmax': maxZ}

        ax.set_xbound(min(x) - tol, max(x) + tol)
        ax.set_xlabel('$x [m]$', fontsize=fntsize)
        ax.set_ybound(min(y) - tol, max(y) + tol)
        ax.set_ylabel('$y [m]$', fontsize=fntsize)
        ax.set_zlabel('$w [mm]$', fontsize=fntsize)
        ax.set_title('Vertical displacement', fontsize=fntsize)

        ax.set_zlim3d(minZ - 0.01 * abs(minZ), maxZ + 0.01 * abs(maxZ))
        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%3.2f' ))

        ax.plot_trisurf(x, y, w_mm[:,index], **surf_opts)
        plt.savefig(path_out + "Sim" + str(n_sim) + "t" + str(index + 1) + ".eps", format="eps")
        # plt.show()
