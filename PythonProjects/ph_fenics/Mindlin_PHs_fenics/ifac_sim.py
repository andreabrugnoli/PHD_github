# Mindlin plate written with the port Hamiltonian approach

from fenics import *
import numpy as np

from scipy import linalg as la
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as inv_sp
from scipy import integrate
import matplotlib.pyplot as plt

from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from Mindlin_PHs_fenics.AnimateSurf import animate2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
plt.rc('text', usetex=True)

n_sim = 2

n = 10
deg = 2

E = 7e10
nu = 0.35
h = 0.1
rho = 2700  # kg/m^3
k =  0.8601 # 5./6. #
L = 1

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


g = Constant(10)
force = Expression("A*sin(2*pi/lx*x[0])", degree=4, lx = l_x, A = 10**5)
# force = Constant(1e6)
# f_p = v_pw * force * ds(2)
f_p1 = - v_pw * rho * h * g * dx
f_p2 = v_pw * force * ds(3) - v_pw * force * ds(4)

if n_sim == 1:
    f_p = f_p1
else:
    f_p = f_p2

F_int = assemble(f_p).get_local()

Mint = M_pl
Jint = J_pl
Rint = np.zeros((n_V, n_V))
Gint = G_pl
Mint_sp = csc_matrix(Mint)
invMint = inv_sp(Mint_sp)
invMint = invMint.toarray()

S = Gint @ la.inv(Gint.T @ invMint @ Gint) @ Gint.T @ invMint

I = np.eye(n_V)

P = I - S

Jsys = P @ Jint
Rsys = P @ Rint
Fsys = P @ F_int

t0 = 0.0
t_fin = 1e-3
t_span = [t0, t_fin]

def sys(t,y):
    if t < 0.25 * t_fin:
        bool_f = 1
    else: bool_f = 0
    dydt = invMint @ ( (Jsys - Rsys) @ y + Fsys *bool_f)
    return dydt


init_con = Expression(('sin(2*pi*x[0])*sin(2*pi*(x[0]-lx))*sin(2*pi*x[1])*sin(2*pi*(x[1]-ly))', \
                      '0', '0', '0', '0', '0', '0', '0', '0'), degree=4, lx=l_x, ly=l_y)

e_pl0 = Function(V)
e_pl0.assign(interpolate(init_con, V))
y0 = np.zeros(n_V,)
# y0[:n_pl] = e_pl0.vector().get_local()

t_ev = np.linspace(t0,t_fin, num=100)
sol = integrate.solve_ivp(sys, t_span, y0, method='RK45', vectorized=False, t_eval=t_ev)

n_ev = len(t_ev)

e_pl = sol.y

e_pw = e_pl[dofs_Vpw, :]  # np.zeros((n_pw,n_t))

w0 = np.zeros((len(dofs_Vpw),))
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
else:
    w_mm = w * 1000

minZ = w_mm.min()
maxZ = w_mm.max()


H_vec = np.zeros((n_ev,))
fntsize = 16

for i in range(n_ev):
    H_vec[i] = 0.5 * (e_pl[:, i].T @ M_pl @ e_pl[:, i])

fig = plt.figure()
plt.plot(t_ev, H_vec, 'b-', label='Hamiltonian (J)')
# plt.plot(t_ev, Ep, 'r-', label = 'Potential Energy (J)')
# plt.plot(t_ev, H_vec + Ep, 'g-', label = 'Total Energy (J)')
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

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
#
# anim.save('Video_n' + str(n_sim) + '.mp4', writer=writer)


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
