# Mindlin plate written with the port Hamiltonian approach

from fenics import *
import numpy as np
np.set_printoptions(threshold=np.inf)
import mshr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy import integrate
from scipy import linalg as la

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
# mesh = RectangleMesh(Point(0, 0), Point(L, L), n_x, n_y, "right/left")

domain = mshr.Rectangle(Point(0, 0), Point(l_x, l_y))
mesh = mshr.generate_mesh(domain, n, "cgal")

d = mesh.geometry().dim()

# plot(mesh)
# plt.show()

class ClampedSide(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],0.0) and on_boundary

class FreeSide(SubDomain):
    def inside(self, x, on_boundary):
        return (abs(x[0]-l_x)< DOLFIN_EPS or abs(x[1])<DOLFIN_EPS or abs(x[1]-l_y)<DOLFIN_EPS) and on_boundary


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

J = assemble(j).array()
M = assemble(m).array()


# Dirichlet Boundary Conditions and related constraints
bcs_p = []
bcs_q = []

if case_study == 'CCCC':
    bc_w = DirichletBC(V.sub(0), Constant(0.0), all_boundary)
    bc_th = DirichletBC(V.sub(1), (Constant(0.0), Constant(0.0)), all_boundary)

    bcs_p = [bc_w, bc_th]

if case_study == 'SSSS':
    bc_w = DirichletBC(V.sub(0), Constant(0.0), all_boundary)

    bc_ths_l = DirichletBC(V.sub(1).sub(1), Constant(0.0), left)
    bc_ths_r = DirichletBC(V.sub(1).sub(1), Constant(0.0), right)
    bc_ths_d = DirichletBC(V.sub(1).sub(0), Constant(0.0), lower)
    bc_ths_u = DirichletBC(V.sub(1).sub(0), Constant(0.0), upper)

    bc_Mnn_l = DirichletBC(V.sub(2).sub(0), Constant(0.0), left)
    bc_Mnn_r = DirichletBC(V.sub(2).sub(0), Constant(0.0), right)
    bc_Mnn_d = DirichletBC(V.sub(2).sub(2), Constant(0.0), lower)
    bc_Mnn_u = DirichletBC(V.sub(2).sub(2), Constant(0.0), upper)

    bcs_p = [bc_w, bc_ths_l, bc_ths_r, bc_ths_d, bc_ths_u]
    # bcs_q = [bc_Mnn_l, bc_Mnn_r, bc_Mnn_d, bc_Mnn_u]

if case_study =='SCSC':
    bc_w = DirichletBC(V.sub(0), Constant(0.0), all_boundary)

    bc_ths_l = DirichletBC(V.sub(1).sub(1), Constant(0.0), left)
    bc_ths_r = DirichletBC(V.sub(1).sub(1), Constant(0.0), right)
    bc_th_d = DirichletBC(V.sub(1), (Constant(0.0), Constant(0.0)), lower)
    bc_th_u = DirichletBC(V.sub(1), (Constant(0.0), Constant(0.0)), upper)

    bc_Mnn_l = DirichletBC(V.sub(2).sub(0), Constant(0.0), left)
    bc_Mnn_r = DirichletBC(V.sub(2).sub(0), Constant(0.0), right)

    bcs_p = [bc_w, bc_ths_l, bc_ths_r, bc_th_d, bc_th_u]
    # bcs_q = [bc_Mnn_l, bc_Mnn_r]

if case_study == 'CCCF':

    bc_w_l = DirichletBC(V.sub(0), Constant(0.0), left)
    bc_w_d = DirichletBC(V.sub(0), Constant(0.0), lower)
    bc_w_u = DirichletBC(V.sub(0), Constant(0.0), upper)


    bc_th_l = DirichletBC(V.sub(1), (Constant(0.0), Constant(0.0) ), left)
    bc_th_d = DirichletBC(V.sub(1), (Constant(0.0), Constant(0.0) ), lower)
    bc_th_u = DirichletBC(V.sub(1), (Constant(0.0), Constant(0.0) ), upper)

    bc_Mxx_r = DirichletBC(V.sub(2).sub(0), Constant(0.0), right)
    bc_Mxy_r = DirichletBC(V.sub(2).sub(1), Constant(0.0), right)
    bc_Qx_r = DirichletBC(V.sub(3).sub(0), Constant(0.0) , right)

    bcs_p = [bc_w_l, bc_w_d, bc_w_u, bc_th_l, bc_th_d, bc_th_u]
    # bcs_q = [bc_Mxx_r, bc_Mxy_r, bc_Qx_r]

if case_study == 'CFFF':
    bc_w = DirichletBC(V.sub(0), Constant(0.0), clamped_side)
    bc_th = DirichletBC(V.sub(1), (Constant(0.0), Constant(0.0)), clamped_side)

    bc_Mxx_r = DirichletBC(V.sub(2).sub(0), Constant(0.0), right)
    bc_Myy_b = DirichletBC(V.sub(2).sub(2), Constant(0.0), lower)
    bc_Myy_t = DirichletBC(V.sub(2).sub(2), Constant(0.0), upper)
    bc_Mxy = DirichletBC(V.sub(2).sub(1), Constant(0.0), free_side)

    bc_Qx_r = DirichletBC(V.sub(3).sub(0), Constant(0.0), right)
    bc_Qy_b = DirichletBC(V.sub(3).sub(1), Constant(0.0), lower)
    bc_Qy_t = DirichletBC(V.sub(3).sub(1), Constant(0.0), upper)

    bcs_p = [bc_w, bc_th]
    # bcs_q =[bc_Mxx_r, bc_Mxy, bc_Myy_b, bc_Myy_t, bc_Qx_r, bc_Qy_b, bc_Qy_t]

if ( (not bcs_p) and  (not bcs_q) ):
    raise ValueError('Empty boundary conditions')


boundary_dofs_p = []
for bc in bcs_p:
    for key in bc.get_boundary_values().keys():
        boundary_dofs_p.append(key)

boundary_dofs_p = sorted(list(set(boundary_dofs_p)))
n_ep = len(boundary_dofs_p)

G_ep = np.zeros((n_ep, n_V))
for (i, j) in enumerate(boundary_dofs_p):
    G_ep[i,j] = 1


boundary_dofs_q = []
for bc in bcs_q:
    for key in bc.get_boundary_values().keys():
        boundary_dofs_q.append(key)
boundary_dofs_q = sorted(list(set(boundary_dofs_q)))

n_eq = len(boundary_dofs_q)

G_eq = np.zeros((n_eq, n_V))
for (i, j) in enumerate(boundary_dofs_q):
    G_eq[i,j] = 1

# # Constraints related to the integration by parts
# n = FacetNormal(mesh)
# t = as_vector([-n[1], n[0] ])
#
# P_u = FiniteElement('P', triangle, deg)
#
# elem_u = MixedElement([P_u, P_u, P_u])
# Vu = FunctionSpace(mesh, elem_u)
#
# u = TrialFunction(Vu)
# v_u = TestFunction(Vu)

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
#
#
# Q_n, M_nn, M_ns = split(u)
# v_Qn, v_Mnn, v_Mns = split(v_u)
#
#
# v_dwdt = v_pw
# v_omn = dot(v_pth, n)
# v_oms = dot(v_pth, t)
#
# dwdt = e_pw
# omega_n = dot(e_pth, n)
# omega_s = dot(e_pth, t)
#
# b_u_Qn  =  v_pw * Q_n * ds
# b_u_Mnn = v_omn * M_nn * ds
# b_u_Mns = v_oms * M_ns * ds
#
# b_y_Qn  =  v_Qn * dwdt * ds
# b_y_Mnn = v_Mnn * omega_n * ds
# b_y_Mns = v_Mns * omega_s * ds
# #
# bc_u = DirichletBC(Vu, (Constant(0.0), Constant(0.0), Constant(0.0)) , clamped_side)
#
# boundary_dofs_u = []
# for key in bc_u.get_boundary_values().keys() :
#     boundary_dofs_u.append(key)
# boundary_dofs_u = sorted(list(set(boundary_dofs_u)))
# n_u = len(boundary_dofs_u)
#
# boundary_dofs_u = sorted(bc_u.get_boundary_values().keys() )
#
# n_u = len(boundary_dofs_u)
#
# B_in, B_out = PETScMatrix(), PETScMatrix()
#
# B_in = assemble(b_u_Qn + b_u_Mnn + b_u_Mns)
# B_out = assemble(b_y_Qn + b_y_Mnn + b_y_Mns)
#
# B_u = B_in.array()[:, boundary_dofs_u]
# B_y = B_out.array()[boundary_dofs_u, :]

# Force applied at the right boundary

b_w = v_pw * ds(3)
B_w = assemble(b_w).get_local()

# Splitting of matrices
n_e = n_eq + n_ep
n_lambda = n_e
n_tot = n_V + n_lambda
n_free = n_V - n_lambda

# Final Assemble of constraint matrix

G = np.vstack((G_ep, G_eq))
GT = G.transpose()

Q = la.inv(M)

f=0
invGQGT = np.linalg.inv(G @ Q @ GT)
def rhs_lambda(t,y):
    if t < 0.1*t_fin:
        lmbda = - invGQGT @ G @ Q @ (J @ (Q @ y) + B_w * f)
        dydt = J @ (Q @ y) + B_w * f + GT @ lmbda
    else:
        lmbda = - invGQGT @ G @ Q @ J @ (Q @ y)
        dydt = J @ (Q @ y) + GT @ lmbda
    return dydt


init_wt = Expression(('sin(2*pi*x[0])*sin(2*pi*(x[0]-lx))*sin(2*pi*x[1])*sin(2*pi*(x[1]-ly))', \
                      '0', '0', '0', '0', '0', '0', '0', '0'), degree=4, lx=l_x, ly=l_y)

e_in = interpolate(init_wt, V)
e_0 = Function(V)
e_0.assign(e_in)

y0 = M @ e_0.vector().get_local()
# y0 = np.zeros(n_V,)

t0 = 0.0
t_fin = 0.001
n_t = 100
t_span = [t0, t_fin]

t_ev = np.linspace(t0,t_fin, num = n_t)
dt = t_fin/(n_t - 1)

sol = integrate.solve_ivp(rhs_lambda, t_span, y0, method='RK45', vectorized=False, t_eval = t_ev)
e = np.zeros((n_V,n_t))
for i in range(0, n_t):
    e[:,i] = Q @ sol.y[:,i]

n_pw = V.sub(0).dim()

e_pw = e[dofs_Vpw,:]  # np.zeros((n_pw,n_t)) #

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
