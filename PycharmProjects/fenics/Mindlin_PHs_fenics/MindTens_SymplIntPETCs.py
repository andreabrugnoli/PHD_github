# Mindlin plate written with the port Hamiltonian approach

# # Retrieving values from numpy array
# # Also see the methods add() (uses global indices), add_local() and apply().
# u = Function(V)
# u.vector().set_local(x)

# # How to assign initial conditions to a mixed space
# e_u0 = Expression(('0.2', '0.3', '0.4'), degree=1)
# e_p0 = Expression('0.1', degree=1)
# u0 = interpolate(e_u0, V.sub(0).collapse())
# p0 = interpolate(e_p0, V.sub(1).collapse())
# w = Function(V)
# assign(w, [u0, p0])

from fenics import *
from math import floor
import numpy as np
# import scipy.linalg as la
import matplotlib.pyplot as plt

import mshr
from mpl_toolkits.mplot3d import Axes3D

parameters["allow_extrapolation"] = True
#
#
# ffc_options = {"optimize": True, \
#                "eliminate_zeros": True, \
#                "precompute_basis_const": True, \
#                "precompute_ip_const": True}


# Plate bending stiffness :math:`D=\dfrac{Eh^3}{12(1-\nu^2)}` and shear stiffness :math:`F = \kappa Gh`
# with a shear correction factor :math:`\kappa = 5/6` for a homogeneous plate
# of thickness :math:`h`::

E = (7e10)
nu = (0.35)
h = 0.1
rho = (2700)  # kg/m^3
k =  0.8601 # 5./6. #
f = -100

# E = Constant(7e10)
# nu = Constant(0.3)
# h = Constant(0.1)
# rho = Constant(2000)  # kg/m^3
# k = Constant(5. / 6.)
# f = Constant(-1.0)


D = E * h ** 3 / (1 - nu ** 2) / 12.
G = E / 2 / (1 + nu)
F = G * h * k

I_w = 1./(rho*h)
I_phi = 12. / (rho * h ** 3)
# Useful Matrices

D_b = as_tensor([
                [D, D * nu, 0],
                [D * nu, D, 0],
                [0, 0, D * (1 - nu) / 2]
                ])

fl_rot = 12. / (E * h ** 3)
C_b = as_tensor([
    [fl_rot, -nu*fl_rot, 0],
    [-nu*fl_rot, fl_rot, 0],
    [0, 0, fl_rot * (1 + nu)/2]
])


# Operators and functions
def gradSym(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    # return sym(nabla_grad(u))

def strain2voigt(eps):
    return as_vector([eps[0, 0], eps[1, 1], 2*eps[0, 1]])

def voigt2stress(S):
    return as_tensor([[S[0], S[2]], [S[2], S[1]]])

def bending_moment(u):
    return voigt2stress(dot(D_b,strain2voigt(u)))

def strain2voigt_Proj(M):
    return as_vector([M[0, 0], M[1, 1], M[0, 1]])

def bending_moment_Proj(u):
    return voigt2stress(dot(D_b,strain2voigt_Proj(u)))

def bending_curv(u):
  return voigt2stress(dot(C_b, strain2voigt(u)))

L= 1
n = 10
# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
l_x, l_y = L, L
n_x, n_y = n, n

# mesh = RectangleMesh(Point(0,0), Point(l_x, l_y), n_x, n_y, "right/left")
domain = mshr.Rectangle(Point(0, 0), Point(l_x, l_y))
mesh = mshr.generate_mesh(domain, n, "cgal")

class AllBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

all_boundary = AllBoundary()

# Finite element defition
deg = 1

P_pw =  FiniteElement('P', triangle, deg)
P_pth = VectorElement('P', triangle, deg)
P_qth = TensorElement('P', triangle, deg, shape=(2,2), symmetry=True)
P_qw =  VectorElement('P', triangle, deg)

V_pw = FunctionSpace(mesh,  P_pw)
V_pth = FunctionSpace(mesh, P_pth)
V_qth = FunctionSpace(mesh, P_qth)
V_qw = FunctionSpace(mesh,  P_qw)

N_pw = V_pw.dim()
N_pth = V_pth.dim()
N_qth = V_qth.dim()
N_qw = V_qw.dim()

dx = Measure('dx')

bc_w = DirichletBC(V_pw, Constant(0.0), all_boundary)
bc_th = DirichletBC(V_pth, ( Constant(0.0), Constant(0.0) ), all_boundary)


e_pw = TrialFunction(V_pw)
e_pth = TrialFunction(V_pth)
e_qth = TrialFunction(V_qth)
e_qw = TrialFunction(V_qw)

v_pw = TestFunction(V_pw)
v_pth = TestFunction(V_pth)
v_qth = TestFunction(V_qth)
v_qw = TestFunction(V_qw)


al_pw = rho*h*e_pw
al_pth = (rho*h**3)/12. * e_pth
al_qth = bending_curv(e_qth)
al_qw = 1./F *e_qw

e_pw0  = Constant(0.0) #Expression("sin(2*pi*x[0])*sin(2*pi*(x[0]-lx))*sin(2*pi*x[1])*sin(2*pi*(x[1]-ly))", degree =4, lx = l_x, ly = l_y) #Constant(0.0)
e_pth0 = Constant( (0.0, 0.0))  #Expression( ('0.0', '0.0'), degree =0)
e_qth0 = Constant( ( (0.0, 0.0),(0.0, 0.0) ) )   #Expression(( ('0.0','0.0'), ('0.0','0.0') ), degree=0)
e_qw0 = Constant( (0.0, 0.0))  #Expression(('0.0', '0.0'), degree=0)

e_pw_n = interpolate(e_pw0, V_pw )
e_pth_n = interpolate(e_pth0, V_pth )
e_qth_n = project(e_qth0, V_qth )
e_qw_n = interpolate(e_qw0, V_qw )

m_pw_form = dot(v_pw, al_pw)*dx
m_pth_form = dot(v_pth, al_pth)*dx
m_qth_form = inner(v_qth, al_qth)*dx
m_qw_form = dot(v_qw, al_qw)*dx

j_div = v_pw*div(e_qw)*dx
j_divIP = -div(v_qw)*e_pw*dx

j_divSym = dot(v_pth, div(e_qth))*dx
j_divSymIP = -dot(div(v_qth), e_pth)*dx

j_grad = dot(v_qw, grad(e_pw))*dx
j_gradIP = -dot(grad(v_pw), e_qw)*dx

j_gradSym = inner(v_qth, gradSym(e_pth))*dx
j_gradSymIP = -inner(gradSym(v_pth), e_qth)*dx

j_Id = dot(v_pth, e_qw)*dx
j_IdIP = -dot(v_qw, e_pth)*dx

# Define Matrices

A_pw = assemble(m_pw_form)
A_pth = assemble(m_pth_form)
A_qth = assemble(m_qth_form)
A_qw = assemble(m_qw_form)

D_divMat = assemble(j_div)
D_divSymMat = assemble(j_divSym)
D_IdMat = assemble(j_Id)

D_divIPMat = assemble(j_divIP)
D_divSymIPMat = assemble(j_divSymIP)
D_IdIPMat = assemble(j_IdIP)

f_pw = assemble(v_pw*f*rho*h*dx)

e_pw_ = Function(V_pw)    # current solution
e_pth_ = Function(V_pth)  # current solution
e_qth_ = Function(V_qth)  # current solution
e_qw_ = Function(V_qw)    # current solution


# Time-stepping
t = 0.0
T = 0.0001

dt =  10 ** (-6)
num_steps = floor(T/dt)

# pvdfile_velocity = File("Simu/Mindlin_Simu.pvd")

e_pw_ = Function(V_pw)
e_pw_.assign(e_pw_n)

# pvdfile_velocity << (e_pw_, t)

# Create progress bar
progress = Progress('Time-stepping')
set_log_level(PROGRESS)

H = 0.5 * (dot(rho*h*e_pw_n, e_pw_n) + dot(rho*h**3/12*e_pth_n, e_pth_n) + inner(bending_curv(e_qth_n), e_qth_n) + dot(1/F * e_qw_n, e_qw_n)) * dx
H0 = assemble(H)
print(H0)
Hd_vec = np.zeros((num_steps+1,1))
Hd_vec[0] = H0
t_vec = dt*np.arange(1, num_steps+2)

# Stormer-Verlet Sympletic
for ii in range(num_steps):

    # First variable n + 1/2

    b_Mpw = Function(V_pw).vector()
    A_pw.mult(e_pw_n.vector(), b_Mpw)

    b_Jdiv = Function(V_pw).vector()
    D_divMat.mult(e_qw_n.vector(), b_Jdiv)

    if t <0.1*T:
        b_pw = b_Mpw + 0.5*dt*(b_Jdiv + f_pw)
    else: b_pw = b_Mpw + 0.5 * dt * (b_Jdiv)

    bc_w.apply(A_pw, b_pw)
    solve(A_pw, e_pw_.vector(), b_pw)

    e_pw_n.assign(e_pw_)

    # Second Variable n + 1/2

    b_Mpth = Function(V_pth).vector()
    A_pth.mult(e_pth_n.vector(), b_Mpth)

    b_JdivSym = Function(V_pth).vector()
    D_divSymMat.mult(e_qth_n.vector(), b_JdivSym)

    b_JId = Function(V_pth).vector()
    D_IdMat.mult(e_qw_n.vector(), b_JId)

    b_pth = b_Mpth + 0.5*dt*(b_JdivSym + b_JId)
    bc_th.apply(A_pth, b_pth)
    solve(A_pth, e_pth_.vector(), b_pth)

    e_pth_n.assign(e_pth_)

    # Third Variable n+1

    b_Mqth = Function(V_qth).vector()
    A_qth.mult(e_qth_n.vector(), b_Mqth)

    b_JdivSymIP = Function(V_qth).vector()
    D_divSymIPMat.mult(e_pth_n.vector(), b_JdivSymIP)

    b_qth = b_Mqth + dt*b_JdivSymIP

    solve(A_qth, e_qth_.vector(), b_qth)

    e_qth_n.assign(e_qth_)

    # Forth Variable n+1

    b_Mqw = Function(V_qw).vector()
    A_qw.mult(e_qw_n.vector(), b_Mqw)

    b_JdivIP = Function(V_qw).vector()
    D_divIPMat.mult(e_pw_n.vector(), b_JdivIP)

    b_JIdIP = Function(V_qw).vector()
    D_IdIPMat.mult(e_pth_n.vector(), b_JIdIP)

    b_qw = b_Mqw + dt*(b_JdivIP + b_JIdIP)

    solve(A_qw, e_qw_.vector(), b_qw)
    e_qw_n.assign(e_qw_)

    # First variable n + 1

    b_Mpw = Function(V_pw).vector()
    A_pw.mult(e_pw_n.vector(), b_Mpw)

    b_Jdiv = Function(V_pw).vector()
    D_divMat.mult(e_qw_n.vector(), b_Jdiv)

    if t <0.1*T:
        b_pw = b_Mpw + 0.5*dt*(b_Jdiv + f_pw)
    else: b_pw = b_Mpw + 0.5 * dt * (b_Jdiv)

    bc_w.apply(A_pw, b_pw)
    solve(A_pw, e_pw_.vector(), b_pw)

    e_pw_n.assign(e_pw_)

    # Second Variable n + 1

    b_Mpth = Function(V_pth).vector()
    A_pth.mult(e_pth_n.vector(), b_Mpth)

    b_JdivSym = Function(V_pth).vector()
    D_divSymMat.mult(e_qth_n.vector(), b_JdivSym)

    b_JId = Function(V_pth).vector()
    D_IdMat.mult(e_qw_n.vector(), b_JId)

    b_pth = b_Mpth + 0.5*dt*(b_JdivSym + b_JId)
    bc_th.apply(A_pth, b_pth)
    solve(A_pth, e_pth_.vector(), b_pth)

    e_pth_n.assign(e_pth_)

    # Hamiltonian

    if ii>=0:
        H = 0.5 * (dot(rho * h * e_pw_n, e_pw_n) + dot(rho * h ** 3 / 12 * e_pth_n, e_pth_n) + inner(
            bending_curv(e_qth_n), e_qth_n) + dot(1 / F * e_qw_n, e_qw_n)) * dx

        Hd = assemble(H)
        # print(Hd)
        Hd_vec[ii+1] = Hd

    progress.update(t/T)
    t = t + dt

    # e_pw_.assign(e_pw_n)
    # plot(e_pw_)
    # plt.show()
    # pvdfile_velocity << (e_pw_, t)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(t_vec, Hd_vec, 'b-')
plt.xlabel(r'{time} (s)',fontsize=16)
plt.ylabel(r'{Hamiltonian} (J)',fontsize=16)
plt.title(r"Hamiltonian trend using Stormer-Verlet integration",
          fontsize=16)
plt.show()

