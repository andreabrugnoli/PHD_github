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

from __future__ import print_function
from fenics import *
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

# parameters["allow_extrapolation"] = True
#
#
# ffc_options = {"optimize": True, \
#                "eliminate_zeros": True, \
#                "precompute_basis_const": True, \
#                "precompute_ip_const": True}



# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
l_x, l_y = 1, 1
n_x, n_y = 10, 10

mesh = RectangleMesh(Point(0,0), Point(l_x, l_y), n_x, n_y, "right/left")



# Domain, Subdomains, Boundary, Suboundaries
tol = 1E-14


class Left(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0, tol)

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], l_x, tol)

class Lower(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0, tol)

class Upper(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], l_y, tol)




# Initialize mesh function for boundary domains

sub_domains = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)

sub_domains.set_all(4)

left = Left()
left.mark(sub_domains,  0)

right = Right()
right.mark(sub_domains, 1)

lower = Lower()
lower.mark(sub_domains, 2)

upper = Upper()
upper.mark(sub_domains, 3)


ds = Measure('ds', domain=mesh, subdomain_data= sub_domains)
n = FacetNormal(mesh)


v_u = Expression("time<= 1.0 ?   sin(pi*time) : 0", degree =2, time= 0.0)
v_l = Expression("time<= 1.0 ?  -sin(pi*time) : 0", degree =2, time= 0.0)




# Finite element defition
P_p = FiniteElement('P', triangle, 2)
P_q = VectorElement('P', triangle, 2)
V = FunctionSpace(mesh, MixedElement([P_p, P_q]))

bc_eq_l = DirichletBC(V.sub(1).sub(0), v_l, left)
bc_eq_u = DirichletBC(V.sub(1).sub(1), v_u, upper)

# bcs = [bc_eq_l, bc_eq_u]

alpha = TrialFunction(V)
al_p, al_q = split(alpha)

v = TestFunction(V)
v_p, v_q = split(v)

# Initial Conditions
p0  = Expression("0.0", degree =0)
q0 = Expression(('0.0', '0.0'), degree =0)

al_p0 = interpolate(p0, V.sub(0).collapse())
al_q0 = interpolate(q0, V.sub(1).collapse())

alpha_n = Function(V)
assign(alpha_n, [al_p0, al_q0])

# Forms and corresponding matrices
M = dot(v, alpha)*dx


j_div = -v_p*div(al_q)*dx
j_divIP = div(v_q)*al_p*dx

j_grad = -dot(v_q, grad(al_p))*dx
j_gradIP = dot(grad(v_p),al_q)*dx

# J = D_div + D_divIP
J = j_grad + j_gradIP


pvdfile_al_p = File("Simu/Waves_Simu.pvd")

# Create progress bar
progress = Progress('Time-stepping')
set_log_level(PROGRESS)

# Time-stepping
t = 0.0
T = 10
num_steps = 1000
dt = T / num_steps



a = M-dt*J
A = assemble(a)
# b = v_w*f*h**3*dx

# Define functions
alpha_ = Function(V)  # current solution

H = 0.5 * dot(alpha_n, alpha_n) * dx
H0 = assemble(H)
# Plot solution

# set  colormap
# v_l.time = 0.5
# v_u.time = 0.5
e_l = interpolate(v_l, V.sub(0).collapse())
e_u = interpolate(v_u, V.sub(0).collapse())

L = dot(v, alpha_n) * dx + dt * v_p * e_l * ds(0) + dt * v_p * e_u * ds(3)

# print(assemble(L).get_local())

Hd_vec = np.zeros((num_steps,1))
t_vec = dt*np.arange(1, num_steps+1)

for ii in range(num_steps):

    t = t + dt

    v_l.time = t
    v_u.time = t

    b = assemble(L)

    print(b.get_local().any() != 0)

    bc_eq_l.apply(A, b)
    bc_eq_u.apply(A, b)
    solve(A, alpha_.vector(), b)
    # solve(a == L, alpha_, bcs=bcs)

    # Update previous solution
    alpha_n.assign(alpha_)

    # Update current time

    al_p_, al_q_ = alpha_.split()

    # p = plot(al_p_, title='Velocity')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plot(al_p_, title='Velocity')
    # plt.draw()
    # plt.pause(0.01)

    # plt.hold(True)



    if ii>=0:
        H = 0.5 * dot(alpha_n, alpha_n) * dx
        Hd = assemble(H)
        print(Hd)
        Hd_vec[ii] = Hd

    # Save solution to file pvd
    pvdfile_al_p << (al_p_, t)
    # # Save nodal values to file
    # timeseries_al_w_.store(al_w_.vector(), t)

    progress.update(t/T)


plt.plot(t_vec, Hd_vec, 'r-')
plt.show()
