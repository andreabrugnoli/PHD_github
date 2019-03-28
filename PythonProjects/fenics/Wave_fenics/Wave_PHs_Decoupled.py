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

parameters["allow_extrapolation"] = True
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

boundarymesh = BoundaryMesh(mesh, 'exterior')


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

# sub_domains.set_all(4)

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

P_p = FiniteElement('P', triangle, 1)
P_q = VectorElement('P', triangle, 1)

V_p = FunctionSpace(mesh, P_p)
V_q = FunctionSpace(mesh, P_q)

N_p = V_p.dim()
N_q = V_q.dim()


bc_eq_l = DirichletBC(V_q.sub(0), v_l, left)
bc_eq_r = DirichletBC(V_q.sub(0), Constant(0.0), right)

bc_eq_u = DirichletBC(V_q.sub(1), v_u, upper)
bc_eq_d = DirichletBC(V_q.sub(1), Constant(0.0), lower)

# bcs = [bc_eq_l, bc_eq_u]

al_p = TrialFunction(V_p)
al_q = TrialFunction(V_q)

v_p = TestFunction(V_p)
v_q = TestFunction(V_q)

# Initial Conditions
p0  = Expression('0.0', degree =0)
q0 = Expression(('0.0', '0.0'), degree =0)

al_p0 = interpolate(p0, V_p)
al_q0 = interpolate(q0, V_q)

al_p_n = Function(V_p)
assign(al_p_n, al_p0)

al_q_n = Function(V_q)
assign(al_q_n, al_q0)

# # Forms and corresponding matrices
# left_mesh = SubMesh(boundarymesh, left)
#
# plot(left_mesh)
# plt.show()
#
# V_u = FunctionSpace(left_mesh, 'DG', 0)
#
# print(V_u.dim())
# psi_u = Function(V_u)
# psi_u.vector()[:] = np.ones(10, dtype=float)
#
# L =  v_p * interpolate(psi_u, V_q.sub(0).collapse()) * ds(0)
# print(assemble(L).get_local())

M_p_form = dot(v_p, al_p)*dx
M_q_form = dot(v_q, al_q)*dx

M_p = assemble(M_p_form).array()
M_q = assemble(M_q_form).array()

M = la.block_diag(M_p, M_q)

D_div = -v_p*div(al_q)*dx
D_divIP = div(v_q)*al_p*dx

D_grad = -dot(v_q, grad(al_p))*dx
D_gradIP = dot(grad(v_p),al_q)*dx

D_p = assemble(D_div).array()
D_pIP = assemble(D_divIP).array()

D_q = assemble(D_grad).array()
D_qIP = assemble(D_gradIP).array()

# J = D_div + D_divIP

Z_p = np.zeros((N_p, N_p))
Z_q = np.zeros((N_q, N_q))
J = np.vstack([np.hstack([Z_p   ,  D_qIP]),
               np.hstack([D_q,  Z_q])   ])



pvdfile_al_p = File("Simu/Waves_Simu.pvd")

# Create progress bar
progress = Progress('Time-stepping')
set_log_level(PROGRESS)

# Time-stepping
t = 0.0
T = 10
num_steps = 1000
dt = T / num_steps

eigs = la.eigvals(J, M)

eigs_im = np.sort(np.imag(eigs))
eigs_im = eigs_im[eigs_im> 1e-8]

# Define functions
al_p_ = Function(V_p)  # current solution
al_q_ = Function(V_q)  # current solution

H = 0.5 * (dot(al_q_n, al_q_n) + dot(al_p_n, al_p_n))* dx
H0 = assemble(H)

A_p = assemble(M_p_form)
A_q = assemble(M_q_form)

Hd_vec = np.zeros((num_steps,1))
t_vec = dt*np.arange(1, num_steps+1)

for ii in range(num_steps):

    v_l.time = t
    v_u.time = t

    e_l = interpolate(v_l, V_p)
    e_u = interpolate(v_u, V_p)

    L_p = dot(v_p, al_p_n) * dx + dt * dot(grad(v_p), al_q_n) * dx + dt * v_p * e_l * ds(0) + dt * v_p * e_u * ds(3)


    b_p = assemble(L_p)

    solve(A_p, al_p_.vector(), b_p)
    # solve(a == L, alpha_, bcs=bcs)

    # Update previous solution
    al_p_n.assign(al_p_)

    L_q = dot(v_q, al_q_n) * dx - dt * dot(v_q, grad(al_p_)) * dx
    b_q = assemble(L_q)
    # bc_eq_l.apply(A_q, b_q)
    # bc_eq_r.apply(A_q, b_q)
    # bc_eq_u.apply(A_q, b_q)
    # bc_eq_d.apply(A_q, b_q)

    solve(A_q, al_q_.vector(), b_q)

    al_q_n.assign(al_q_)
    # p = plot(al_p_, title='Velocity')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plot(al_p_, title='Velocity')
    # plt.draw()
    # plt.pause(0.01)

    # plt.hold(True)

    if ii>=0:
        H = 0.5 * (dot(al_q_n, al_q_n) + dot(al_p_n, al_p_n)) * dx
        Hd = assemble(H)
        print(Hd)
        Hd_vec[ii] = Hd

    # Save solution to file pvd
    pvdfile_al_p << (al_p_, t)
    # # Save nodal values to file
    # timeseries_al_w_.store(al_w_.vector(), t)

    progress.update(t/T)

    t = t + dt


plt.plot(t_vec, Hd_vec, 'r-')
plt.show()
