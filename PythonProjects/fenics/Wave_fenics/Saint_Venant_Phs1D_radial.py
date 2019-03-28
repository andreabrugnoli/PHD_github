# Mindlin plate written with the port Hamiltonian approach

from fenics import *
import numpy as np
from math import pi

np.set_printoptions(threshold=np.inf)
import mshr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy import integrate
from scipy import linalg as la


n = 20
deg = 1

g = 10
rho = 1000 # kg/m^3


# Operators and functions

# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
# R = 1
# circle = mshr.Circle(Point(0, 0), R)
# mesh = mshr.generate_mesh(circle, n)

L = 1
mesh = IntervalMesh(n, 0, L)

d = mesh.geometry().dim()
# plot(mesh)
# plt.show()

class AllBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class Left(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 0.0) < DOLFIN_EPS and on_boundary

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - L) < DOLFIN_EPS and on_boundary


# Boundary conditions on displacement
all_boundary = AllBoundary()
# Boundary conditions on rotations
left = Left()
right = Right()
# lower = Lower()
# upper = Upper()

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)
ds = Measure('ds', subdomain_data= boundaries)
# lower.mark(boundaries, 3)
# upper.mark(boundaries, 4)

dx = Measure('dx')
ds = Measure('ds', subdomain_data=boundaries)

# Finite element defition

P_p = FiniteElement('DG', mesh.ufl_cell(), 0)
P_q = FiniteElement('CG', mesh.ufl_cell(), deg)


Vp = FunctionSpace(mesh, P_p)
Vq = FunctionSpace(mesh, P_q)

n_Vp = Vp.dim()
n_Vq = Vq.dim()
n_V = n_Vp + n_Vq

dofVp_x = Vp.tabulate_dof_coordinates().reshape((-1, d))
dofVq_x = Vq.tabulate_dof_coordinates().reshape((-1, d))

dofs_Vp = Vp.dofmap().dofs()
dofs_Vq = Vq.dofmap().dofs()

v_p = TestFunction(Vp)
v_q = TestFunction(Vq)

al_p = TrialFunction(Vp)
al_q = TrialFunction(Vq)

r = Expression("x[0]", degree = 4)

m_p = v_p * al_p * r * dx #inner(v_p, al_p) * dx
m_q = v_q * al_q * r * dx #inner(v_q, al_q) * dx

j_grad = -v_p * r * al_q.dx(0) * dx # -dot(v_p, grad(al_q) * dx
j_gradIP = v_q.dx(0) * al_p *r * dx # dot(grad(v_q), al_p) * dx

j_div = -v_q * al_p.dx(0) * dx # -v_q * div(al_p)) * dx
j_divIP = v_p.dx(0) * al_q *dx # div(v_p)* al_q * dx


B_l = assemble(-v_q * r * ds(1))
B_r = assemble(v_q * r * ds(2))

B_l = B_l.get_local()
B_lT = np.transpose(B_l)
B_r = B_r.get_local()
B_rT = np.transpose(B_r)


#j_alldiv_p = j_div
#j_alldiv_q = j_divIP

j_allgrad_p = j_gradIP
j_allgrad_q = j_grad

j_p = j_gradIP
j_q = j_grad

# Assemble the interconnection matrix and the mass matrix.
J_p, J_q, M_p, M_q = PETScMatrix(), PETScMatrix(), PETScMatrix(), PETScMatrix()

J_p = assemble(j_p)
J_q = assemble(j_q)

M_p = assemble(m_p).array()
M_q = assemble(m_q).array()

D_p = J_p.array()
D_q = J_q.array()

# r = Expression("sqrt(r*r+x[1]*x[1])", degree=2)
# theta = Expression("atan2(x[1],r)", degree=2)
#

# Final Assemble

al_p_ = Function(Vp)
al_q_ = Function(Vq)

Hd = 0.5*(1./rho* al_q_*dot(al_p_, al_p_) + rho*g*al_q_**2)*r*dx
e_p_ = derivative(Hd, al_p_)
e_q_ = derivative(Hd, al_q_)



M = la.block_diag(M_p, M_q)
J = np.zeros((n_V, n_V))
J[:n_Vp, n_Vp:n_V] = D_q
J[n_Vp:n_V, :n_Vp] = D_p

invM = la.inv(M)
Jtilde = invM @ J @ invM

# Stormer Verlet integrator
B = np.concatenate((np.zeros((n_Vp,)), B_r), axis=0).reshape((-1,1))
z = 0.00001
R = z * B @ B.T
Rtilde = invM @ R

def fun(t,y):
    al_p = y[:n_Vp]
    al_q = y[n_Vp:n_V]

    al_p_.vector()[:] = al_p
    al_q_.vector()[:] = al_q

    e_p = assemble(e_p_).get_local()
    e_q = assemble(e_q_).get_local()

    e = np.concatenate((e_p, e_q), axis = 0)

    dydt = Jtilde @ e - Rtilde @ e * (t>0.2)

    return dydt

H = 1; h = 0.001
init_p = Expression('0', degree=0)
init_q = Expression('H + h *sin(pi/L*x[0])', degree=4, H = H, h = h, L = L)

al_p_.assign(interpolate(init_p, Vp))
al_q_.assign(interpolate(init_q, Vq))

alp_0 = al_p_.vector().get_local()
alq_0 = al_q_.vector().get_local()

ep_0 = assemble(e_p_).get_local()
eq_0 = assemble(e_q_).get_local()

y0 = np.concatenate((alp_0, alq_0), axis = 0)

t0 = 0.0
t_fin = 1
n_t = 100
t_span = [t0, t_fin]

t_ev = np.linspace(t0,t_fin, num = n_t)

sol = integrate.solve_ivp(fun, t_span, y0, method='LSODA', vectorized=False, t_eval = t_ev, \
                          atol = 1e-5, rtol = 1e-5)

al_sol = sol.y
t_ev = sol.t
n_ev = len(t_ev)

alp_sol = al_sol[:n_Vp]
alq_sol = al_sol[n_Vp:n_V]

r_ev = dofVp_x[:,0]

perm = np.argsort(r_ev)
r_ev.sort()
alq_sol = alq_sol[perm, :]

n_r = len(r_ev)
n_th = 10
th_ev = np.linspace(0, 2*pi, num= n_th)

r_plot = np.tile(r_ev, n_th)
th_plot = np.repeat(th_ev, n_r)
alq_sol_plot = np.tile(alq_sol, (n_th, 1))
# minZ = min(alq_sol)
# maxZ = max(alq_sol)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['text.usetex'] = True


if matplotlib.is_interactive():
    plt.ioff()
plt.close('all')

H_vec = np.zeros((n_ev))
for i in range(n_ev):
    H_vec[i] = 0.5 *(al_sol[:, i] @ M @ al_sol[:, i])

fig0 = plt.figure(0)
plt.plot(t_ev, H_vec, 'g-', label = 'Total Energy (J)')
plt.xlabel(r'{Time} (s)',fontsize=16)
plt.legend(loc='upper left')

# from matplotlib import animation
#
# # First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure(1)
# ax = plt.axes(xlim=(0, L), ylim=(H -1.01*h, H +1.01*h))
# line, = ax.plot([], [], lw=2)
#
# # initialization function: plot the background of each frame
# def init():
#     line.set_data(x_ev, alq_0)
#     return line,
#
# # animation function.  This is called sequentially
# def animate(i):
#     line.set_data(x_ev, alq_sol[:,i])
#     return line,
#
# # call the animator.  blit=True means only re-draw the parts that have changed.
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=len(t_ev), interval=20, blit=False)
#
# # save the animation as an mp4.  This requires ffmpeg or mencoder to be
# # installed.  The extra_args ensure that the x264 codec is used, so that
# # the video can be embedded in html5.  You may need to adjust this for
# # your system: for more information, see
# # http://matplotlib.sourceforge.net/api/animation_api.html
# # anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
#
# plt.show()


# Make data.

fig1 = plt.figure(1)
ax = fig1.gca(projection='3d')
X_ev, T_ev = np.meshgrid(r_ev, t_ev)

fntsize = 20
tol = 1e-4
ax.set_xlim(min(r_ev) - tol, max(r_ev) + tol)
ax.set_xlabel('Space', fontsize=fntsize)

ax.set_ylim(min(t_ev) - tol* abs(min(t_ev)), max(t_ev) + tol* abs(max(t_ev)))
ax.set_ylabel('Time', fontsize=fntsize)

ax.set_zlabel('$w$', fontsize=fntsize)

# Customize the z axis.

# ax.set_zlim(minZ-1e-3*abs(minZ) , maxZ+1e-3*abs(maxZ))
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.06f'))

# Plot the surface.
Z = np.transpose(alq_sol)
h_plot = ax.plot_surface(X_ev, T_ev, Z, cmap=cm.jet, \
                       linewidth=0, antialiased=False, label = 'Wave $w$')

# Add a color bar which maps values to colors.


h_plot._facecolors2d=h_plot._facecolors3d
h_plot._edgecolors2d=h_plot._edgecolors3d

fig1.colorbar(h_plot, shrink=0.5, aspect=5)
plt.show()


from AnimateSurf import animate2D

x_plot = r_plot*np.cos(th_plot)
y_plot = r_plot*np.sin(th_plot)

anim = animate2D(x_plot, y_plot, alq_sol_plot, t_ev)

plt.show()