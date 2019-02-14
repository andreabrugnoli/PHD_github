# Mindlin plate written with the port Hamiltonian approach

from fenics import *
import numpy as np
np.set_printoptions(threshold=np.inf)
# import mshr
# from scipy import integrate
from scipy import linalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


n = 100
deg = 1

rho = 1
T = 10000
k = 100
m = 2
r = 0

L = 1
mesh = IntervalMesh(n, 0, L)

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
        return abs(x[0] - L) < DOLFIN_EPS and on_boundary

# Boundary conditions on rotations
left = Left()
right = Right()


boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
left.mark(boundaries, 1)
right.mark(boundaries, 2)


dx = Measure('dx')
ds = Measure('ds', subdomain_data= boundaries)

# Finite element defition

Pp = FiniteElement('CG', mesh.ufl_cell(), deg)
Pq = FiniteElement('CG', mesh.ufl_cell(), deg)

Vp = FunctionSpace(mesh, Pp)
Vq = FunctionSpace(mesh, Pq)

n_Vp = Vp.dim()
n_Vq = Vq.dim()
n_V = n_Vp + n_Vq

dofVp_x = Vp.tabulate_dof_coordinates().reshape((-1, d))
dofVq_x = Vq.tabulate_dof_coordinates().reshape((-1, d))

vertex_x = mesh.coordinates().reshape((-1, d))

dofs_Vp = Vp.dofmap().dofs()
dofs_Vq = Vq.dofmap().dofs()

dofVp_x = dofVp_x[dofs_Vp]

v_p = TestFunction(Vp)
v_q = TestFunction(Vq)

e_p = TrialFunction(Vp)
e_q = TrialFunction(Vq)

al_p = rho * e_p
al_q = 1./T * e_q


m_p = inner(v_p, al_p) * dx
m_q = inner(v_q, al_q) * dx

j_div = v_p * e_q.dx(0) * dx # v_p * div(e_q) * dx
j_divIP = -v_q.dx(0) * e_p *dx # -div(v_q) * e_p * dx

j_grad = v_q * e_p.dx(0) * dx # dot(v_q, grad(e_p)) * dx
j_gradIP = -v_p.dx(0) * e_q *dx # dot(-grad(v_p), e_q) * dx

j_alldiv_q = j_div
j_alldiv_p = j_divIP

j_allgrad_p = j_grad
j_allgrad_q = j_gradIP

# j_p = j_allgrad_p
# j_q = j_allgrad_q

# Dirichlet BC
j_p = j_allgrad_p
j_q = j_allgrad_q

# Assemble the interconnection matrix and the mass matrix.
J_p, J_q, M_p, M_q = PETScMatrix(), PETScMatrix(), PETScMatrix(), PETScMatrix()


J_p = assemble(j_p)
J_q = assemble(j_q)

M_p = assemble(m_p).array()
M_q = assemble(m_q).array()

D_p = J_p.array()
D_q = J_q.array()


B_l, B_r = PETScVector(), PETScVector()

B_l = assemble(-v_p * ds(1))
B_r = assemble(v_p * ds(2))

B_l = B_l.get_local()
B_lT = np.transpose(B_l)
B_r = B_r.get_local()
B_rT = np.transpose(B_r)


# Final Assemble
Mp_sp = csr_matrix(M_p)
Mq_sp = csr_matrix(M_q)

invMp = la.inv(M_p)
invMq = la.inv(M_q)


B_lambda = np.array([B_l, B_r]).transpose()
B_lambdaT = np.transpose(B_lambda)
A_lambda = B_lambdaT @ invMp @ B_lambda + 1./m * np.array([[0, 0], [0, 1]], dtype = float)

Bst_q = - B_lambdaT @ invMp @ D_q
Bos_q = - 1./m * np.array([0, 1], dtype = float)
Bos_p =   r*Bos_q

invA_lam = la.inv(A_lambda)

invA_lam_r = invA_lam[1, :]



t_0 = 0
dt = 1e-5
t_f = 0.1
n_ev = 100
t_ev = np.linspace(t_0, t_f, n_ev)

n_t = int(np.floor(t_f/dt))

ep_st = np.zeros((n_Vp, n_ev))
eq_st = np.zeros((n_Vq, n_ev))

v0_st = Expression('sin((2*pi)/L*x[0])', degree=3, L=L) # Expression('sin((pi/2)/L*x[0])', degree=3, L=L)
T0_st = Expression('0', degree=0)

x_ev = dofVp_x[:,0]

ind = np.where( x_ev == 1)
# y_ev = dofVp_x[:,1]


ep_in = interpolate(v0_st, Vp)
ep0_st = Function(Vp)
ep0_st.assign(ep_in)
ep_st_old = np.zeros((n_Vp)) # ep0_st.vector().get_local() #
eq_st_old = np.zeros((n_Vq))

ep_os_old = 10
eq_os_old = 0

ep_st_old[ind] = ep_os_old

ep_st[:,0] = ep_st_old
eq_st[:,0] = eq_st_old

ep_os = np.zeros((n_ev,))
eq_os = np.zeros((n_ev,))

ep_os[0] = ep_os_old
eq_os[0] = eq_os_old


jj = 1


a_os_p = m + 0.5*dt*r + 0.5*dt*invA_lam_r @ Bos_p
# Stormer Verlet integrator

for i in range(1,n_t + 1):

    t = i * dt

    # Intergation for p (n+1/2)

    b_os_p = m*ep_os_old + 0.5*dt*(-invA_lam_r @ (Bos_q*eq_os_old + Bst_q @ eq_st_old) - eq_os_old)
    ep_os_new = b_os_p / a_os_p

    lmbda = invA_lam @ (Bst_q @ eq_st_old + Bos_p * ep_os_new + Bos_q * eq_os_old)

    bp_st = M_p @ ep_st_old + 0.5 *dt * (D_q @ eq_st_old + B_lambda @ lmbda)
    bp_sp = csr_matrix(bp_st).reshape((n_Vp,1))

    ep_st_new = spsolve(Mp_sp, bp_sp)

    ep_st_old = ep_st_new
    ep_os_old = ep_os_new

    # Integration of q (n+1)

    bq_st = M_q @ eq_st_old + dt * D_p @ ep_st_new

    bq_sp =csr_matrix(bq_st).reshape((n_Vq,1))

    eq_st_new = spsolve(Mq_sp, bq_sp)

    eq_os_new = eq_os_old + k* dt* ep_os_new

    eq_st_old = eq_st_new
    eq_os_old = eq_os_new

    # Intergation for p (n+1)

    b_os_p = m * ep_os_old + 0.5 * dt * (-invA_lam_r @ (Bos_q * eq_os_old + Bst_q @ eq_st_old) - eq_os_old)
    ep_os_new = b_os_p / a_os_p

    lmbda = invA_lam @ (Bst_q @ eq_st_old + Bos_p * ep_os_new + Bos_q * eq_os_old)

    bp_st = M_p @ ep_st_old + 0.5 * dt * (D_q @ eq_st_old + B_lambda @ lmbda)
    bp_sp = csr_matrix(bp_st).reshape((n_Vp, 1))

    ep_st_new = spsolve(Mp_sp, bp_sp)

    ep_st_old = ep_st_new
    ep_os_old = ep_os_new

    if  t>=t_ev[jj]:
        ep_st[:, jj] = ep_st_new
        eq_st[:,jj] = eq_st_new
        ep_os[jj] = ep_os_new
        ep_os[jj] = ep_os_new

        jj = jj + 1
        print('Solution number ' +str(jj) + ' computed')

n_p = Vp.dim()

w_st0 = np.zeros((n_p,))
w_st = np.zeros(ep_st.shape)
w_st[:,0] = w_st0
w_st_old = w_st[:,0]

w_os0 = 0
w_os = np.zeros((n_ev,))
w_os[0] = w_os0
w_os_old = w_os[0]

for i in range(1,n_ev):
    w_st[:,i] = w_st_old + 0.5*(ep_st[:,i-1] + ep_st[:,i])*dt
    w_st_old  = w_st[:,i]

    w_os[i] = w_os_old + 0.5*(ep_os[i-1] + ep_os[i])*dt
    w_os_old = w_os[i]

minZ = min((w_st.min(), w_os.min()))
maxZ = max((w_st.max(), w_os.max()))

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True



if matplotlib.is_interactive():
    plt.ioff()
plt.close('all')

path_out = "/home/a.brugnoli/PycharmProjects/Mindlin_Phs_fenics/Temp_Simulation/"

Hst_vec = np.zeros((n_ev,))
Hos_vec = np.zeros((n_ev,))

for i in range(n_ev):
    Hst_vec[i] = 0.5  * (np.transpose(ep_st[:,i]) @ M_p @ ep_st[:,i] + np.transpose(eq_st[:,i]) @ M_q @ eq_st[:,i])
    if k!=0:
        Hos_vec[i] = 0.5 * (m * ep_os[i]**2 + 1./k * eq_os[i]**2)
    else:  Hos_vec[i] = 0.5 * m * ep_os[i]**2

H_tot = Hst_vec + Hos_vec
t_ev = np.linspace(t_0, t_f, n_ev)
fig0 = plt.figure(0)
plt.plot(t_ev, Hst_vec, 'b-', label = 'Hamiltonian String (J)')
plt.plot(t_ev, Hos_vec, 'r-', label = 'Hamiltonian Oscillator (J)')
plt.plot(t_ev, H_tot, 'g-', label = 'Total Energy (J)')
plt.xlabel(r'{Time} (s)',fontsize=16)

plt.legend(loc='upper left')


fig1 = plt.figure(1)
ax = fig1.gca(projection='3d')

# Make data.


X_ev, T_ev = np.meshgrid(x_ev, t_ev)

fntsize = 20
tol = 1e-4
ax.set_xlim(min(x_ev) - tol, max(x_ev) + tol)
ax.set_xlabel('Space', fontsize=fntsize)

ax.set_ylim(min(t_ev) - tol* abs(min(t_ev)), max(t_ev) + tol* abs(max(t_ev)))
ax.set_ylabel('Time', fontsize=fntsize)

ax.set_zlabel('$w$', fontsize=fntsize)


# Customize the z axis.
ax.set_zlim(minZ-1e-3*abs(minZ) , maxZ+1e-3*abs(maxZ))
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.06f'))

# Plot the surface.
w_stT = np.transpose(w_st)
w_st_plot = ax.plot_surface(X_ev, T_ev, w_stT, cmap=cm.jet, \
                       linewidth=0, antialiased=False, label = 'Wave $w$')

# Add a color bar which maps values to colors.


w_st_plot._facecolors2d=w_st_plot._facecolors3d
w_st_plot._edgecolors2d=w_st_plot._edgecolors3d

x_os = np.ones((n_ev,))
w_os_plot = ax.plot(x_os, t_ev, w_os, label='Spring $w$', color = 'black')
ax.legend(handles=[w_os_plot[0]])

fig1.colorbar(w_st_plot, shrink=0.5, aspect=5)
plt.show()

