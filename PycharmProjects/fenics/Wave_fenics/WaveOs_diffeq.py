# Mindlin plate written with the port Hamiltonian approach

from fenics import *
import numpy as np
np.set_printoptions(threshold=np.inf)
from scipy import linalg as la
# import mshr
# from scipy.sparse import csr_matrix
# from scipy.sparse.linalg import spsolve


n = 10
deg = 1

rho = 1
T = 1
k_os = 1
m_os = 2
r_os = 0

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

V = FunctionSpace(mesh, MixedElement([Pp, Pq]))

n_V = V.dim()

dofV_x = V.tabulate_dof_coordinates().reshape((-1, d))
Vp = V.sub(0)
Vq = V.sub(1)
dofs_Vp = Vp.dofmap().dofs()
dofs_Vq = Vq.dofmap().dofs()

dofVp_x = dofV_x[dofs_Vp]

v_p, v_q = TestFunction(V)

e_p, e_q = TrialFunction(V)


al_p = rho * e_p
al_q = 1./T * e_q


m_p = inner(v_p, al_p) * dx
m_q = inner(v_q, al_q) * dx
m = m_p + m_q

j_div = v_p * e_q.dx(0) * dx # v_p * div(e_q) * dx
j_divIP = -v_q.dx(0) * e_p *dx # -div(v_q) * e_p * dx

j_grad = v_q * e_p.dx(0) * dx # dot(v_q, grad(e_p)) * dx
j_gradIP = -v_p.dx(0) * e_q *dx # dot(-grad(v_p), e_q) * dx


j = j_grad + j_gradIP

# Assemble the interconnection matrix and the mass matrix.

Jst = assemble(j).array()
Mst = assemble(m).array()

G_l = assemble(-v_p * ds(1)).get_local()
G_r = assemble(v_p * ds(2)).get_local()

G_l = np.reshape(G_l, (-1, 1))
G_r = np.reshape(G_r, (-1, 1))
G_lT = G_l.T
G_rT = G_r.T

G = np.concatenate((G_l, G_r), axis=1)
GT = G.T

Mos = np.diag([m_os, 1./k_os])
invMst = la.inv(Mst)
invMos = la.inv(Mos)

Jos = np.array([[0, -1],
                [1,  0]])
Ros = np.array([[r_os, 0],
                [0, 0]])

x_ev = dofVp_x[:,0]
ind = np.where(x_ev == 1)


e_in =Function(V)
e_in.assign(interpolate(Expression(('sin(pi/(2*L)*x[0])', '0'), degree=3, L=L) , V))
e_st0 = e_in.vector().get_local()

e_os0 = np.zeros((2,))
e_os0[0] = e_st0[ind]
n_l = 1; n_r =1
n_os = 2

n_tot = n_V + n_l + n_r + n_os
end1 = n_V
end2 = end1 + n_l + n_r
end3 = n_tot

Jaug_pl = np.zeros((end2, end2))
Jaug_pl[:end1, :end1] = Jst
Jaug_pl[:end1, end1:end2] = G
Jaug_pl[end1:end2, :end1] = -GT


Cint = np.array([[0, 0],[1, 0]])

Mint = la.block_diag(Mst, np.zeros((n_l, n_l)), np.zeros((n_r, n_r)), Mos)

Jint = np.zeros((end3, end3))
Jint[:end2, :end2] = Jaug_pl
Jint[end2:end3, end2:end3] = Jos
Jint[end1:end2, end2:end3] = Cint
Jint[end2:end3, end1:end2] = -Cint.T

Rint = np.zeros((end3, end3))
Rint[end2:end3, end2:end3] = Ros

y0 = np.zeros((n_tot, )) # Initial conditions
y0[:end1] = e_st0
y0[end2:end3] = e_os0



def sysODE(u, p, t):
    return  (Jint - Rint) @ u

def sysDAE(du, u, p, t):
    residual = Mint @ du - (Jint - Rint) @ u
    return

from diffeqpy import de

t_0 = 0.
t_f = 10.
n_ev = 100
t_ev = np.linspace(t_0, t_f, n_ev)

y0 = np.zeros((n_tot, )) # Initial conditions
y0[:end1] = e_st0
y0[end2:end3] = e_os0
yd0 = np.zeros((n_tot, ))  # Initial conditions

t_span = (t_0, t_f)

# m_ode_prob = de.ODEProblem(sysODE, y0, t_span)

differential_vars = np.zeros((end3,))
differential_vars[:end1] = 1
differential_vars[end2:end3] = 1
dae_prob = de.DAEProblem(sysDAE, y0, yd0, t_span, differential_vars=differential_vars)

sol = de.solve(dae_prob)


# dt_vec = np.diff(t)
# n_ev  = len(sol[:,1])
# e_st = sol[:, :end1].T
# e_os = sol[:, end2:end3].T
#
# ep_st = e_st[dofs_Vp, :]
# ep_os = e_os[0, :]
#
# n_p = Vp.dim()
#
# w_st0 = np.zeros((n_p,))
# w_st = np.zeros(ep_st.shape)
# w_st[:,0] = w_st0
# w_st_old = w_st[:,0]
#
# w_os0 = 0
# w_os = np.zeros((n_ev,))
# w_os[0] = w_os0
# w_os_old = w_os[0]
#
# for i in range(1,n_ev):
#     w_st[:,i] = w_st_old + 0.5*(ep_st[:,i-1] + ep_st[:,i])*dt_vec[i-1]
#     w_st_old  = w_st[:,i]
#
#     w_os[i] = w_os_old + 0.5*(ep_os[i-1] + ep_os[i])*dt_vec[i-1]
#     w_os_old = w_os[i]
#
# minZ = min((w_st.min(), w_os.min()))
# maxZ = max((w_st.max(), w_os.max()))
#
# import matplotlib
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# from mpl_toolkits.mplot3d import Axes3D
#
# matplotlib.rcParams['text.usetex'] = True
#
#
# if matplotlib.is_interactive():
#     plt.ioff()
# plt.close('all')
#
# path_out = "/home/a.brugnoli/PycharmProjects/Mindlin_Phs_fenics/Temp_Simulation/"
#
# Hst_vec = np.zeros((n_ev,))
# Hos_vec = np.zeros((n_ev,))
#
# for i in range(n_ev):
#     Hst_vec[i] = 0.5  * (e_st[:, i].T @ Mst @ e_st[:, i])
#     Hos_vec[i] = 0.5 * (e_os[:, i].T @ Mos @ e_os[:, i])
#
# H_tot = Hst_vec + Hos_vec
# t_ev = np.linspace(t_0, t_f, n_ev)
# fig0 = plt.figure(0)
# plt.plot(t_ev, Hst_vec, 'b-', label = 'Hamiltonian String (J)')
# plt.plot(t_ev, Hos_vec, 'r-', label = 'Hamiltonian Oscillator (J)')
# plt.plot(t_ev, H_tot, 'g-', label = 'Total Energy (J)')
# plt.xlabel(r'{Time} (s)',fontsize=16)
#
# plt.legend(loc='upper left')
#
#
# fig1 = plt.figure(1)
# ax = fig1.gca(projection='3d')
#
# # Make data.
#
#
# X_ev, T_ev = np.meshgrid(x_ev, t_ev)
#
# fntsize = 20
# tol = 1e-4
# ax.set_xlim(min(x_ev) - tol, max(x_ev) + tol)
# ax.set_xlabel('Space', fontsize=fntsize)
#
# ax.set_ylim(min(t_ev) - tol* abs(min(t_ev)), max(t_ev) + tol* abs(max(t_ev)))
# ax.set_ylabel('Time', fontsize=fntsize)
#
# ax.set_zlabel('$w$', fontsize=fntsize)
#
#
# # Customize the z axis.
# ax.set_zlim(minZ-1e-3*abs(minZ) , maxZ+1e-3*abs(maxZ))
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.06f'))
#
# # Plot the surface.
# w_stT = np.transpose(w_st)
# w_st_plot = ax.plot_surface(X_ev, T_ev, w_stT, cmap=cm.jet, \
#                        linewidth=0, antialiased=False, label = 'Wave $w$')
#
# # Add a color bar which maps values to colors.
#
#
# w_st_plot._facecolors2d=w_st_plot._facecolors3d
# w_st_plot._edgecolors2d=w_st_plot._edgecolors3d
#
# x_os = np.ones((n_ev,))
# w_os_plot = ax.plot(x_os, t_ev, w_os, label='Spring $w$', color = 'black')
# ax.legend(handles=[w_os_plot[0]])
# #
# # fig1.colorbar(w_st_plot, shrink=0.5, aspect=5)
# plt.show()
#
