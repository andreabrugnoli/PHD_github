# Mindlin plate written with the port Hamiltonian approach

from firedrake import *
import numpy as np

np.set_printoptions(threshold=np.inf)
from math import floor, sqrt, pi
import matplotlib.pyplot as plt

from scipy import linalg as la
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


E = 7e10
nu = 0.35
h = 0.05 # 0.01
rho = 2700  # kg/m^3
D = E * h ** 3 / (1 - nu ** 2) / 12.

L = 1
l_x = L
l_y = L

n = 5 #int(input("N element on each side: "))

m_rod = 100
Jxx_rod = 1. / 12 * m_rod * L ** 2

k_sp1 = 10
k_sp2 = 10

r_sp1 = 0
r_sp2 = 0

# Plate bending stiffness :math:`D=\dfrac{Eh^3}{12(1-\nu^2)}` and shear stiffness :math:`F = \kappa Gh`
# with a shear correction factor :math:`\kappa = 5/6` for a homogeneous plate
# of thickness :math:`h`::
# E = Constant(7e10)
# nu = Constant(0.3)
# h = Constant(0.2)
# rho = Constant(2000)  # kg/m^3
# k = Constant(5. / 6.)

# Useful Matrices

D_b = as_tensor([
    [D, D * nu, 0],
    [D * nu, D, 0],
    [0, 0, D * (1 - nu) / 2]
])

fl_rot = 12. / (E * h ** 3)

C_b_vec = as_tensor([
    [fl_rot, -nu * fl_rot, 0],
    [-nu * fl_rot, fl_rot, 0],
    [0, 0, fl_rot * 2 * (1 + nu)]
])


# Vectorial Formulation possible only
def bending_moment_vec(kk):
    return dot(D_b, kk)

def bending_curv_vec(MM):
    return dot(C_b_vec, MM)

def tensor_divDiv_vec(MM):
    return MM[0].dx(0).dx(0) + MM[1].dx(1).dx(1) + 2 * MM[2].dx(0).dx(1)

def Gradgrad_vec(u):
    return as_vector([ u.dx(0).dx(0), u.dx(1).dx(1), 2 * u.dx(0).dx(1) ])

def tensor_Div_vec(MM):
    return as_vector([ MM[0].dx(0) + MM[2].dx(1), MM[2].dx(0) + MM[1].dx(1) ])

# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
L = 1
l_x = L
l_y = L
n_x, n_y = n, n
mesh = UnitSquareMesh(n_x, n_y, quadrilateral=False)

# plot(mesh)
# plt.show()

nameFE = 'Bell'
name_FEp = nameFE
name_FEq = nameFE

if name_FEp == 'Morley':
    deg_p = 2
elif name_FEp == 'Hermite':
    deg_p = 3
elif name_FEp == 'Argyris' or name_FEp == 'Bell':
    deg_p = 5

if name_FEq == 'Morley':
    deg_q = 2
elif name_FEq == 'Hermite':
    deg_q = 3
elif name_FEq == 'Argyris' or name_FEq == 'Bell':
    deg_q = 5

Vp = FunctionSpace(mesh, name_FEp, deg_p)
Vq = VectorFunctionSpace(mesh, name_FEq, deg_q, dim=3)


n_Vp = Vp.dim()
n_Vq = Vq.dim()



v_p = TestFunction(Vp)
v_q = TestFunction(Vq)

e_p = TrialFunction(Vp)
e_q = TrialFunction(Vq)

al_p = rho * h * e_p
al_q = bending_curv_vec(e_q)

# e_p = 1./(rho * h) * al_p
# e_q = bending_moment_vec(al_q)

# alpha = TrialFunction(V)
# al_pw, al_pth, al_qth, al_qw = split(alpha)

# e_p = 1. / (rho * h) * al_p
# e_q = bending_moment_vec(al_q)

dx = Measure('dx')
ds = Measure('ds')
m_p = dot(v_p, al_p) * dx
m_q = inner(v_q, al_q) * dx
m = m_p + m_q

j_divDiv = -v_p * tensor_divDiv_vec(e_q) * dx
j_divDivIP = tensor_divDiv_vec(v_q) * e_p * dx

j_gradgrad = inner(v_q, Gradgrad_vec(e_p)) * dx
j_gradgradIP = -inner(Gradgrad_vec(v_p), e_q) * dx

# j = j_gradgrad + j_gradgradIP  #
j_p = j_gradgrad
j_q = j_gradgradIP

Jp = assemble(j_p, mat_type='aij')
Mp = assemble(m_p, mat_type='aij')

Mq = assemble(m_q, mat_type='aij')
Jq = assemble(j_q, mat_type='aij')


petsc_j_p = Jp.M.handle
petsc_m_p = Mp.M.handle

petsc_j_q = Jq.M.handle
petsc_m_q = Mq.M.handle

Dp_pl = np.array(petsc_j_p.convert("dense").getDenseArray())
Mp_pl = np.array(petsc_m_p.convert("dense").getDenseArray())

Dq_pl = np.array(petsc_j_q.convert("dense").getDenseArray())
Mq_pl = np.array(petsc_m_q.convert("dense").getDenseArray())

# Dirichlet Boundary Conditions and related constraints
# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1

n = FacetNormal(mesh)
# s = as_vector([-n[1], n[0]])

V_qn = FunctionSpace(mesh, 'Lagrange', 2)
V_Mnn = FunctionSpace(mesh, 'Lagrange', 2)

Vu = V_qn * V_Mnn

q_n, M_nn = TrialFunction(Vu)

v_wt, v_omn = TestFunction(Vu)

v_omn = dot(grad(v_p), n)
b_l = v_p * q_n * ds(1) + v_omn * M_nn * ds(1)
b_r = v_p * q_n * ds(2)

# Assemble the stiffness matrix and the mass matrix.

B_l = assemble(b_l, mat_type='aij')
B_r = assemble(b_r, mat_type='aij')

petsc_b_l = B_l.M.handle
petsc_b_r = B_r.M.handle

# print(B_u.array().shape)
G_l = np.array(petsc_b_l.convert("dense").getDenseArray())
G_r = np.array(petsc_b_r.convert("dense").getDenseArray())

bd_dofs_l = np.where(G_l.any(axis=0))[0]# np.where(~np.all(B_in == 0, axis=0) == True) #
bd_dofs_r = np.where(G_r.any(axis=0))[0]

G_l = G_l[:, bd_dofs_l]
G_r = G_r[:, bd_dofs_r]
n_l = len(bd_dofs_l)
n_r = len(bd_dofs_r)

# Splitting of matrices

# Force applied at the right boundary
x, y = SpatialCoordinate(mesh)
g = Constant(10)
A = Constant(10**5)
f_w = project(A*sin(2*pi/l_y*y), Vp) # project(A*(y + (y-l_y/2)**2), Vp) #
# f_w = Expression("1000000*sin(2*pi*x[0])", degree=4)
# f_qn = Expression("1000000*sin(2*pi*x[1])", degree= 4) # Constant(1e5) #
b_p = v_p * f_w * dx                           # v_p * f_w * ds(3) - v_p * f_w * ds(4)
#                                # v_p * f_qn * ds(3) #
#                                # -v_p * rho * h * g * dx #

B_p = assemble(b_p, mat_type='aij').vector().get_local()

# Final Assemble
Mp_sp = csr_matrix(Mp_pl)
Mq_sp = csr_matrix(Mq_pl)

# Final Assemble

Mp_rod = np.diag([m_rod, Jxx_rod])
invMp_rod = la.inv(Mp_rod)
Mq_rod = np.diag([1./k_sp1, 1./k_sp2])
invMq_rod = np.diag([k_sp1, k_sp2])

Dp_rod = np.array([[1, l_y/2], [1, -l_y/2]])
Dq_rod = - Dp_rod.T


r_v = r_sp1 + r_sp2
r_th = l_y**2/4*(r_sp1 - r_sp2)
r_vth = l_y/2*(r_sp1 - r_sp2)
R_rod = np.array([[r_v, r_vth], [r_vth, r_th]])

Mp_sp = csr_matrix(Mp_pl)
Mq_sp = csr_matrix(Mq_pl)

G_rT = G_r.T
G_lT = G_l.T

G = np.concatenate((G_l, G_r), axis=1)
GT = G.T

invMp_pl = la.inv(Mp_pl)
invMq_pl = la.inv(Mq_pl)

CT = np.zeros((n_r, 2))


CT[:, 0] = assemble(v_wt * ds(2)).vector().get_local()[bd_dofs_r]
CT[:, 1] = assemble(v_wt * (y - l_y/2) * ds(2)).vector().get_local()[bd_dofs_r]

C = CT.T
GMG = GT @ invMp_pl @ G

CMC_rr = CT @ invMp_rod @ C
CMC = np.zeros((n_l + n_r, n_l + n_r))
CMC[n_l: n_l + n_r, n_l:n_l + n_r] = CMC_rr
M_lambda = GT @ invMp_pl @ G + CMC
invM_lambda = la.inv(M_lambda)

invM_lmb_r = invM_lambda[n_l:, :]

Bpl_q = GT @ invMp_pl @ Dq_pl
Brod_q = np.vstack( (np.zeros((n_l, 2)), CT) ) @ invMp_rod @ Dq_rod
Brod_p = np.vstack( (np.zeros((n_l, 2)), CT) ) @ invMp_rod @ R_rod

t_0 = 0
dt = 1e-6
fac = 5
t_fac = 1
t_f = 0.001 * t_fac
n_ev = 100
t_ev = np.linspace(t_0, t_f, n_ev)

n_t = int(t_f / dt)

ep_pl_sol = np.zeros((n_Vp, n_ev))
eq_pl_sol = np.zeros((n_Vq, n_ev))
ep_rod_sol = np.zeros((2, n_ev))
eq_rod_sol = np.zeros((2, n_ev))

ep_pl0 = Function(Vp)
ep_pl0.assign(project(x/100, Vp))
ep_pl_old = np.zeros((n_Vp)) # ep_pl0.vector().get_local()  #
eq_pl_old = np.zeros((n_Vq))

ep_rod_old = np.zeros((2,)) # np.array([0.01, 0]) #
eq_rod_old = np.zeros((2,))

ep_pl_sol[:, 0] = ep_pl_old
eq_pl_sol[:, 0] = eq_pl_old

ep_rod_sol[:, 0] = ep_rod_old
eq_rod_sol[:, 0] = eq_rod_old

Arod_p = Mp_rod + 0.5*dt*R_rod - 0.5*dt* C @ invM_lmb_r @ Brod_p

invArod_p = la.inv(Arod_p)

k = 1
f = 0
for i in range(1, n_t + 1):

    t = i * dt
    if t < t_f / (fac * t_fac):
        f = 1
    else:
        f = 0
    # Intergation for p (n+1/2)

    # Bpl_q = GT @ invMp_pl @ Dq_pl
    # Brod_q = np.vstack((np.zeros((n_l, 2)), CT)) @ invMp_rod @ Dq_rod
    # Brod_p = np.vstack((np.zeros((n_l, 2)), CT)) @ invMp_rod @ R_rod

    # lmbda = invM_lambda @ (-GT @ invMp_pl @ (Dq_pl @ eq_pl_old + B_p * f)\
    #                        +np.vstack( (np.zeros((n_l, 2)), CT) ) @ Mp_rod @ Dq_rod @ eq_rod_old)
    #
    # bp_rod = Mp_rod @ ep_rod_old + 0.5 * dt * (- C @ lmbda[n_l:] + Dq_rod @ eq_rod_old)
    #
    # ep_rod_new = invMp_rod @ bp_rod
    #
    # bp_pl = Mp_pl @ ep_pl_old + 0.5 * dt * (Dq_pl @ eq_pl_old + G @ lmbda + B_p * f)
    # bp_pl_sp = csr_matrix(bp_pl).reshape((n_Vp, 1))
    # ep_pl_new = spsolve(Mp_sp, bp_pl_sp)
    #
    # ep_pl_old = ep_pl_new
    # ep_rod_old = ep_rod_new

    bp_rod = Mp_rod @ ep_rod_old +\
             0.5 * dt * (- C @ invM_lmb_r @ (Brod_q @ eq_rod_old - Bpl_q @ eq_pl_old \
                          -GT @ invMp_pl @ B_p * f) + Dq_rod @ eq_rod_old)

    ep_rod_new = invArod_p @ bp_rod

    lmbda = invM_lambda @ ( - Bpl_q @ eq_pl_old \
                            + Brod_q @ eq_rod_old \
                            - Brod_p @ ep_rod_new -GT @ invMp_pl @ B_p * f)

    bp_pl = Mp_pl @ ep_pl_old + 0.5 * dt * (Dq_pl @ eq_pl_old + G @ lmbda + B_p * f)
    bp_pl_sp = csr_matrix(bp_pl).reshape((n_Vp, 1))
    ep_pl_new = spsolve(Mp_sp, bp_pl_sp)

    ep_pl_old = ep_pl_new
    ep_rod_old = ep_rod_new

    # Integration of q (n+1)
    bq_rod = Mq_rod @ eq_rod_old + dt * Dp_rod @ ep_rod_new

    eq_rod_new = invMq_rod @ bq_rod

    bq_pl = Mq_pl @ eq_pl_old + dt * Dp_pl @ ep_pl_new

    bq_pl_sp = csr_matrix(bq_pl).reshape((n_Vq, 1))
    eq_pl_new = spsolve(Mq_sp, bq_pl_sp)

    eq_pl_old = eq_pl_new
    eq_rod_old = eq_rod_new

    # Intergation for p (n+1)

    bp_rod = Mp_rod @ ep_rod_old + \
             0.5 * dt * (- C @ invM_lmb_r @ (Brod_q @ eq_rod_old - Bpl_q @ eq_pl_old \
                                             - GT @ invMp_pl @ B_p * f) + Dq_rod @ eq_rod_old)

    ep_rod_new = invArod_p @ bp_rod

    lmbda = invM_lambda @ (- Bpl_q @ eq_pl_old \
                           + Brod_q @ eq_rod_old \
                           - Brod_p @ ep_rod_new - GT @ invMp_pl @ B_p * f)

    bp_pl = Mp_pl @ ep_pl_old + 0.5 * dt * (Dq_pl @ eq_pl_old + G @ lmbda + B_p * f)
    bp_pl_sp = csr_matrix(bp_pl).reshape((n_Vp, 1))
    ep_pl_new = spsolve(Mp_sp, bp_pl_sp)

    ep_pl_old = ep_pl_new
    ep_rod_old = ep_rod_new

    # # Verify Constraints

    if t >= t_ev[k]:
        ep_pl_sol[:, k] = ep_pl_new
        eq_pl_sol[:, k] = eq_pl_new

        ep_rod_sol[:, k] = ep_rod_new
        eq_rod_sol[:, k] = eq_rod_new
        k = k + 1
        print('Solution number ' + str(k) + ' computed')


e_pw = ep_pl_sol # np.zeros((n_pw,n_t))

w0_pl = np.zeros((n_Vp,))
w_pl = np.zeros(e_pw.shape)
w_pl[:, 0] = w0_pl
w_pl_old = w_pl[:, 0]
Deltat = t_f / (n_ev - 1)
for i in range(1, n_ev):
    w_pl[:, i] = w_pl_old + 0.5 * (e_pw[:, i - 1] + e_pw[:, i]) * Deltat
    w_pl_old = w_pl[:, i]


y_rod = np.array([0, 1])
n_rod = len(y_rod)
x_rod = np.ones((n_rod,))

v_rod = np.zeros((n_rod, n_ev))
w_rod = np.zeros((n_rod, n_ev))
w_rod_old = w_rod[:, 0]

for i in range(n_ev):

    v_rod[:, i] = x_rod * ep_rod_sol[0, i] + (y_rod - l_y / 2) * ep_rod_sol[1, i]
    if i >= 1:
        w_rod[:, i] = w_rod_old + 0.5 * (v_rod[:, i - 1] + v_rod[:, i]) * Deltat
        #x_rod * eq_rod_sol[1, i] / k_sp2 + y_rod * eq_rod_sol[0, i] / k_sp1
        w_rod_old = w_rod[:, i]

w_pl_mm = w_pl * 1000
w_rod_mm = w_rod * 1000

wmm_pl_CGvec = []
w_fun = Function(Vp)
Vp_CG = FunctionSpace(mesh, 'Lagrange', 3)
n_VpCG = Vp_CG.dim()
print(n_Vp, n_VpCG)

maxZvec = np.zeros(n_ev)
minZvec = np.zeros(n_ev)
for i in range(n_ev):
    w_fun.vector()[:] = w_pl_mm[:, i]
    wmm_pl_CG = project(w_fun, Vp_CG)
    wmm_pl_CGvec.append(wmm_pl_CG)

    maxZvec[i] = max(wmm_pl_CG.vector())
    minZvec[i] = min(wmm_pl_CG.vector())

maxZ = max(maxZvec)
minZ = min(minZvec)

import drawNow2, matplotlib
# import tkinter as tk

matplotlib.interactive(True)

matplotlib.rcParams['text.usetex'] = True

plotter = drawNow2.plot3dClass(wmm_pl_CGvec[0], minZ=minZ, maxZ=maxZ, X2=x_rod, Y2=y_rod, \
                               xlabel='$x[m]$', ylabel='$y [m]$', \
                               zlabel='$w [mm]$', title='Vertical Displacement')

for i in range(n_ev):
    wpl_t = w_pl_mm[:, i]
    wrod_t = w_rod_mm[:, i]
    plotter.drawNow2(wmm_pl_CGvec[i], wrod_t, z2label='Rod $w[mm]$')

# tk.mainloop()

if matplotlib.is_interactive():
    plt.ioff()
# plt.close("all")

Hpl_vec = np.zeros((n_ev,))
Hrod_vec = np.zeros((n_ev,))

for i in range(n_ev):
    Hpl_vec[i] = 0.5 * (ep_pl_sol[:, i].T @ Mp_pl @ ep_pl_sol[:, i]\
                        + eq_pl_sol[:, i].T @ Mq_pl @ eq_pl_sol[:, i])
    Hrod_vec[i] = 0.5 * (ep_rod_sol[:, i].T @ Mp_rod @ ep_rod_sol[:, i]\
                         + eq_rod_sol[:, i].T @ Mq_rod @ eq_rod_sol[:, i])

t_ev = np.linspace(t_0, t_f, n_ev)
fig = plt.figure(0)
plt.plot(t_ev, Hpl_vec, 'b-', label='Hamiltonian Plate (J)')
plt.plot(t_ev, Hrod_vec, 'r-', label='Hamiltonian Rod (J)')
plt.plot(t_ev, Hpl_vec + Hrod_vec, 'g-', label='Total Energy (J)')
plt.xlabel(r'{Time} (s)', fontsize=16)
plt.ylabel(r'{Hamiltonian} (J)', fontsize=16)
plt.title(r"Hamiltonian trend",
          fontsize=16)
plt.legend(loc='upper left')

plt.show()