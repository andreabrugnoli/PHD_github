# EB beam written with the port Hamiltonian approach

from firedrake import *
import numpy as np
import scipy.linalg as la

import matplotlib
import matplotlib.pyplot as plt

plt.close('all')
matplotlib.rcParams['text.usetex'] = True

E = 2e11
rho = 7900  # kg/m^3
nu = 0.3

b = 0.05
h = 0.01
A = b * h

I = 1./12 * b * h**3

EI = E * I
L = 0.1


n = 10
deg = 3


# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
L = 1

mesh = IntervalMesh(n, L)

# plot(mesh)
# plt.show()


# Finite element defition

V_p = FunctionSpace(mesh, "Hermite", deg)
V_q = FunctionSpace(mesh, "Hermite", deg)

V = V_p * V_q

n = V.dim()
n_p = V_p.dim()

v = TestFunction(V)
v_p, v_q = split(v)

e_v = TrialFunction(V)
e_p, e_q = split(e_v)

al_p = rho * A * e_p
al_q = 1./EI * e_q

dx = Measure('dx')
ds = Measure('ds')
m_p = v_p * al_p * dx
m_q = v_q * al_q * dx
m =  m_p + m_q

j_divDiv = -v_p * e_q.dx(0).dx(0) * dx
j_divDivIP = v_q.dx(0).dx(0) * e_p * dx

j_gradgrad = v_q * e_p.dx(0).dx(0) * dx
j_gradgradIP = -v_p.dx(0).dx(0) * e_q * dx

j = j_gradgrad + j_gradgradIP
# j = j_divDiv + j_divDivIP

# bc_w = DirichletBC(V.sub(0), Constant(0.0), 1)
# bc_M = DirichletBC(V.sub(0), Constant(0.0), 2)
# boundary_dofs = sorted(bc_w.nodes)

gCC_Hess = [- v_p * ds(1), + v_p.dx(0) * ds(1), - v_p * ds(2), v_p.dx(0) * ds(2)]
gSS_Hess = [- v_p * ds(1), - v_p * ds(2)]

gCF_Hess = [- v_p * ds(1), + v_p.dx(0) * ds(1)]

gFF_divDiv = [v_q * ds(1), - v_q.dx(0) * ds(1), + v_q * ds(2), - v_q.dx(0) * ds(2)]
gSS_divDiv = [v_q * ds(1), v_q * ds(2)]

gCF_divDiv = [+ v_q * ds(2), - v_q.dx(0) * ds(2)]

g_l = gCF_Hess
g_r = [] # gCF_divDiv

G_L = np.zeros((n, len(g_l)))
G_R = np.zeros((n, len(g_r)))


for counter, item in enumerate(g_l):
    G_L[:, counter] = assemble(item).vector().get_local()

for counter, item in enumerate(g_r):
    G_R[:, counter] = assemble(item).vector().get_local()

G = np.concatenate((G_L, G_R), axis=1)

# Assemble the stiffness matrix and the mass matrix.
J = assemble(j, mat_type='aij')
M = assemble(m, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())

N_al = V.dim()
N_u = len(G.T)
# print(N_u)

Z_u = np.zeros((N_u, N_u))


J_aug = np.vstack([ np.hstack([JJ, G]),
                    np.hstack([-G.T, Z_u])
                ])

Z_al_u = np.zeros((N_al, N_u))
Z_u_al = np.zeros((N_u, N_al))

M_aug = np.vstack([ np.hstack([MM, Z_al_u]),
                    np.hstack([Z_u_al,    Z_u])
                 ])


tol = 1e-9
eigenvalues, eigvectors = la.eig(J_aug, M_aug)
omega_all = np.imag(eigenvalues)

index = omega_all >= tol

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()

k_n = omega**(0.5)*L*(rho*A/(EI))**(0.25)
print("Smallest positive normalized eigenvalues computed: ")
for i in range(10):
    print(k_n[i])

# plt.plot(np.real(eigenvalues), np.imag(eigenvalues), 'bo')
# plt.show()

eigvec_w = eigvec_omega[:n_p, :]
eigvec_w_real = np.real(eigvec_w)
eigvec_w_imag = np.imag(eigvec_w)

eig_funH2 = Function(V_p)
Vp_4proj = FunctionSpace(mesh, "CG", 2)
eig_funH1 = Function(Vp_4proj)

n_fig = 3
plot_eigenvector = True
if plot_eigenvector:

    for i in range(n_fig):
        z_real = eigvec_w_real[:, i]
        z_imag = eigvec_w_imag[:, i]

        tol = 1e-6
        fntsize = 20


        # eig_funH2.vector()[:] = z_real
        # eig_funH1.assign(project(eig_funH2, Vp_4proj))
        # plot(eig_funH1)

        eig_funH2.vector()[:] = z_imag
        eig_funH1.assign(project(eig_funH2, Vp_4proj))
        plot(eig_funH1)
        plt.xlabel('$x$', fontsize=fntsize)
        plt.title('Eigenvector $e_p$', fontsize=fntsize)

    plt.show()


