# Mindlin plate written with the port Hamiltonian approach

from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)
from math import floor, sqrt, pi
import matplotlib.pyplot as plt
import scipy.linalg as la



E = 2e11
rho = 7900  # kg/m^3
nu = 0.3

b = 0.05
h = 0.01
A = b * h

I = 1./12 * b * h**3

EI = E * I
L = 1
coeff_norm = L*(rho*A/EI)**(0.25)

n = 2
deg = 3

mesh = IntervalMesh(n, L)

# plot(mesh)
# plt.show()


# Finite element defition

V_p = FunctionSpace(mesh, "Hermite", deg)
V_q = FunctionSpace(mesh, "Hermite", deg)

V = V_p * V_q

print(V.dim())

n_Vp = V_p.dim()

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


# j = j_divDiv + j_divDivIP
j = j_gradgrad + j_gradgradIP


bc_w = DirichletBC(V.sub(0), Constant(0.0), 1)
# bc_M = DirichletBC(V.sub(0), Constant(0.0), 2)

g_Hess = - v_p * ds + v_p.dx(0) * ds

boundary_dofs = sorted(bc_w.nodes)

# Assemble the stiffness matrix and the mass matrix.


J = assemble(j, mat_type='aij')
M = assemble(m, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())


tol = 0
eigenvalues, eigvectors = la.eig(JJ, MM)
omega_all = np.imag(eigenvalues)

index = omega_all>tol

omega = omega_all[index]
eigvec_omega = eigvectors[:, index]
perm = np.argsort(omega)
eigvec_omega = eigvec_omega[:, perm]

omega.sort()

k_n = np.sqrt(omega)*coeff_norm
print("Smallest positive normalized eigenvalues computed: ")
for i in range(10):
    print(k_n[i])
#
# plt.plot(np.real(eigenvalues), np.imag(eigenvalues), 'bo')

eigvec_w = eigvec_omega[:n_Vp, :]
eigvec_w_real = np.real(eigvec_w)
eigvec_w_imag = np.imag(eigvec_w)

eig_funH2 = Function(V_p)
Vp_4proj = FunctionSpace(mesh, "CG", 2)
eig_funH1 = Function(Vp_4proj)

n_fig = 3
plot_eigenvector = False
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
#
#
