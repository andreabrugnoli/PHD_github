from firedrake import *
import numpy as np
import scipy.linalg as la
from modules_phdae.classes_phsystem import SysPhdaeRig, check_positive_matrix
import warnings
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from firedrake.plot import _two_dimension_triangle_func_val
from mpl_toolkits.mplot3d import Axes3D
from math import pi
plt.rc('text', usetex=True)
from waves import models

Lx = 1
Ly = 1
T = 1
rho = 1
nx = 8
ny = 8
wave_dirich = models.DirichletWave(Lx, Ly, rho, T, nx, ny, modes=False)
wave_neumann = models.NeumannWave(Lx, Ly, rho, T, nx, ny, modes=False)

wave = wave_dirich
E_full = wave.E
J_full = wave.J
B_full = wave.B

s0 = 0.01
n_red = 30
if wave == wave_neumann:
    oper = "grad"
else:
    oper = "div"

wave_red, V_red = wave.reduce_system(s0, n_red, oper)
Vall_red = la.block_diag(V_red, np.eye(wave.n_lmb))

E_red = wave_red.E
J_red = wave_red.J
B_red = wave_red.B

# J_red = np.vstack([np.hstack([J_red, B_red]),
#                       np.hstack([-B_red.T, Z_u])
#                       ])
#
# E_red = la.block_diag(E_red, Z_u)

tol = 10 ** (-6)

eigenvaluesF, eigvectorsF = la.eig(J_full, E_full)
omega_allF = np.imag(eigenvaluesF)

indexF = omega_allF >= tol

omega_full = omega_allF[indexF]
eigvec_full = eigvectorsF[:, indexF]
permF = np.argsort(omega_full)
eigvec_full = eigvec_full[:, permF]
omega_full.sort()

eigenvaluesR, eigvectorsR = la.eig(J_red, E_red)
omega_allR = np.imag(eigenvaluesR)

index = omega_allR >= tol

omega_red = omega_allR[index]
eigvec_red = eigvectorsR[:, index]
permR = np.argsort(omega_red)
eigvec_red = eigvec_red[:, permR]
omega_red.sort()

plt.plot(np.real(eigenvaluesF), np.imag(eigenvaluesF), 'r+', np.real(eigenvaluesR), np.imag(eigenvaluesR), 'bo')
plt.legend(("Eigenvalues full", "Eigenvalues reduced"))
plt.show()

# NonDimensional China Paper

n_om = 5

omegaF_tilde = omega_full
omegaR_tilde = omega_red

for i in range(n_om):
    print(omegaF_tilde[i])

for i in range(n_om):
    print(omegaF_tilde[i], omegaR_tilde[i])
