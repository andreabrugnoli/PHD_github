from firedrake import *
import numpy as np
import scipy.linalg as la
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt
from scipy.io import savemat
plt.rc('text', usetex=True)
# Finite element defition

n = 5
# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
# ind = 9
mesh = UnitSquareMesh(n, n, quadrilateral=False)

figure = plt.figure()
ax = figure.add_subplot(111)
# plot(mesh, axes=ax)
# plt.show()

rho = 1
T = 1


deg_p = 0
deg_q = 1
Vp = FunctionSpace(mesh, "DG", deg_p)
Vq = FunctionSpace(mesh, "RT", deg_q)
# Vq = VectorFunctionSpace(mesh, "Lagrange", deg_q)

V = Vp * Vq

n_V = V.dim()
print(n_V)

v = TestFunction(V)
v_p, v_q = split(v)

e_v = TrialFunction(V)
e_p, e_q = split(e_v)

al_p = rho * e_p
al_q = 1/T * e_q


dx = Measure('dx')
ds = Measure('ds')

m_p = v_p * al_p * dx
m_q = dot(v_q, al_q) * dx
m_form = m_p + m_q

j_div = v_p * div(e_q) * dx
j_divIP = -div(v_q) * e_p * dx

j_form = j_div + j_divIP
petsc_j = assemble(j_form, mat_type='aij').M.handle
petsc_m = assemble(m_form, mat_type='aij').M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())

n_p = Vp.dim()
n_q = Vq.dim()

n = FacetNormal(mesh)

V_u = FunctionSpace(mesh, "CG", 1)

u = TrialFunction(V_u)

b_form = dot(v_q, n) * u *ds
petsc_b_u = assemble(b_form, mat_type='aij').M.handle
BB = np.array(petsc_b_u.convert("dense").getDenseArray())
boundary_dofs = np.where(BB.any(axis=0))[0]  # np.where(~np.all(B_in == 0, axis=0) == True) #
BB = BB[:, boundary_dofs]

n_u = BB.shape[1]

QQ = la.inv(MM)
A_sys = JJ @ QQ
B_sys = BB
C_sys = BB.T @ QQ
D_sys = np.zeros((n_u, n_u))

pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/PH/PH_Control/Matrices_EB/'
A_file = 'A'; B_file = 'B'; C_file = 'C'; D_file = 'D';
savemat(pathout + A_file, mdict={A_file: np.array(A_sys)}, appendmat=True)
savemat(pathout + B_file, mdict={B_file: np.array(B_sys)}, appendmat=True)
savemat(pathout + C_file, mdict={C_file: np.array(C_sys)}, appendmat=True)
savemat(pathout + D_file, mdict={D_file: np.array(D_sys)}, appendmat=True)