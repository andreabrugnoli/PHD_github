# Controllability Kirchhoff plate HHJ

from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt

import scipy.linalg as la

import matplotlib
import matplotlib.pyplot as plt
from firedrake.plot import _two_dimension_triangle_func_val
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from scipy.io import savemat

matplotlib.rcParams['text.usetex'] = True

n = 10
r = 1 #int(input('Degree for FE: '))

E = 2e11 # Pa
rho = 8000  # kg/m^3
# E = 1
# rho = 1  # kg/m^3

nu = 0.3
h = 0.01

plot_eigenvector = 'y'

bc_input = input('Select Boundary Condition: ')

L = 1

D = E * h ** 3 / (1 - nu ** 2) / 12
fl_rot = 12 / (E * h ** 3)
norm_coeff = L ** 2 * np.sqrt(rho*h/D)
# Useful Matrices

# Operators and functions
def gradSym(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
    # return sym(nabla_grad(u))

def bending_curv(momenta):
    kappa = fl_rot * ((1+nu)*momenta - nu * Identity(2) * tr(momenta))
    return kappa


# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
n_x, n_y = n, n
mesh = UnitSquareMesh(n_x, n_y, quadrilateral=False)

# domain = mshr.Rectangle(Point(0, 0), Point(l_x, l_y))
# mesh = mshr.generate_mesh(domain, n, "cgal")

# plot(mesh)
# plt.show()


# Finite element defition

Vp = FunctionSpace(mesh, 'CG', r)
Vq = FunctionSpace(mesh, 'HHJ', r-1)
V = Vp * Vq

n_Vp = V.sub(0).dim()
n_Vq = V.sub(1).dim()
n_V = V.dim()
print(n_V)

v = TestFunction(V)
v_p, v_q = split(v)

e = TrialFunction(V)
e_p, e_q = split(e)

al_p = rho * h * e_p
al_q = bending_curv(e_q)

dx = Measure('dx')
ds = Measure('ds')
dS = Measure("dS")

m_form = inner(v_p, al_p) * dx + inner(v_q, al_q) * dx

n_ver = FacetNormal(mesh)
s_ver = as_vector([-n_ver[1], n_ver[0]])

e_mnn = inner(e_q, outer(n_ver, n_ver))
v_mnn = inner(v_q, outer(n_ver, n_ver))

e_mns = inner(e_q, outer(n_ver, s_ver))
v_mns = inner(v_q, outer(n_ver, s_ver))

j_1 = - inner(grad(grad(v_p)), e_q) * dx \
      + jump(grad(v_p), n_ver) * dot(dot(e_q('+'), n_ver('+')), n_ver('+')) * dS \
      + dot(grad(v_p), n_ver) * dot(dot(e_q, n_ver), n_ver) * ds

j_2 = + inner(v_q, grad(grad(e_p))) * dx \
      - dot(dot(v_q('+'), n_ver('+')), n_ver('+')) * jump(grad(e_p), n_ver) * dS \
      - dot(dot(v_q, n_ver), n_ver) * dot(grad(e_p), n_ver) * ds \



j_form = j_1 + j_2

bc_1, bc_2, bc_3, bc_4 = bc_input

bc_dict = {1: bc_1, 3: bc_2, 2: bc_3, 4: bc_4}

# The boundary edges in this mesh are numbered as follows:

# 1: plane x == 0
# 2: plane x == 1
# 3: plane y == 0
# 4: plane y == 1

bcs = []
boundary_dofs = []

for key, val in bc_dict.items():

    if val == 'C':
        bc_p = DirichletBC(Vp, Constant(0.0), key)
        for node in bc_p.nodes:
            boundary_dofs.append(node)
        bcs.append(bc_p)

    elif val == 'S':
        bc_p = DirichletBC(Vp, Constant(0.0), key)
        bc_q = DirichletBC(Vq, Constant(((0.0, 0.0), (0.0, 0.0))), key)

        for node in bc_p.nodes:
            boundary_dofs.append(node)
        bcs.append(bc_p)

        for node in bc_q.nodes:
            boundary_dofs.append(n_Vp + node)
        bcs.append(bc_q)

    elif val == 'F':
        bc_q = DirichletBC(Vq, Constant(((0.0, 0.0), (0.0, 0.0))), key)
        for node in bc_q.nodes:
            boundary_dofs.append(n_Vp + node)
        bcs.append(bc_q)

boundary_dofs = sorted(boundary_dofs)
n_lmb = len(boundary_dofs)

G = np.zeros((n_V, n_lmb))
for (i, j) in enumerate(boundary_dofs):
    G[j, i] = 1

J = assemble(j_form, mat_type='aij')
M = assemble(m_form, mat_type='aij')

petsc_j = J.M.handle
petsc_m = M.M.handle

JJ = np.array(petsc_j.convert("dense").getDenseArray())
MM = np.array(petsc_m.convert("dense").getDenseArray())

BB = G
n_u = BB.shape[1]

QQ = la.inv(MM)
A_sys = JJ @ QQ
B_sys = G
C_sys = G.T @ QQ
D_sys = np.zeros((n_u, n_u))

pathout = '/home/a.brugnoli/GitProjects/MatlabProjects/PH/PH_Control/Matrices_EB/'
A_file = 'A'; B_file = 'B'; C_file = 'C'; D_file = 'D';
savemat(pathout + A_file, mdict={A_file: np.array(A_sys)}, appendmat=True)
savemat(pathout + B_file, mdict={B_file: np.array(B_sys)}, appendmat=True)
savemat(pathout + C_file, mdict={C_file: np.array(C_sys)}, appendmat=True)
savemat(pathout + D_file, mdict={D_file: np.array(D_sys)}, appendmat=True)