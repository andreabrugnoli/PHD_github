from firedrake import *
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse import vstack, hstack, block_diag
from scipy.sparse.linalg import eigs

n_el  = 40
mesh  = UnitSquareMesh(n_el, n_el)
V_w   = FunctionSpace(mesh, "CG", 1)
V_th  = VectorFunctionSpace(mesh, "CG", 1)
V_kap = VectorFunctionSpace(mesh, "DG", 0, dim=3)
V_gam = VectorFunctionSpace(mesh, "DG", 0)
V     = MixedFunctionSpace([V_w, V_th, V_kap, V_gam])

# Physical parameters
E = 1e12
nu = 0.3
rho = 2600
h = 0.1
k = 5 / 6
G = E / 2 / (1 + nu)
F = G * h * k


# Definition of the bending curvature operator
def bending_curv(mom):
    kappa = 12. / (E * h ** 3) * ((1 + nu) * mom - nu * Identity(2) * tr(mom))
    return kappa


# Test variables
v = TestFunction(V)
v_w, v_th, v_kap, v_gam = split(v)

# Co-energy variables
e = TrialFunction(V)
e_w, e_th, e_kap, e_gam = split(e)

# Convert the R^3 vector to a symmetric tensor
v_kap = as_tensor([[v_kap[0], v_kap[1]],
                   [v_kap[1], v_kap[2]]])
e_kap = as_tensor([[e_kap[0], e_kap[1]],
                   [e_kap[1], e_kap[2]]])

# Energy variables
a_w = rho * h * e_w
a_th = (rho * h ** 3) / 12. * e_th
a_kap = bending_curv(e_kap)
a_gam = 1. / F * e_gam

# Mass bilinear form
m_form = v_w * a_w * dx + dot(v_th, a_th) * dx + \
         inner(v_kap, a_kap) * dx + dot(v_gam, a_gam) * dx

# Interconnection bilinear form
j_form = dot(v_gam, grad(e_w)) * dx - dot(grad(v_w), e_gam) * dx + \
         inner(v_kap, sym(grad(e_th))) * dx - \
         inner(sym(grad(v_th)), e_kap) * dx + \
         dot(v_th, e_gam) * dx - dot(v_gam, e_th) * dx

bcs = []
# bcs.append(DirichletBC(V.sub(0), Constant(0.0), "on_boundary"))
# bcs.append(DirichletBC(V.sub(1), Constant((0.0, 0.0)), "on_boundary"))

J_ass = assemble(j_form, bcs=bcs, mat_type='aij')
M_ass = assemble(m_form, bcs=bcs, mat_type='aij')
J = J_ass.M.handle
M = M_ass.M.handle
# Check for SLEPc
# from firedrake.petsc import PETSc
#
# try:
#     from slepc4py import SLEPc
# except ImportError:
#     import sys
#
#     warning("Unable to import SLEPc (try firedrake-update --slepc)")
#     sys.exit(0)
#
# # Options for the solver.
# opts = PETSc.Options()
# opts.setValue("pos_gen_non_hermitian", None)
# opts.setValue("st_pc_factor_shift_type", "NONZERO")
# opts.setValue("eps_type", "krylovschur")
# opts.setValue("st_type", "sinvert")
# opts.setValue("st_shift", 1 / (((2 * (1 + nu) * rho) / E) ** 0.5))
# opts.setValue("eps_target", 1 / (((2 * (1 + nu) * rho) / E) ** 0.5))
#
# # Construction of the eigensolver.
# es = SLEPc.EPS().create(comm=COMM_WORLD)
# es.setDimensions(40)
# es.setOperators(J, M)
# es.setFromOptions()
# es.solve()
#
# n_conv = es.getConverged()
# psi_r, psi_i = J.getVecs()
# omega_tilde = []
# for i in range(n_conv):
#     lam = es.getEigenpair(i, psi_r, psi_i)
#     lam_i = np.imag(lam)
#
#     if lam_i > 1e-5:
#         omega_tilde.append(lam_i * ((2 * (1 + nu) * rho) / E) ** 0.5)
#
# print(omega_tilde)

V_qn  = FunctionSpace(mesh, 'CG', 1)
V_Mnn = FunctionSpace(mesh, 'CG', 1)
V_Mns = FunctionSpace(mesh, 'CG', 1)

V_u = V_qn * V_Mnn * V_Mns
q_n, M_nn, M_ns = TrialFunction(V_u)

n_ver = FacetNormal(mesh)  # outward normal to the boundary
s_ver = as_vector([-n_ver[1], n_ver[0]])  # tangent versor to the boundary

b_form = v_w * q_n * ds + dot(v_th, n_ver) * M_nn * ds + \
		 dot(v_th, s_ver) * M_ns * ds
B_ass = assemble(b_form, mat_type='aij')
B = B_ass.M.handle

# Conversion to CSR scipy format
B_scipy_cols = csr_matrix(B.getValuesCSR()[::-1])

# Non zero rows and columns
rows, cols = csr_matrix.nonzero(B_scipy_cols)

# Indexes of non zero columns
set_cols = np.array(list(set(cols)))

# Number of non zero columns (i.e. input number)
n_u = len(set(cols))

# Initialization of the final matrix in lil folmat
# for efficient incremental construction.
B_scipy = lil_matrix((V.dim(), n_u))

for r, c in zip(rows, cols):
    # Column index in the final matrix
    ind_col = np.where(set_cols == c)[0]

    # Fill the matrix with the values
    B_scipy[r, ind_col] = B_scipy_cols[r, c]

# Convert to csr format
B_scipy.tocsr()

# Conversion to scipy CSR matrices
# Important: no boundary conditions imposed
J_scipy = csr_matrix(J.getValuesCSR()[::-1])  # for fenics J.mat()
M_scipy = csr_matrix(M.getValuesCSR()[::-1])  # for fenics M.mat()

Z_mat = csr_matrix((n_u, n_u))
J_aug = vstack((hstack((J_scipy, B_scipy)), hstack((-B_scipy.T, Z_mat))))
M_aug = block_diag((M_scipy, Z_mat))

# Shift value
shift = 1/(((2*(1+nu)*rho)/E)**0.5)
eigenvalues, eigvectors = eigs(J_aug, k=40, M=M_aug,\
	 sigma=shift, which='LM', tol=1e-6, maxiter=5000)

omega_all = np.imag(eigenvalues)
index = omega_all >= 1e-5
omega = omega_all[index]
omega.sort()

omega_tilde = omega*((2*(1+nu)*rho)/E)**0.5

print(omega_tilde)