from fenics import *

import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
np.set_printoptions(threshold=np.inf)

L = 1
n_el = 1
deg = 1

msh = RectangleMesh(Point(0, 0), Point(L, L), n_el, n_el)
n_ver = FacetNormal(msh)

# Domain, Subdomains, Boundary, Suboundaries (for different bcs only)
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - 0.0) < DOLFIN_EPS and on_boundary

class Right(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[0] - L) < DOLFIN_EPS and on_boundary

class Lower(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1] - 0.0) < DOLFIN_EPS and on_boundary

class Upper(SubDomain):
    def inside(self, x, on_boundary):
        return abs(x[1] - L) < DOLFIN_EPS and on_boundary

# Boundary conditions on rotations
left = Left()
right = Right()
lower = Lower()
upper = Upper()

boundaries = MeshFunction("size_t", msh, msh.topology().dim() - 1)
boundaries.set_all(5)
lower.mark(boundaries, 1)
right.mark(boundaries, 2)
upper.mark(boundaries, 3)
left.mark(boundaries, 4)

dx = Measure('dx')
ds = Measure('ds', subdomain_data=boundaries)

# P_0 = FiniteElement("CG", triangle, deg, variant='feec')
# P_1 = FiniteElement("N1curl", triangle, deg, variant='feec')
# P_1til = FiniteElement("RT", triangle, deg, variant='feec')
# P_2 = FiniteElement("DG", triangle, deg-1, variant='feec')

P_0 = FiniteElement("CG", triangle, deg)
P_1 = FiniteElement("N1curl", triangle, deg)
P_1til = FiniteElement("RT", triangle, deg+1)
P_2 = FiniteElement("DG", triangle, deg-1)

V_0 = FunctionSpace(msh, P_0)
V_1 = FunctionSpace(msh, P_1)
V_1til = FunctionSpace(msh, P_1til)
V_2 = FunctionSpace(msh, P_2)

v_0 = TestFunction(V_0)
u_0 = TrialFunction(V_0)

v_1 = TestFunction(V_1)
u_1 = TrialFunction(V_1)

v_1til = TestFunction(V_1til)
u_1til = TrialFunction(V_1til)

v_2 = TestFunction(V_2)
u_2 = TrialFunction(V_2)

def cntr_form_N(v_0, u_1til):
    return v_0 * dot(u_1til, n_ver) * ds

def cntr_form_D(v_1til, u_0):
    return dot(v_1til, n_ver) * u_0 * ds

def remove_cols(Bscipy_cols, rows_B, cols_B):
    n_V, n_col_full = Bscipy_cols.shape

    tol = 1e-6
    indtol_vec = []
    # Remove nonzeros rows and columns below a given tolerance
    for kk in range(len(rows_B)):
        ind_row = rows_B[kk]
        ind_col = cols_B[kk]

        if abs(Bscipy_cols[ind_row, ind_col]) < tol:
            indtol_vec.append(kk)

    rows_B = np.delete(rows_B, indtol_vec)
    cols_B = np.delete(cols_B, indtol_vec)


    set_cols = np.array(list(set(cols_B)))
    # Number of non zero columns (i.e. number of inputs)
    n_u = len(set(cols_B))
    # Initialization of the final matrix in lil folmat for efficient incremental construction.
    B_scipy = lil_matrix((n_V, n_u))
    for r, c in zip(rows_B, cols_B):
        # Column index in the final matrix
        ind_col = np.where(set_cols == c)[0]
        # Fill the matrix with the values
        B_scipy[r, ind_col] = Bscipy_cols[r, c]
        # Convert to csr format
        B_scipy.tocsr()

    return B_scipy

B_D_petsc = PETScMatrix()
B_N_petsc = PETScMatrix()

b_D_form = cntr_form_D(v_1til, u_0)
b_N_form = cntr_form_N(v_0, u_1til)

assemble(b_D_form, B_D_petsc)
assemble(b_N_form, B_N_petsc)

B_D_scipy_cols = csr_matrix(B_D_petsc.mat().getValuesCSR()[::-1])
# Non zero rows and columns
rows_B_D, cols_B_D = csr_matrix.nonzero(B_D_scipy_cols)

B_D_scipy = remove_cols(B_D_scipy_cols, rows_B_D, cols_B_D)

print("B_D matrix rank")
print(B_D_scipy.shape)
print(np.linalg.matrix_rank(B_D_scipy.todense()))

B_N_scipy_cols = csr_matrix(B_N_petsc.mat().getValuesCSR()[::-1])
# Non zero rows and columns
rows_B_N, cols_B_N = csr_matrix.nonzero(B_N_scipy_cols)

B_N_scipy = remove_cols(B_N_scipy_cols, rows_B_N, cols_B_N)

print("B_N matrix rank")
print(B_N_scipy.todense())
print(np.linalg.matrix_rank(B_N_scipy.todense()))