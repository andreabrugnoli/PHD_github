# Mindlin plate written with the port Hamiltonian approach
# with weak symmetry
from ufl import indices
from fenics import *
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
import numpy as np


class Elasticity2DConfig:
    def __init__(self, n_el=1, deg_FE=1, Lx=1, Ly=1, rho=1, Eyoung=1, nu=0.3):
        """
        :param n_el: number of finite elements
        :param deg_FE: polynomial degree of the finite elements
        :param bd_cond: string representing the boundary condition for each side (D = dirichlet, N = Neumann).
                        The order for the sides is lower, right, upper, left
        :param Lx: length of the domain along x (m)
        :param Ly: legth of the domain along y (m)
        :param rho: mass density per unit area (kg/m^2)
        :param E: Young modulus (Pa)
        :param nu: Poisson ratio (dimensionless)
        :return: all parameters
        """
        assert type(n_el) is int, "n_el must be an integer"
        assert type(deg_FE) is int and deg_FE>=1, "deg_FE must be an integer>=1"
        # assert len(bd_cond) == 4, "bd_cond is a string of 4 char"
        # allowed_bd = "DN"
        # assert all(ch in allowed_bd for ch in bd_cond), "bd_cond must contain either D or N"
        assert Lx > 0, "Lx must be positive"
        assert Ly > 0, "Ly must be positive"
        assert rho > 0, "rho must be positive"
        assert Eyoung > 0, "E must be positive"
        assert -1 <= nu <= 0.5, "nu must be -1 <= nu <= 0.5"

        self.n_el = n_el
        self.deg_FE = deg_FE
        self.bd_cond = 'DDDD'
        self.Lx = Lx
        self.Ly = Ly
        self.rho = rho
        self.Eyoung = Eyoung
        self.nu = nu



def construct_system(Elasticity2DConfig):

    # Some useful functions

    def compliance_tensor(sigma):
        """
        Compliance tensor for 2D elasticity
        :param sigma: the stress tensor
        :return: the strain tensor
        """
        return (1 + Elasticity2DConfig.nu)/Elasticity2DConfig.Eyoung * \
               (sigma - Elasticity2DConfig.nu * Identity(2) * tr(sigma))

    msh = RectangleMesh(Point(0, 0), Point(Elasticity2DConfig.Lx, Elasticity2DConfig.Ly), \
                         Elasticity2DConfig.n_el, Elasticity2DConfig.n_el)

    n_ver = FacetNormal(msh)

    # # Domain, Subdomains, Boundary, Suboundaries (for different bcs only)
    # class Left(SubDomain):
    #     def inside(self, x, on_boundary):
    #         return abs(x[0] - 0.0) < DOLFIN_EPS and on_boundary
    #
    # class Right(SubDomain):
    #     def inside(self, x, on_boundary):
    #         return abs(x[0] - Elasticity2DConfig.Lx) < DOLFIN_EPS and on_boundary
    #
    # class Lower(SubDomain):
    #     def inside(self, x, on_boundary):
    #         return abs(x[1] - 0.0) < DOLFIN_EPS and on_boundary
    #
    # class Upper(SubDomain):
    #     def inside(self, x, on_boundary):
    #         return abs(x[1] - Elasticity2DConfig.Ly) < DOLFIN_EPS and on_boundary
    #
    # # Boundary conditions on rotations
    # left = Left()
    # right = Right()
    # lower = Lower()
    # upper = Upper()
    #
    # boundaries = MeshFunction("size_t", msh, msh.topology().dim() - 1)
    # boundaries.set_all(5)
    # lower.mark(boundaries, 1)
    # right.mark(boundaries, 2)
    # upper.mark(boundaries, 3)
    # left.mark(boundaries, 4)

    # bc_1, bc_2, bc_3, bc_4 = Elasticity2DConfig.bd_cond
    #
    # bc_dict = {lower: bc_1, right: bc_2, upper: bc_3, left: bc_4}
    #
    # bcs = []
    # for key, val in bc_dict.items():
    #     if val == 'N':
    #         bcs.append(DirichletBC(Vstate.sub(1), Constant((0.0, 0.0)), key))
    #         bcs.append(DirichletBC(Vstate.sub(2), Constant((0.0, 0.0)), key))

    # Finite element definition: we use the AFW with weak symmetry

    Pvel = VectorElement('DG', triangle, Elasticity2DConfig.deg_FE - 1)
    Psig1 = FiniteElement('BDM', triangle, Elasticity2DConfig.deg_FE)
    Psig2 = FiniteElement('BDM', triangle, Elasticity2DConfig.deg_FE)
    Pskw = FiniteElement('DG', triangle, Elasticity2DConfig.deg_FE - 1)

    AFW_elem = MixedElement([Pvel, Psig1, Psig2, Pskw])
    Vstate = FunctionSpace(msh, AFW_elem)

    Pcntr = VectorElement('CG', triangle, Elasticity2DConfig.deg_FE)
    Vcntr = FunctionSpace(msh, Pcntr)

    v_state = TestFunction(Vstate)
    v_vel, v_sig1, v_sig2, v_wsym = split(v_state)

    e_state = TrialFunction(Vstate)
    e_vel, e_sig1, e_sig2, e_wsym = split(e_state)

    u_cntr = TrialFunction(Vcntr)

    # The tensors have to be defined to follow the div convection of UFL
    v_sig = as_tensor([[v_sig1[0], v_sig1[1]],
                       [v_sig2[0], v_sig2[1]]
                       ])

    e_sig = as_tensor([[e_sig1[0], e_sig1[1]],
                       [e_sig2[0], e_sig2[1]]
                       ])

    v_skw = as_tensor([[0, v_wsym],
                       [-v_wsym, 0]])

    e_skw = as_tensor([[0, e_wsym],
                        [-e_wsym, 0]])

    dx = Measure('dx')
    ds = Measure('ds')

    def mass_form(v_vel, e_vel, v_sig, e_sig, v_skw, e_skw):
        """
        :param v_vel: test function for the velocity space
        :param e_vel: velocity field
        :param v_sig: test function for the sigma
        :param e_sig: stress tensor field
        :param v_skw: test function associated to the skew symmetric part of the stress tensor
        :param e_skw: skew-symmetric part of the stress tensor
        :return: the mass bilinear form
        """
        return Elasticity2DConfig.rho * dot(v_vel, e_vel) * dx \
                + inner(v_sig, compliance_tensor(e_sig)) * dx + inner(v_sig, e_skw) * dx \
                + inner(v_skw, e_sig) * dx

    def int_form(v_vel, e_vel, v_sig, e_sig):
        """
        :param v_vel: test function for the velocity space
        :param e_vel: velocity field
        :param v_sig: test function for the sigma
        :param e_sig: stress tensor field
        :return: the interconnection bilinear form
        """
        return dot(v_vel, div(e_sig)) * dx - dot(div(v_sig), e_vel) * dx

    def cntr_form(v_sig, u):
        i, j = indices(2)
        return v_sig[i, j]*n_ver[j]*u[i] * ds

    Jpetsc = PETScMatrix()
    Mpetsc = PETScMatrix()
    Bpetsc = PETScMatrix()

    m_form = mass_form(v_vel, e_vel, v_sig, e_sig, v_skw, e_skw)
    j_form = int_form(v_vel, e_vel, v_sig, e_sig)
    b_form = cntr_form(v_sig, u_cntr)

    assemble(m_form, Mpetsc)
    assemble(j_form, Jpetsc)
    assemble(b_form, Bpetsc)

    # Conversion to CSR scipy format
    Jscipy = csr_matrix(Jpetsc.mat().getValuesCSR()[::-1])
    Mscipy = csr_matrix(Mpetsc.mat().getValuesCSR()[::-1])
    Bscipy_cols = csr_matrix(Bpetsc.mat().getValuesCSR()[::-1])
    # Non zero rows and columns
    rows_B, cols_B = csr_matrix.nonzero(Bscipy_cols)

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

    # Indexes of non zero columns
    set_cols = np.array(list(set(cols_B)))
    # Number of non zero columns (i.e. number of inputs)
    n_u = len(set_cols)
    # Initialization of the final matrix in lil folmat for efficient incremental construction.
    Bscipy = lil_matrix((Vstate.dim(), n_u))
    for r, c in zip(rows_B, cols_B):
        # Column index in the final matrix
        ind_col = np.where(set_cols == c)[0]
        # Fill the matrix with the values
        Bscipy[r, ind_col] = Bscipy_cols[r, c]
        # Convert to csr format
        Bscipy.tocsr()

    return Mscipy, Jscipy, Bscipy


instance_El2D = Elasticity2DConfig()

M, J, B = construct_system(instance_El2D)

print(B.shape)
print(np.linalg.matrix_rank(B.todense()))


