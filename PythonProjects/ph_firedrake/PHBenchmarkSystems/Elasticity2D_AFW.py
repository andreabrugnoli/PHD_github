# Mindlin plate written with the port Hamiltonian approach
# with weak symmetry
from ufl import indices
from firedrake import *
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
from scipy.io import savemat
import numpy as np
import matplotlib.pyplot as plt

class Elasticity2DConfig:
    def __init__(self, n_el=1, deg_FE=1, Lx=1, Ly=1, rho=1, lamda=20, mu=4):
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
        assert Lx > 0, "Lx must be positive"
        assert Ly > 0, "Ly must be positive"
        assert rho > 0, "rho must be positive"
        assert lamda > 0, "E must be positive"
        assert mu > 0, "mu must be positive"

        self.n_el = n_el
        self.deg_FE = deg_FE
        self.Lx = Lx
        self.Ly = Ly
        self.rho = rho
        self.lamda = lamda
        self.mu = mu



def construct_system(Elasticity2DConfig):

    # Some useful functions

    def compliance_tensor(sigma):
        """
        Compliance tensor for 2D elasticity
        :param sigma: the stress tensor
        :return: the strain tensor
        """
        return 1/(2*Elasticity2DConfig.mu) * \
               (sigma - Elasticity2DConfig.lamda/(2*(Elasticity2DConfig.lamda + Elasticity2DConfig.mu)) * Identity(2) * tr(sigma))

    msh = RectangleMesh(Elasticity2DConfig.n_el, Elasticity2DConfig.n_el, \
                        Elasticity2DConfig.Lx, Elasticity2DConfig.Ly )

    n_ver = FacetNormal(msh)

    x, y = SpatialCoordinate(msh)
    # location_f = conditional(And(And(gt(x, Elasticity2DConfig.Lx/4), lt(x, 3 * Elasticity2DConfig.Lx/4)), \
    #                              And(gt(y, Elasticity2DConfig.Ly/4), lt(y, 3 * Elasticity2DConfig.Ly/4))), 1, 0)
    location_f = exp(-0.5*(100*(x-Elasticity2DConfig.Lx/2)**2 + 100*(y-Elasticity2DConfig.Ly/2)**2))
    # trisurf(interpolate(location_f, FunctionSpace(msh, "CG", 2)))
    # plt.show()

    # Finite element definition: we use the AFW with weak symmetry

    Pvel = VectorElement('DG', triangle, Elasticity2DConfig.deg_FE - 1)
    Psig1 = FiniteElement('BDM', triangle, Elasticity2DConfig.deg_FE, variant="integral")
    Psig2 = FiniteElement('BDM', triangle, Elasticity2DConfig.deg_FE, variant="integral")
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
                + inner(v_sig, compliance_tensor(e_sig)) * dx \
                + inner(v_sig, e_skw) * dx \
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

    def cntr_form_x(v_vel):
        return v_vel[0] * location_f * ds

    def cntr_form_y(v_vel):
        return v_vel[1] * location_f * ds

    m_form = mass_form(v_vel, e_vel, v_sig, e_sig, v_skw, e_skw)
    j_form = int_form(v_vel, e_vel, v_sig, e_sig)

    b_form_x = cntr_form_x(v_vel)
    b_form_y = cntr_form_y(v_vel)

    Mpetsc = assemble(m_form, mat_type='aij').M.handle  # Petsc format of the matrix
    Jpetsc = assemble(j_form, mat_type='aij').M.handle  # Petsc format of the matrix


    # Conversion to CSR scipy format
    Jscipy = csr_matrix(Jpetsc.getValuesCSR()[::-1])
    Mscipy = csr_matrix(Mpetsc.getValuesCSR()[::-1])

    Bscipy_x = assemble(b_form_x).vector().get_local()
    Bscipy_y = assemble(b_form_y).vector().get_local()

    Bscipy = np.hstack((Bscipy_x.reshape((-1, 1)), Bscipy_y.reshape((-1, 1))))
    return Mscipy, Jscipy, Bscipy


# instance_El2D_case1 = Elasticity2DConfig(n_el=5, deg_FE=2)
# E_1, J_1, B_1 = construct_system(instance_El2D_case1)
#
# n1 = E_1.shape[0]
# dic_case1 = {"E": E_1.todense(), "J": J_1.todense(), "B": B_1}
# savemat("/home/andrea/Data/PH_Benchmark/el2Dafw-n" + str(n1) + ".mat", dic_case1)

# instance_El2D_case2 = Elasticity2DConfig(n_el=10, deg_FE=1)
# E_2, J_2, B_2 = construct_system(instance_El2D_case2)
#
# # A_2 = spsolve(E_2.tocsc(), J_2.tocsc())
#
# n2 = E_2.shape[0]
#
# dic_case2 = {"E": E_2, "J": J_2, "B": B_2}
#
# savemat("/home/andrea/Data/PH_Benchmark/el2Dafw-n" + str(n2) + ".mat", dic_case2)
#
# instance_El2D_case3 = Elasticity2DConfig(n_el=10, deg_FE=2)
# E_3, J_3, B_3 = construct_system(instance_El2D_case3)
#
# n3 = E_3.shape[0]
#
# dic_case3 = {"E": E_3, "J": J_3, "B": B_3}
#
# savemat("/home/andrea/Data/PH_Benchmark/el2Dafw-n" + str(n3) + ".mat", dic_case3)
#

