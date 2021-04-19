from fenics import *
import numpy as np


# from scipy import linalg as la

def matrices_wave(n_el=10, deg=1, e0_string=('0', '0'), eT_string=('0', '0'), \
                        rho=1, E=1, A=1, L=1, r_damp=1):
    """
    Computes matrices M, J, B for the Timoshenko beam.
    Parameters:
    n_el: number of finite elements
    deg: degree of  palynomial basis functions
    rho: density per unit volume [kg/m^3]
    E: Young modulus
    A: Cross section Area
    L: beam length
    """

    # Mesh
    mesh = IntervalMesh(n_el, 0, L)
    d = mesh.geometry().dim()

    class AllBoundary(SubDomain):
        """
        Class for defining the two boundaries
        """

        def inside(self, x, on_boundary):
            return on_boundary

    class Left(SubDomain):
        """
        Class for defining the left boundary
        """

        def inside(self, x, on_boundary):
            return abs(x[0] - 0.0) < DOLFIN_EPS and on_boundary

    class Right(SubDomain):
        """
        Class for defining the right boundary
        """

        def inside(self, x, on_boundary):
            return abs(x[0] - L) < DOLFIN_EPS and on_boundary

    # Boundary conditions on displacement
    all_boundary = AllBoundary()
    # Boundary conditions on rotations
    left = Left()
    right = Right()

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    left.mark(boundaries, 1)
    right.mark(boundaries, 2)

    # Measures for the evaluation of forms on the domain
    dx = Measure('dx')
    # Measure for evaluating boundary forms
    ds = Measure('ds', subdomain_data=boundaries)

    # Finite elements defition

    P_p = FiniteElement('CG', mesh.ufl_cell(), deg)
    P_q = FiniteElement('DG', mesh.ufl_cell(), deg - 1)

    # Our problem is defined on a mixed function space
    element = MixedElement([P_p, P_q])

    V = FunctionSpace(mesh, element)

    v = TestFunction(V)
    v_p, v_q = split(v)

    e = TrialFunction(V)
    e_p, e_q = split(e)

    exp_e0 = Expression(e0_string, degree=2)
    e0_array = interpolate(exp_e0, V).vector().get_local()

    exp_eT = Expression(eT_string, degree=2)
    eT_array = interpolate(exp_eT, V).vector().get_local()

    def get_m_form(v_p, v_q, e_p, e_q):
        """
        Defines the mass form. Once assembled the mass matrix is obtained
        """
        # Energy variables
        al_p = rho * A * e_p
        al_q = 1/(E*A) * e_q

        m = v_p * al_p * dx \
            + v_q * al_q * dx

        return m

    def get_j_form(v_p, v_q, e_p, e_q):
        """
        Defines the interconnection form.
        Once assembled the interconnection matrix is obtained
        """

        j_grad = v_q * e_p.dx(0) * dx
        j_gradIP = -v_p.dx(0) * e_q * dx

        j = j_grad + j_gradIP

        return j

    def get_r_form(v_p, e_p):
        """
        Defines the damping form.
        Once assembled the damping matrix is obtained
        """
        r_form = r_damp * v_p * e_p * dx

        return r_form

    # Boundary conditions
    bcs = []

    bc_p = DirichletBC(V.sub(0), Constant(0.0), left)

    bcs.append(bc_p)

    dofs_bc = []
    for bc in bcs:
        for key in bc.get_boundary_values().keys():
            dofs_bc.append(key)

    # Matrices assembly
    m_form = get_m_form(v_p, v_q, e_p, e_q)
    j_form = get_j_form(v_p, v_q, e_p, e_q)
    r_form = get_r_form(v_p, e_p)

    J_petsc = assemble(j_form)
    M_petsc = assemble(m_form)
    R_petsc = assemble(r_form)

    J_mat = J_petsc.array()
    M_mat = M_petsc.array()
    R_mat = R_petsc.array()

    B_vec = assemble(v_p * ds(2)).get_local()

    # Dofs and coordinates for subspaces
    dofs2x = V.tabulate_dof_coordinates().reshape((-1, d))

    dofsVp = V.sub(0).dofmap().dofs()
    dofsVq = V.sub(1).dofmap().dofs()

    xVp = dofs2x[dofsVp, 0]
    xVq = dofs2x[dofsVq, 0]

    i_max_Vq = np.argmax(xVq)

    dof_last_Vq = dofsVq[i_max_Vq]

    dofs2x_bc = np.delete(dofs2x, dofs_bc)

    dofsVp_bc = dofsVp.copy()
    dofsVq_bc = dofsVq.copy()

    # Eliminate bc dofs form subspaces
    for bc_dof in dofs_bc:

        dofsVp_bc.remove(bc_dof)

    # Recompute the dofs after bc elimination
    for (i_bc, dof_bc) in enumerate(dofs_bc):
        # Correction for last Vq
        if dof_last_Vq > dof_bc - i_bc:
            dof_last_Vq += -1

        # Correction of dofs for Vp
        for (ind, dof) in enumerate(dofsVp_bc):
            if dof > dof_bc - i_bc:
                dofsVp_bc[ind] += -1
        # Correction of dofs for Vq
        for (ind, dof) in enumerate(dofsVq_bc):
            if dof > dof_bc - i_bc:
                dofsVq_bc[ind] += -1

    xVp_bc = dofs2x_bc[dofsVp_bc]
    xVq_bc = dofs2x_bc[dofsVq_bc]

    # Dictonary containing the dofs
    dofs_dict = {'v': dofsVp_bc, 'sig': dofsVq_bc}

    # Dictonary containing the coordinates
    x_dict = {'v': xVp_bc, 'sig': xVq_bc}

    # Eliminate bc dofs from matrices and vectors
    M_red = np.delete(np.delete(M_mat, dofs_bc, axis=0), dofs_bc, axis=1)
    J_red = np.delete(np.delete(J_mat, dofs_bc, axis=0), dofs_bc, axis=1)
    R_red = np.delete(np.delete(R_mat, dofs_bc, axis=0), dofs_bc, axis=1)

    B_red = np.delete(B_vec, dofs_bc, axis=0)
    e0_red = np.delete(e0_array, dofs_bc, axis=0)
    eT_red = np.delete(eT_array, dofs_bc, axis=0)

    return M_red, J_red, R_red, B_red, e0_red, eT_red,\
           dofs_dict, x_dict, dof_last_Vq








