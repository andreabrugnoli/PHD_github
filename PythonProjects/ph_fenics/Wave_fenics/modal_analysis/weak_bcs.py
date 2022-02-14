## Weak imposition of the boundary coniditons via the Reissner-Hellinger method

import os

import numpy as np
from numpy import ndarray

os.environ["OMP_NUM_THREADS"] = "1"

# import numpy as np
from fenics import *
import matplotlib.pyplot as plt
from tools_plotting import setup


save_plots = input("Save plots? ")

def compute_eig(n_el, n_eig, deg=1):
    """Compute the numerical solution of the wave equation with the dual field method

        Parameters:
        n_el: number of elements for the discretization
        n_t: number of time instants
        deg: polynomial degree for finite
        Returns:
        some plots

       """

    Lx = pi
    Ly = pi
    mesh = RectangleMesh(Point(0, 0), Point(Lx, Ly), n_el, n_el)
    n_ver = FacetNormal(mesh)

    class Left(SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[0] - 0.0) < DOLFIN_EPS and on_boundary

    class Right(SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[0] - Lx) < DOLFIN_EPS and on_boundary

    class Lower(SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[1] - 0.0) < DOLFIN_EPS and on_boundary

    class Upper(SubDomain):
        def inside(self, x, on_boundary):
            return abs(x[1] - Ly) < DOLFIN_EPS and on_boundary

    # Boundary conditions on rotations
    left = Left()
    right = Right()
    lower = Lower()
    upper = Upper()

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(5)
    left.mark(boundaries, 1)
    right.mark(boundaries, 2)
    lower.mark(boundaries, 3)
    upper.mark(boundaries, 4)


    # Finite element defition

    P_1 = FiniteElement("CG", triangle, deg)
    P_2 = FiniteElement("RT", triangle, deg, variant='integral')

    element = MixedElement([P_1, P_2])
    V = FunctionSpace(mesh, element)


    v = TestFunction(V)
    v_1, v_2 = split(v)

    e = TrialFunction(V)
    e_1, e_2 = split(e)

    dx = Measure('dx')
    ds = Measure('ds')

    ## Exact eigenvalues

    omega_ex_vec = np.empty((n_eig**2))
    k = 0
    for m in range(n_eig):
        for n in range(1,n_eig+1):
            lambda_n = n*pi/Lx
            mu_m = m*pi/Ly

            omega_ex_vec[k] = np.sqrt(lambda_n**2 + mu_m**2)
            k = k+1

    omega_ex_vec.sort()

    ## Bilinear forms
    j_form = dot(v_2, grad(e_1)) * dx - dot(grad(v_1), e_2) * dx \
            + v_1 * dot(e_2, n_ver) * ds(1) + v_1 * dot(e_2, n_ver) * ds(2) \
            - e_1 * dot(v_2, n_ver) * ds(1) - e_1 * dot(v_2, n_ver) * ds(2)

    m_form = inner(v_1, e_1) * dx + inner(v_2, e_2) * dx


    J_weak = PETScMatrix()
    assemble(j_form, tensor=J_weak)

    M_weak = PETScMatrix()
    assemble(m_form, tensor=M_weak)

    bc1 = DirichletBC(V.sub(0), Constant(0.0), left)
    bc2 = DirichletBC(V.sub(0), Constant(0.0), right)

    bcs = [bc1, bc2]

    j_form_st = dot(v_2, grad(e_1)) * dx - dot(grad(v_1), e_2) * dx

    l_form = Constant(1.) * v_1 * dx
    J_strong = PETScMatrix()

    assemble_system(j_form_st, l_form, bcs, A_tensor=J_strong)

    M_strong = PETScMatrix()
    assemble_system(m_form, l_form, bcs, A_tensor=M_strong)
    [bc.zero(M_strong) for bc in bcs]

    num_eigenvalues = 2*n_eig**2

    tol = 1e-11
    tol_zero = 1e-12
    solver_wk = SLEPcEigenSolver(J_weak, M_weak)
    solver_wk.parameters["solver"] = "krylov-schur"
    solver_wk.parameters["problem_type"] = "pos_gen_non_hermitian"
    solver_wk.parameters['spectral_transform'] = 'shift-and-invert'
    solver_wk.parameters["spectrum"] = "target imaginary"
    solver_wk.parameters['spectral_shift'] = 1.

    solver_wk.solve(num_eigenvalues)
    nconv_wk = solver_wk.get_number_converged()


    omega_num_pos_wk = []
    omega_num_zero_wk = []
    for i in range(nconv_wk):
        lam_r, lam_i, psi_r, psi_i = solver_wk.get_eigenpair(i)


        if lam_i>tol:
            omega_num_pos_wk.append(lam_i)
        elif np.abs(lam_i)<tol_zero:
            omega_num_zero_wk.append(lam_i)

    omega_num_pos_wk.sort()

    ## Compute numerical eigenvalues strong

    solver_st = SLEPcEigenSolver(J_strong, M_strong)
    solver_st.parameters["solver"] = "krylov-schur"
    solver_st.parameters["problem_type"] = "pos_gen_non_hermitian"
    solver_st.parameters['spectral_transform'] = 'shift-and-invert'
    solver_st.parameters["spectrum"] = "target imaginary"
    solver_st.parameters['spectral_shift'] = 1.

    solver_st.solve(num_eigenvalues)
    nconv_st = solver_st.get_number_converged()

    omega_num_pos_st = []
    omega_num_zero_st = []
    for i in range(nconv_st):
        lam_r, lam_i, psi_r, psi_i = solver_st.get_eigenpair(i)


        if lam_i > tol:
            omega_num_pos_st.append(lam_i)
        elif np.abs(lam_i)<tol_zero:
            omega_num_zero_st.append(lam_i)

    omega_num_pos_st.sort()

    print("First 5 eigenvalues weak")
    print(omega_num_pos_wk[:20])

    return omega_num_pos_wk, omega_num_zero_wk, omega_num_pos_st, omega_num_zero_st, omega_ex_vec

n_elem = 5
n_eig = 50
n_eigs_plot = 50

num_eigs_pos_wk_deg1, num_eigs_zero_wk_deg1, num_eigs_pos_st_deg1, num_eigs_zero_st_deg1, ex_eigs = compute_eig(n_elem, n_eig, 1)
print("Degree 1 completed")
num_eigs_pos_wk_deg2, num_eigs_zero_wk_deg2, num_eigs_pos_st_deg2, num_eigs_zero_st_deg2, ex_eigs = compute_eig(n_elem, n_eig, 2)
print("Degree 2 completed")
num_eigs_pos_wk_deg3, num_eigs_zero_wk_deg3, num_eigs_pos_st_deg3, num_eigs_zero_st_deg3, ex_eigs = compute_eig(n_elem, n_eig, 3)
print("Degree 3 completed")
num_eigs_pos_wk_deg4, num_eigs_zero_wk_deg4, num_eigs_pos_st_deg4, num_eigs_zero_st_deg4, ex_eigs = compute_eig(n_elem, n_eig, 4)
print("Degree 4 completed")
# num_eigs_pos_wk_deg5, num_eigs_zero_wk_deg5, num_eigs_pos_st_deg5, num_eigs_zero_st_deg5, ex_eigs = compute_eig(n_elem, n_eig, 5)
# print("Degree 5 completed")

print("Zero eigenvalues deg = 1")
print("weak: "+ str(len(num_eigs_zero_wk_deg1)))
print("strong: "+ str(len(num_eigs_zero_st_deg1)))

print("Zero eigenvalues deg = 2")
print("weak: "+ str(len(num_eigs_zero_wk_deg2)))
print("strong: "+ str(len(num_eigs_zero_st_deg2)))

print("Zero eigenvalues deg = 3")
print("weak: "+ str(len(num_eigs_zero_wk_deg3)))
print("strong: "+ str(len(num_eigs_zero_st_deg3)))

print("Zero eigenvalues deg = 4")
print("weak: " + str(len(num_eigs_zero_wk_deg4)))
print("strong: " + str(len(num_eigs_zero_st_deg4)))

plt.figure()

plt.plot(num_eigs_pos_wk_deg1[:n_eigs_plot], '<', mfc='none', label=r'$r=1$')
plt.plot(num_eigs_pos_wk_deg2[:n_eigs_plot], '>', mfc='none', label=r'$r=2$')
plt.plot(num_eigs_pos_wk_deg3[:n_eigs_plot], '^', mfc='none', label=r'$r=3$')
plt.plot(num_eigs_pos_wk_deg4[:n_eigs_plot], 'v', mfc='none', label=r'$r=4$')

plt.plot(ex_eigs[:n_eigs_plot], '+', label=r'Exact eigs')

plt.xlabel(r'N$^\circ$ eigenvalue')
plt.title(r'Eigenvalues (weak bcs)')
plt.legend()

path_fig="/home/andrea/Pictures/PythonPlots/MTNS22/"

if save_plots:
    plt.savefig(path_fig + "Eigs_weak_fenics.pdf", format="pdf")

plt.figure()

plt.plot(num_eigs_pos_st_deg1[:n_eigs_plot], '<', mfc='none', label=r'$r=1$')
plt.plot(num_eigs_pos_st_deg2[:n_eigs_plot], '>', mfc='none', label=r'$r=2$')
plt.plot(num_eigs_pos_st_deg3[:n_eigs_plot], '^', mfc='none', label=r'$r=3$')
plt.plot(num_eigs_pos_st_deg4[:n_eigs_plot], 'v', mfc='none', label=r'$r=4$')

plt.plot(ex_eigs[:n_eigs_plot], '+', label=r'Exact eigs')

plt.xlabel(r'N$^\circ$ eigenvalue')
plt.title(r'Eigenvalues (strong bcs)')
plt.legend()

if save_plots:
    plt.savefig(path_fig + "Eigs_strong_fenics.pdf", format="pdf")

plt.show()