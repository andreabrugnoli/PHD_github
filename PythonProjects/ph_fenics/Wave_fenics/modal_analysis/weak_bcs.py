## Weak imposition of the boundary coniditons via the Reissner-Hellinger method

import os

import numpy as np
from numpy import ndarray

os.environ["OMP_NUM_THREADS"] = "1"

# import numpy as np
from fenics import *
import matplotlib.pyplot as plt
from tools_plotting import setup
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigs


save_plots = input("Save plots? ")

def compute_eig(n_el, n_eig, deg=1, solver="SLEPc"):
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
    boundaries.set_all(0)
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
    ds = Measure('ds', subdomain_data=boundaries)

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

    if solver=="SLEPc":
        ## Compute numerical eigenvalues weak
        omega_num_pos_wk, omega_num_zero_wk = compute_eigs_SLEPc(J_weak, M_weak, num_eigenvalues)
        ## Compute numerical eigenvalues strong
        omega_num_pos_st, omega_num_zero_st = compute_eigs_SLEPc(J_strong, M_strong, num_eigenvalues)

    else:
        ## Compute numerical eigenvalues weak
        omega_num_pos_wk, omega_num_zero_wk = compute_eigs_Scipy(J_weak, M_weak, num_eigenvalues)
        ## Compute numerical eigenvalues strong
        omega_num_pos_st, omega_num_zero_st = compute_eigs_Scipy(J_strong, M_strong, num_eigenvalues)

    print("First 5 eigenvalues weak")
    print(omega_num_pos_wk[:20])

    print("First 5 eigenvalues strong")
    print(omega_num_pos_st[:20])

    return omega_num_pos_wk, omega_num_zero_wk, omega_num_pos_st, omega_num_zero_st, omega_ex_vec

def compute_eigs_SLEPc(J, M, n_eig_solver=20, tol_zero=1e-12):
    solver = SLEPcEigenSolver(J, M)
    solver.parameters["solver"] = "krylov-schur"
    solver.parameters["problem_type"] = "pos_gen_non_hermitian"
    solver.parameters['spectral_transform'] = 'shift-and-invert'
    solver.parameters["spectrum"] = "target imaginary"
    solver.parameters['spectral_shift'] = 1.

    solver.solve(n_eig_solver)
    nconv = solver.get_number_converged()

    omega_num_pos = []
    omega_num_zero = []
    for i in range(nconv):
        lam_r, lam_i, psi_r, psi_i = solver.get_eigenpair(i)

        if lam_i > tol_zero:
            omega_num_pos.append(lam_i)
        elif np.abs(lam_i) < tol_zero:
            omega_num_zero.append(lam_i)

    omega_num_pos.sort()

    return omega_num_pos, omega_num_zero


def compute_eigs_Scipy(J, M, n_eig_scipy=20, tol=1e-6, tol_zero=1e-12):
    J_scipy = csr_matrix(J.mat().getValuesCSR()[::-1])  # for fenics J.mat()
    M_scipy = csr_matrix(M.mat().getValuesCSR()[::-1])  # for fenics M.mat()
   # Shift value
    shift = 1



    if n_eig_scipy>=(M_scipy.shape[0]-1):
        k_scipy = M_scipy.shape[0]-2
    else: k_scipy=n_eig_scipy
    # Resolution of the eigenproblem

    eigenvalues, eigvectors = eigs(J_scipy, k=k_scipy, M=M_scipy, \
                                   sigma=shift, which='LM', tol=tol)

    omega_num_pos = []
    omega_num_zero = []
    for i in range(len(eigenvalues)):

        lam_i = np.imag(eigenvalues[i])
        if lam_i > tol_zero:
            omega_num_pos.append(lam_i)
        elif np.abs(lam_i) < tol_zero:
            omega_num_zero.append(lam_i)

    omega_num_pos.sort()

    return omega_num_pos, omega_num_zero

n_elem = 5
n_eig = 50
n_eigs_plot = 50

num_eigs_pos_wk_deg1, num_eigs_zero_wk_deg1, num_eigs_pos_st_deg1, num_eigs_zero_st_deg1, ex_eigs = compute_eig(n_elem, n_eig, 1)
# num_eigs_pos_wk_deg1, num_eigs_zero_wk_deg1, num_eigs_pos_st_deg1, num_eigs_zero_st_deg1, ex_eigs = compute_eig(n_elem, n_eig, 1, solver="Scipy")
print("Degree 1 completed")
num_eigs_pos_wk_deg2, num_eigs_zero_wk_deg2, num_eigs_pos_st_deg2, num_eigs_zero_st_deg2, ex_eigs = compute_eig(n_elem, n_eig, 2)
# num_eigs_pos_wk_deg2, num_eigs_zero_wk_deg2, num_eigs_pos_st_deg2, num_eigs_zero_st_deg2, ex_eigs = compute_eig(n_elem, n_eig, 2, solver="Scipy")
print("Degree 2 completed")
num_eigs_pos_wk_deg3, num_eigs_zero_wk_deg3, num_eigs_pos_st_deg3, num_eigs_zero_st_deg3, ex_eigs = compute_eig(n_elem, n_eig, 3)
# print("Degree 3 completed")
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