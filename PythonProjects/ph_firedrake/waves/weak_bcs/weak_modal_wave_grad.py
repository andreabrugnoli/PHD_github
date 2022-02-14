## Weak imposition of the boundary coniditons via the Reissner-Hellinger method

import os

import numpy as np
from numpy import ndarray

os.environ["OMP_NUM_THREADS"] = "1"

# import numpy as np
from firedrake import *
import matplotlib.pyplot as plt
from tools_plotting import setup

from firedrake.petsc import PETSc
try:
    from slepc4py import SLEPc
except ImportError:
    import sys
    warning("Unable to import SLEPc, eigenvalue computation not possible (try firedrake-update --slepc)")
    sys.exit(0)

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
    mesh = RectangleMesh(n_el, n_el, Lx, Ly, quadrilateral=False)
    n_ver = FacetNormal(mesh)

    P_1 = FiniteElement("CG", triangle, deg)
    P_2 = FiniteElement("RT", triangle, deg, variant='integral')

    V_1 = FunctionSpace(mesh, P_1)
    V_2 = FunctionSpace(mesh, P_2)

    V = V_1 * V_2

    v = TestFunction(V)
    v_1, v_2 = split(v)

    e = TrialFunction(V)
    e_1, e_2 = split(e)

    dx = Measure('dx')
    ds = Measure('ds')

    x, y = SpatialCoordinate(mesh)

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

    J_weak = assemble(j_form, mat_type='aij')
    M_weak = assemble(m_form, mat_type='aij')
    petsc_j_wk = J_weak.M.handle
    petsc_m_wk = M_weak.M.handle

    bc1 = DirichletBC(V.sub(0), Constant(0.0), 1)
    bc2 = DirichletBC(V.sub(0), Constant(0.0), 2)

    bcs = [bc1, bc2]

    J_strong = assemble(j_form, bcs=bcs, mat_type='aij')
    M_strong = assemble(m_form, mat_type='aij')
    petsc_m_st = M_strong.M.handle
    petsc_j_st = J_strong.M.handle

    num_eigenvalues = 2*n_eig**2

    tol = 1e-11
    target = 1
    opts = PETSc.Options()
    opts.setValue("pos_gen_non_hermitian", None)
    # opts.setValue("pos_gen_hermitian", True)

    opts.setValue("st_pc_factor_shift_type", "NONZERO")
    opts.setValue("eps_type", "krylovschur")
    opts.setValue("eps_tol", tol)
    opts.setValue("st_type", "sinvert")
    # opts.setValue("eps_target_imaginary", None)
    opts.setValue("st_shift", target)
    opts.setValue("eps_target", target)

    es = SLEPc.EPS().create(comm=COMM_WORLD)
    es.setFromOptions()
    es.setDimensions(num_eigenvalues)
    # st = es.getST()
    # st.setShift(0)
    # st.setType("sinvert")
    # es.setST(st)
    # es.setWhichEigenpairs(1)

    tol_zero = 1e-16
    ## Compute numerical eigenvalues weak
    es.setOperators(petsc_j_wk, petsc_m_wk)
    es.solve()

    nconv_wk = es.getConverged()
    if nconv_wk == 0:
        import sys
        warning("Did not converge any eigenvalues (weak)")
        sys.exit(0)

    omega_num_pos_wk = []
    omega_num_zero_wk = []
    vr_wk, vi_wk = petsc_j_wk.getVecs()
    for i in range(nconv_wk):
        om_i_wk = es.getEigenpair(i, vr_wk, vi_wk)

        imag_om_i_wk = np.imag(om_i_wk)

        if imag_om_i_wk>tol:
            omega_num_pos_wk.append(imag_om_i_wk)
        elif np.abs(imag_om_i_wk)<tol_zero:
            omega_num_zero_wk.append(imag_om_i_wk)

    omega_num_pos_wk.sort()

    ## Compute numerical eigenvalues strong
    es.setOperators(petsc_j_st, petsc_m_st)
    es.solve()

    nconv_st = es.getConverged()
    if nconv_st == 0:
        import sys
        warning("Did not converge any eigenvalues (strong)")
        sys.exit(0)

    omega_num_pos_st = []
    omega_num_zero_st = []
    vr_st, vi_st = petsc_j_st.getVecs()
    for i in range(nconv_st):
        om_i_st = es.getEigenpair(i, vr_st, vi_st)

        imag_om_i_st = np.imag(om_i_st)

        if imag_om_i_st > tol:
            omega_num_pos_st.append(imag_om_i_st)
        elif np.abs(imag_om_i_st)<tol_zero:
            omega_num_zero_st.append(imag_om_i_st)

    omega_num_pos_st.sort()

    print("First 5 eigenvalues strong")
    print(omega_num_pos_st[:5])

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
    plt.savefig(path_fig + "Eigs_weak.pdf", format="pdf")

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
    plt.savefig(path_fig + "Eigs_strong.pdf", format="pdf")

plt.show()