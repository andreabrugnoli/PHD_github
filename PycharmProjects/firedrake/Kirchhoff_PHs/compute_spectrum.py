import numpy as np

def compute_spectrum(M, J, n_eig, filename = None):
    from slepc4py import SLEPc
    from petsc4py import PETSc

    petsc_M, petsc_J = PETSc.Mat().create(), PETSc.Mat().create()
    n, m = M.shape
    if n != m or (M.shape != J.shape):
        raise ValueError('Not square Matrices')

    petsc_M.setSizes([n, n])
    petsc_J.setSizes([n, n])
    petsc_M.setType("aij")
    petsc_J.setType("aij")

    petsc_M.setUp()
    petsc_J.setUp()

    # First arg is list of row indices, second list of column indices
    petsc_M.setValues(list(range(n)), list(range(n)), M)
    petsc_J.setValues(list(range(n)), list(range(n)), J)

    petsc_M.assemble()
    petsc_J.assemble()

    opts = PETSc.Options()
    opts.setValue("eps_gen_non_hermitian", None)
    opts.setValue("st_pc_factor_shift_type", "NONZERO")
    opts.setValue("eps_type", "krylovschur")
    opts.setValue("eps_smallest_imaginary", None)
    opts.setValue("eps_tol", 1e-4)

    eps = SLEPc.EPS().create()
    eps.setDimensions(n_eig)
    eps.setOperators(petsc_J, petsc_M)
    eps.setFromOptions()
    # Compute 10 eigenvalues with smallest magnitude

    # eps.setWhichEigenpairs(eps.Which.SMALLEST_MAGNITUDE)
    eps.solve()
    smallest = np.array([eps.getEigenvalue(i) for i in range(eps.getConverged())])
    print(smallest)
    # np.save(filename + '_smallest.npy', smallest)
    # # Compute 10 eigenvalues with largest magnitude
    # eps.setWhichEigenpairs(eps.Which.LARGEST_MAGNITUDE)
    # eps.solve()
    # largest = np.array([eps.getEigenvalue(i) for i in range(eps.getConverged())])
    # np.save(filename + '_largest.npy', largest)