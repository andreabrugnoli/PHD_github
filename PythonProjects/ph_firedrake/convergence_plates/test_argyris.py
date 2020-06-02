from firedrake import *
import numpy as np
import scipy as sp
np.set_printoptions(threshold=np.inf)
from math import pi, floor

import scipy.linalg as la
import scipy.sparse as spa
import scipy.sparse.linalg as sp_la
# import matplotlib.pyplot as plt

L = 1
n = 5
mesh = RectangleMesh(n, n, L, L, quadrilateral=False)

#plot(mesh); plt.show()

# Domain, Subdomains, Boundary, Suboundaries

# Finite element defition


Vp = FunctionSpace(mesh, "Argyris", 5)
n_p = Vp.dim()
print("Vp dim: "+ str(n_p))

v_p = TestFunction(Vp)

e_p = TrialFunction(Vp)


for i in range(1, 5):
    boundary_nodes_t = sorted(set(Vp.boundary_nodes(i, "topological")))
    print("topological " + str(i) + ": " + str(boundary_nodes_t))


    m_form = v_p * e_p * ds(i)

    petsc_m = assemble(m_form, mat_type='aij').M.handle
    M = sp.sparse.csr_matrix(petsc_m.getValuesCSR()[::-1])
    rows, cols = spa.csr_matrix.nonzero(M)

    set_rows = np.array(list(set(rows)))
    set_cols = np.array(list(set(cols)))

    #print("rows " + str(i) + ": " + str(set_rows))
    print("cols " + str(i) + ": " + str(set_cols))

    diff = sorted(set(boundary_nodes_t).difference(set(boundary_nodes_t).difference(set_cols)))
    print("diff " + str(i) + ": " + str(diff))


# boundary_nodes_t = sorted(set(Vp.boundary_nodes("on_boundary", "topological")))
# print("topological "+ ": " + str(boundary_nodes_t))
# print(len(boundary_nodes_t))
#
# m_form = v_p * e_p * ds
#
# petsc_m = assemble(m_form, mat_type='aij').M.handle
# M = sp.sparse.csr_matrix(petsc_m.getValuesCSR()[::-1])
# rows, cols = spa.csr_matrix.nonzero(M)
#
# set_rows = np.array(list(set(rows)))
# set_cols = np.array(list(set(cols)))
#
# #print("rows "  + ": " + str(set_rows))
# print("cols "  + ": " + str(set_cols))
# print(len(set_cols))


# non_boundary_t = sorted(set(set(range(n_p)).difference(set(boundary_nodes_t))))
# print("non_boundary top: " + str(non_boundary_t))
# print(len(non_boundary_t))

# boundary_nodes_g = Vp.boundary_nodes("on_boundary", "geometric")
# print("geometrical: " + str(boundary_nodes_g))
#
# non_boundary_g = set(range(n_p)).difference(set(boundary_nodes_g))
# print("non_boundary geo: " + str(non_boundary_g))

# non_boundary_set = sorted(set(set(range(n_p)).difference(set(set_cols))))
# print("non_boundary set: " + str(non_boundary_set))
# print(len(non_boundary_set))
#
#
# diff_set_t = sorted(set(non_boundary_set).difference(set(non_boundary_t)))
# print("diff: " + str(diff_set_t))
