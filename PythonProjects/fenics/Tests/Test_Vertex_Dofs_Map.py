from fenics import *
import numpy as np

mesh = UnitSquareMesh(2, 2)
d = mesh.geometry().dim()

V = FiniteElement('CG', triangle, 1)
Q = VectorElement('CG', triangle, 1)
W = FunctionSpace(mesh, MixedElement([V, Q]))
# W = MixedFunctionSpace([Q, Q])

v_d = vertex_to_dof_map(W)
d_v = dof_to_vertex_map(W)

# Coordinates of all dofs
dof_x = W.tabulate_dof_coordinates().reshape((-1, d))
# di_dx[dof_index] = coordinates of dof
di_dx = dof_x.tolist()
# Coordinates of all vertices
vertex_x = mesh.coordinates().reshape((-1, d))
# vi_vx[vertex_index] = coordinates of index
vi_vx = vertex_x.tolist()

dofs_V = W.sub(0).dofmap().dofs()
dofV_x = dof_x[dofs_V]

n = W.dofmap().num_entity_dofs(0)
# The test shows how the vertex_to_dofmap and dof_to_vertex_map
# should be interpreted. [vi/n] reflects the shift due to mixed space
assert all(np.allclose(di_dx[di], vi_vx[int(vi/n)])
               for vi, di in enumerate(v_d))
assert all(np.allclose(di_dx[di], vi_vx[int(vi/n)])
               for di, vi in enumerate(d_v))

# Given vertex_index as a value from range(mesh.num_vertices()) and
# a subspace index, what dof sits there? Note that the subspace index
#  corresponds to the flattened space. That is for W=[V, Q],
#  0 -> W[0][0], 1 -> W[0][1], 2 -> W[1]
W_sub_vertex_to_dof_map = [[v_d[vi_n*n + sub]
                            for vi_n in range(mesh.num_vertices())]
                           for sub in range(n)]

# Check the map
def check(sub_vertex_index):
    sub, vertex_index = sub_vertex_index
    my_dof = W_sub_vertex_to_dof_map[sub][vertex_index]
    return v_d[d_v[my_dof]] == my_dof

assert all(map(check, ((sub, vertex_index) for vertex_index in range(mesh.num_vertices())
                                            for sub in range(n))))
