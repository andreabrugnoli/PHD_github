from firedrake import *
import matplotlib.pyplot as plt

n_el = 4
L = 1
m = IntervalMesh(n_el, L)
n_points = n_el + 1
# mesh = ExtrudedMesh(m, n_points, layer_height=L/n_el, extrusion_type='uniform')

mesh = RectangleMesh(n_el, n_el, L, L, quadrilateral=True)

# plot(mesh); plt.show()
Her = FiniteElement("Hermite", mesh.ufl_cell(), 3)
Her_quad = TensorProductElement(Her, Her)

V = FunctionSpace(mesh, Her_quad)