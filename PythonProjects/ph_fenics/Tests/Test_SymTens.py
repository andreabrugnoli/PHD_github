

from fenics import *
import numpy as np
set_log_level(ERROR)

mesh = UnitSquareMesh(1, 1)
# plot(mesh)
# plt.show()
SDG = TensorFunctionSpace(mesh, "DG", 0, shape=(2,2),  symmetry = True)
DG = TensorFunctionSpace(mesh, "DG", 0, shape=(2,2))
vv= TestFunction(SDG)
N_sym = SDG.dim()
N_noSym = DG.dim()

print(N_noSym)
print(N_sym)
print("Given a constant symmetric tensor ((1, 2), (2, 3)).")
g = Constant( ( (1.0, 1.0), (1.0, 1.0) ) )

# g = as_tensor( [[1.0, 1.0], [1.0, 1.0]] )

f = interpolate(g, SDG )

ff = Function(SDG)
assign(ff, f)

ff1, ff2, ff3 = ff.split()


V = FunctionSpace(mesh, "DG", 0)
value = project(ff2, V)

# b = assemble(inner(vv,ff)*dx).get_local()
# print(b)

print(value.vector()[0])

value = project(ff3, V)
print(value.vector()[0])

import matplotlib.pyplot as plt








