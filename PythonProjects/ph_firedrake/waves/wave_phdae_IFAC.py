# Mindlin plate written with the port Hamiltonian approach
from firedrake import *
import numpy as np
import scipy.linalg as la
import warnings
np.set_printoptions(threshold=np.inf)
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm
from firedrake.plot import _two_dimension_triangle_func_val
from mpl_toolkits.mplot3d import Axes3D
plt.rc('text', usetex=True)

deg = 1


# Finite element defition

# The unit square mesh is divided in :math:`N\times N` quadrilaterals::
path_mesh = "/home/a.brugnoli/GitProjects/PythonProjects/ph_firedrake/waves/meshes/"
mesh1 = Mesh(path_mesh + "dom1.msh")
mesh2 = Mesh(path_mesh + "dom2.msh")
# mesh1 = Mesh(path_mesh + "circle1.msh")
# mesh2 = Mesh(path_mesh + "circle2.msh")
figure = plt.figure()
ax = figure.add_subplot(111)
plot(mesh1, axes=ax)
plot(mesh2, axes=ax)
plt.show()
plt.show()
