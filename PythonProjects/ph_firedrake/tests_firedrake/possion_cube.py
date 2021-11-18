from firedrake import *

import numpy as np
np.set_printoptions(threshold=np.inf)

from scipy.sparse import csr_matrix, csc_matrix, lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
from tools_plotting import setup

import os
os.environ["OMP_NUM_THREADS"] = "1"

tol = 1e-10

L = 1
n_el = 1
deg = 3

mesh = Mesh("/home/andrea/cube.msh")
triplot(mesh)
plt.show()