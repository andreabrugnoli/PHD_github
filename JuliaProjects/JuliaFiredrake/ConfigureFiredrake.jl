ENV["PYCALL_JL_RUNTIME_PYTHON"] = Sys.which("python")

using PyCall

py"""
from firedrake import *

import numpy as np

import matplotlib.pyplot as plt

import os
os.environ["OMP_NUM_THREADS"] = "1"

tol = 1e-10

L = 1
n_el = 1
deg = 3

mesh = Mesh("/home/andrea/cube.msh")
triplot(mesh)
x = SpatialCoordinate(mesh)

V = FunctionSpace(mesh, "CG", 2)
u = Function(V)

u.assign(sin(x[0]*pi)*sin(2*x[1]*pi))

triplot(u)

plt.show()
"""
