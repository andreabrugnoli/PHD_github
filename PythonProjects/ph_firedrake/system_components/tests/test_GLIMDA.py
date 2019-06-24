from assimulo.solvers import GLIMDA
from assimulo.implicit_ode import Implicit_Problem
import numpy as np

def res(t, y, yd): #Note that y and yd are 1-D numpy arrays.
    res = yd[0]-1.0
    return np.array([res]) #Note that the return must be numpy array, NOT a scalar.

y0  = [1.0]
yd0 = [1.0]
t0  = 1.0

mod = Implicit_Problem(res, y0, yd0, t0)

sim = GLIMDA(mod)

