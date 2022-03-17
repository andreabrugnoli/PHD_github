from FEEC.navier_stokes.problems.problem_base import *
import numpy as np
from math import pi
import sympy as sym

class TaylorGreen3D(ProblemBase):
    "3D Taylor Green problem."
    def __init__(self, options):
        ProblemBase.__init__(self, options)

        self.mesh = PeriodicBoxMesh(self.n_el, self.n_el, self.n_el, 2*pi, 2*pi, 2*pi)
        # Translate mesh so that left down corner (-pi,-pi,-pi)
        self.mesh.coordinates.dat.data[:, 0] -= pi
        self.mesh.coordinates.dat.data[:, 1] -= pi
        self.mesh.coordinates.dat.data[:, 2] -= pi
        self.init_mesh()
        self.structured_time_grid()

        # Set viscosity
        self.mu = 1.0 / 500
        # Set density
        self.rho = 1
        # Reynolds number
        self.Re = self.rho / self.mu
        # Periodic problem
        self.periodic = True
        # Solution is not exact
        self.exact = False
        # Quad mesh
        self.quad = False

    def initial_conditions(self, V_v, V_w, V_p):
        x, y, z = SpatialCoordinate(self.mesh)
        v_0 = as_vector([sin(x)*cos(y)*cos(z), -cos(x)*sin(y)*cos(z), 0])
        v_init = interpolate(v_0, V_v)

        w_0 = as_vector([-cos(x)*sin(y)*sin(z), -sin(x)*cos(y)*sin(z), 2*sin(x)*sin(y)*cos(z)])
        w_init = interpolate(w_0, V_w)

        p_0 = 1/16*(cos(2*x) + cos(2*y))*(cos(2*z) + 2)
        p_init = interpolate(p_0, V_p)
        return [v_init, w_init, p_init]

    def __str__(self):
        return "TaylorGreen3D"
