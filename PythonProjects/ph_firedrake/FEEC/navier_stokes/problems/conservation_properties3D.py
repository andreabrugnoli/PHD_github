from FEEC.navier_stokes.problems.problem_base import *
import numpy as np
from math import pi
import sympy as sym

class ConservationProperties3D(ProblemBase):
    "3D conservation properties"
    def __init__(self, options):
        ProblemBase.__init__(self, options)

        # mesh_quad = PeriodicUnitSquareMesh(self.n_el, self.n_el, direction="both", quadrilateral=True)
        # self.mesh = ExtrudedMesh(mesh_quad, self.n_el)
        # self.quad = True

        self.mesh = PeriodicUnitCubeMesh(self.n_el, self.n_el, self.n_el)
        self.quad = False

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

    def initial_conditions(self, V_v, V_w, V_p):
        x, y, z = SpatialCoordinate(self.mesh)


        v_0 = as_vector([cos(2*pi*z), sin(2*pi*z), sin(2*pi*x)])
        w_0 = as_vector([-2*pi*cos(2*pi*z), -2*pi*(cos(2*pi*x) + sin(2*pi*z)), 0])
        p_0 = Constant(0)

        if self.quad == False:
            v_init = interpolate(v_0, V_v)
            w_init = interpolate(w_0, V_w)
            p_init = interpolate(p_0, V_p)
        else:
            v_init = project(v_0, V_v)
            w_init = project(w_0, V_w)
            p_init = project(p_0, V_p)
        return [v_init, w_init, p_init]


    def __str__(self):
        return "ConservationProperties3D"
