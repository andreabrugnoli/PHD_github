from FEEC.navier_stokes.problems.problem_base import *
import numpy as np
from math import pi
import sympy as sym


class ExactEuler2D(ProblemBase):
    "2D Taylor Green problem."
    def __init__(self, options):
        ProblemBase.__init__(self, options)

        self.mesh = PeriodicUnitSquareMesh(self.n_el, self.n_el)
        self.init_mesh()
        self.structured_time_grid()

        # Periodic Problem
        self.periodic = True
        # Solution exact
        self.exact = True
        # Quad mesh
        self.quad = False


    def exact_solution(self, time=0):

        x, y = SpatialCoordinate(self.mesh)
        t = Constant(time)

        v_1 = 1-2*cos(2*pi*(x-t))*sin(2*pi*(y-t))
        v_2 = 1+2*sin(2*pi*(x-t))*cos(2*pi*(y-t))

        p = -cos(4*pi*(x-t)) - cos(4*pi*(y-t))

        w = 8*pi*cos(2*pi*(x-t))*cos(2*pi*(y-t))
        return as_vector([v_1,v_2]), w, p

    def get_exact_sol_at_t(self, t_i):
        v_ex, w_ex, p_ex = self.exact_solution(time=t_i)
        return v_ex, w_ex, p_ex

    def initial_conditions(self, V_v, V_w, V_p):
        if self.quad == False:
            v_ex, w_ex, p_ex = self.exact_solution()
            v_init = interpolate(v_ex, V_v)
            if V_w is not None:
                w_init = interpolate(w_ex, V_w)
            else:
                w_init = None
            p_init = interpolate(p_ex, V_p)
        else:
            v_ex, w_ex, p_ex = self.exact_solution()
            v_init = project(v_ex, V_v)
            if V_w is not None:
                w_init = project(w_ex, V_w)
            else:
                w_init = None
            p_init = project(p_ex, V_p)
        return [v_init, w_init, p_init]

    def init_outputs(self, t_c):
        # 6 outputs --> 3 exact states (velocity , vorticity and pressure)
        # and 3 exact integral quantities at time t (energy, enstrophy, helicity)
        u_ex_t, w_ex_t, p_ex_t = self.get_exact_sol_at_t(t_c)
        H_ex_t = 0.5 * (inner(u_ex_t, u_ex_t) * dx(domain=self.mesh))
        E_ex_t = 0.5 * (inner(w_ex_t, w_ex_t) * dx(domain=self.mesh))
        if self.dimM == 2:
            Ch_ex_t = None
        elif self.dimM == 2:
            Ch_ex_t = inner(u_ex_t, w_ex_t) * dx(domain=self.mesh)
        return [u_ex_t, w_ex_t, p_ex_t,H_ex_t,E_ex_t,Ch_ex_t]

    def calculate_outputs(self,exact_arr, u_t,w_t,p_t):
        err_u = errornorm(exact_arr[0], u_t, norm_type="L2")
        if w_t is not None:
            err_w = errornorm(exact_arr[1], w_t, norm_type="L2")
        else:
            err_w = 0.0 # Indicating that solver has no vorticity information
        err_p = errornorm(exact_arr[2], p_t, norm_type="L2")
        H_ex_t = assemble(exact_arr[3])
        E_ex_t = assemble(exact_arr[4])
        Ch_ex_t = 0.0#assemble(exact_arr[5])
        return np.array([err_u,err_w,err_p,H_ex_t,E_ex_t,Ch_ex_t])



    def __str__(self):
        return "PeriodicAnalyticalEuler2D"
