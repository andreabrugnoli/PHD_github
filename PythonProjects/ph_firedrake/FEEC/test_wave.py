## This is a first test to solve the wave equation in 2D domains using the dual filed method

from firedrake import *
import matplotlib.pyplot as plt



def compute_sol(n_el):
    """Compute the numerical solution of the wave equation with the dual field method

        Parameters:
        n_el: number of elements for the discretization

        Returns:
        some plots

       """

    L = 1
    mesh = RectangleMesh(n_el, n_el, L, L, quadrilateral=False)

    Vp = FunctionSpace(mesh, 'DG', 0)
    Vq = FunctionSpace(mesh, 'N1curl', 1)

    Vp_dual = FunctionSpace(mesh, 'CG', 1)
    Vq_dual = FunctionSpace(mesh, 'RT', 1)

    V = Vp * Vq * Vp_dual * Vq_dual

    t = 0.
    t_ = Constant(t)
    x, y = SpatialCoordinate(mesh)

    w_ex = sin(2*pi*x)*sin(2*pi*y)*sin(2*pi*t_)

    v_ex = 2*pi*sin(2*pi*x)*sin(2*pi*y)*cos(2*pi*t_)
    sig_ex = as_vector([2*pi*cos(2*pi*x)*sin(2*pi*y)*sin(2*pi*t_),
                        2*pi*sin(2*pi*x)*cos(2*pi*y)*sin(2*pi*t_)])







    return V


V = compute_sol(1)