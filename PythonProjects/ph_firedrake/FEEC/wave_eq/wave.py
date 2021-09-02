## This is a first test to solve the wave equation in 2D domains using the dual filed method
# from warnings import simplefilter
# simplefilter(action='ignore', category=DeprecationWarning)

import os
os.environ["OMP_NUM_THREADS"] = "1"

# import numpy as np
from firedrake import *
from irksome import GaussLegendre, Dt, AdaptiveTimeStepper, TimeStepper, LobattoIIIA
import matplotlib.pyplot as plt
from tools_plotting import setup
from tqdm import tqdm
# from time import sleaep


def compute_sol(n_el, n_t, deg=1, t_fin=1):
    """Compute the numerical solution of the wave equation with the dual field method

        Parameters:
        n_el: number of elements for the discretization
        n_t: number of time instants
        deg: polynomial degree for finite
        Returns:
        some plots

       """
    def cross_2D(a, b):
        return a[0] * b[1] - a[1] * b[0]

    L = 1
    mesh = RectangleMesh(n_el, n_el, L, L, quadrilateral=False)
    n_ver = FacetNormal(mesh)

    P_0 = FiniteElement("CG", triangle, deg)
    P_1e = FiniteElement("N1curl", triangle, deg, variant='integral')
    P_1f = FiniteElement("RT", triangle, deg, variant='integral')
    P_2 = FiniteElement("DG", triangle, deg - 1)

    Vp = FunctionSpace(mesh, P_2)
    Vq = FunctionSpace(mesh, P_1e)

    Vp_d = FunctionSpace(mesh, P_0)
    Vq_d = FunctionSpace(mesh, P_1f)

    # Vp = FunctionSpace(mesh, 'DG', deg-1)
    # Vq = FunctionSpace(mesh, 'N1curl', deg, variant='integral')
    #
    # Vp_d = FunctionSpace(mesh, 'CG', deg)
    # Vq_d = FunctionSpace(mesh, 'RT', deg, variant='integral')

    V = Vp * Vq * Vp_d * Vq_d

    v = TestFunction(V)
    vp, vq, vp_d, vq_d = split(v)

    e = TrialFunction(V)
    p, q, p_d, q_d = split(e)

    dx = Measure('dx')
    ds = Measure('ds')

    x, y = SpatialCoordinate(mesh)

    om_x = pi
    om_y = pi
    om_t = np.sqrt(om_x**2 + om_y**2)
    phi_x = 0
    phi_y = 0
    phi_t = 0

    t = Constant(0.0)
    w_ex = sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_t * t + phi_t)

    v_ex = om_t * sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * cos(om_t * t + phi_t)
    sig_ex = as_vector([om_x * cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_t * t + phi_t),
                        om_y * sin(om_x * x + phi_x) * cos(om_y * y + phi_y) * sin(om_t * t + phi_t)])

    in_cond = project(as_vector([v_ex, sig_ex[0], sig_ex[1], v_ex, sig_ex[0], sig_ex[1]]), V)

    p0, q0, p0_d, q0_d = split(in_cond)

    E_L2Hdiv = 0.5*(inner(p0, p0) * dx + inner(q0_d, q0_d) * dx)
    E_H1Hrot = 0.5*(inner(p0_d, p0_d) * dx + inner(q0, q0) * dx)

    # Es = 0.5 * cross_2D(q0, q0_d) * dx

    Hdot = div(q0_d) * p0_d * dx + inner(grad(p0_d), q0_d) * dx
    bdflow = p0_d * dot(q0_d, n_ver) * ds

    m_form = inner(vp, Dt(p0)) * dx + inner(vq, Dt(q0)) * dx + inner(vp_d, Dt(p0_d)) * dx + inner(vq_d, Dt(q0_d)) * dx

    # Check for sign in adjoint system
    j_form = dot(vp, div(q0_d)) * dx + dot(vq, grad(p0_d)) * dx - dot(grad(vp_d), q0) * dx - dot(div(vq_d), p0) * dx #\
               # + vp_d * dot(q0_d, n_ver) * ds + dot(vq_d, n_ver) * p0_d * ds
    # Form defininig the problem
    f_form = m_form - j_form


    # Method of manifactured solution: check demo on Firedrake irksome
    # rhs = expand_derivatives(diff(uexact, t)) - div(grad(uexact))

    dt = Constant(t_fin / n_t)
    butcher_tableau = GaussLegendre(1)
    # butcher_tableau = LobattoIIIA(2)

    params = {"mat_type": "aij",
              "snes_type": "ksponly",
              "ksp_type": "preonly",
              "pc_type": "lu"}

    bc = DirichletBC(V.sub(2), v_ex, "on_boundary")
    # bc = []

    stepper = TimeStepper(f_form, butcher_tableau, t, dt, in_cond,
                          bcs=bc, solver_parameters=params)

    E_L2Hdiv_vec = np.zeros((1+n_t, ))
    E_H1Hrot_vec = np.zeros((1+n_t, ))

    Hdot_vec = np.zeros((1 + n_t,))
    bdflow_vec = np.zeros((1 + n_t,))

    Ppoint = (L/2, L/2)

    p_P = np.zeros((1+n_t,))
    p_P[0] = interpolate(v_ex, Vp).at(Ppoint)

    pd_P = np.zeros((1+n_t, ))
    pd_P[0] = interpolate(v_ex, Vp_d).at(Ppoint)

    t_vec = np.linspace(0, n_t * float(dt), 1 + n_t)

    Hdot_vec[0] = assemble(Hdot)
    bdflow_vec[0] = assemble(bdflow)

    E_L2Hdiv_vec[0] = assemble(E_L2Hdiv)
    E_H1Hrot_vec[0] = assemble(E_H1Hrot)

    print("Time    Energy")
    print("==============")

    pn = Function(Vp, name="p primal at t_n")
    pn_d = Function(Vp_d, name="p dual at t_n")

    err_p = Function(Vp, name="p error primal at t_n")
    err_pd = Function(Vp_d, name="p error dual at t_n")

    for ii in tqdm(range(n_t)):
        stepper.advance()

        Hdot_vec[ii+1] = assemble(Hdot)
        bdflow_vec[ii+1] = assemble(bdflow)

        E_L2Hdiv_vec[ii+1] = assemble(E_L2Hdiv)
        E_H1Hrot_vec[ii+1] = assemble(E_H1Hrot)

        pn.assign(interpolate(p0, Vp))
        pn_d.assign(interpolate(p0_d, Vp_d))

        p_P[ii+1] = pn.at(Ppoint)
        pd_P[ii+1] = pn_d.at(Ppoint)

        t.assign(float(t) + float(dt))
        # print("Primal energy")
        # print("{0:1.1e} {1:5e}".format(float(t), Ep_vec[ii]))
        # print("Dual energy")
        # print("{0:1.1e} {1:5e}".format(float(t), Ed_vec[ii]))
        # print("Scattering energy")
        # print("{0:1.1e} {1:5e}".format(float(t), Es_vec[ii]))

    err_p.assign(pn - interpolate(v_ex, Vp))
    err_pd.assign(pn_d - interpolate(v_ex, Vp_d))

    # err_p.assign(interpolate(v_ex, Vp))
    # err_pd.assign(interpolate(v_ex, Vp_d))

    print(r"Initial and final L2Hdiv energy:")
    print(r"Inital: ", E_L2Hdiv_vec[0])
    print(r"Final: ", E_L2Hdiv_vec[-1])
    print(r"Delta: ", E_L2Hdiv_vec[-1] - E_L2Hdiv_vec[0])

    print(r"Initial and final H1Hrot energy:")
    print(r"Inital: ", E_L2Hdiv_vec[0])
    print(r"Final: ", E_L2Hdiv_vec[-1])
    print(r"Delta: ", E_L2Hdiv_vec[-1] - E_L2Hdiv_vec[0])

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    contours = trisurf(err_p, axes=axes, cmap="inferno")
    axes.set_aspect("auto")
    axes.set_title("Error primal velocity")
    fig.colorbar(contours)

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    contours = trisurf(err_pd, axes=axes, cmap="inferno")
    axes.set_aspect("auto")
    axes.set_title("Error dual velocity")
    fig.colorbar(contours)

    # fig = plt.figure()
    # axes = fig.add_subplot(111, projection='3d')
    # contours = trisurf(pn, axes=axes, cmap="inferno")
    # axes.set_aspect("auto")
    # axes.set_title("Primal velocity")
    # fig.colorbar(contours)
    #
    # fig = plt.figure()
    # axes = fig.add_subplot(111, projection='3d')
    # contours = trisurf(pn_d, axes=axes, cmap="inferno")
    # axes.set_aspect("auto")
    # axes.set_title("Dual velocity")
    # fig.colorbar(contours)

    plt.figure()
    plt.plot(t_vec, p_P, 'r-', label=r'primal $p$')
    plt.plot(t_vec, pd_P, 'b-', label=r'dual $p$')
    plt.plot(t_vec, om_t * np.sin(om_x * Ppoint[0]) * np.sin(om_y * Ppoint[1]) * np.cos(om_t * t_vec), 'g-',
             label=r'exact $p$')
    plt.xlabel(r'Time [s]')
    plt.title(r'$p$ at ' + str(Ppoint))
    plt.legend()

    dict_res = {"t_span": t_vec, "energy_p": Ep_vec, "energy_d": Ed_vec, "energy_s": Es_vec, "power": Hdot_vec, \
                "flow": bdflow_vec, "energy_L2Hdiv": Emix_L2Hdiv_vec, "energy_H1Hrot": Emix_H1Hrot_vec}

    return dict_res

results = compute_sol(10, 100, 1, 1)

t_vec = results["t_span"]
Ep_vec = results["energy_p"]
Ed_vec = results["energy_d"]
Es_vec = results["energy_s"]
Hdot_vec = results["power"]
bdflow_vec = results["flow"]

Emix1 = results["energy_L2Hdiv"]
Emix2 = results["energy_H1Hrot"]

plt.figure()
plt.plot(t_vec, Emix1, 'r', label=r'L2Hdiv')
plt.plot(t_vec, Emix2, 'b', label=r'H1Hrot')
plt.xlabel(r'Time [s]')
plt.title(r' Mixed energy')
plt.legend()


plt.figure()
plt.plot(t_vec, Hdot_vec - bdflow_vec, 'ro', label=r'Energy residual')
plt.xlabel(r'Time [s]')
plt.title(r'Energy residual')
plt.legend()
plt.show()