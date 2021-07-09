## This is a first test to solve the wave equation in 2D domains using the dual filed method
# from warnings import simplefilter
# simplefilter(action='ignore', category=DeprecationWarning)

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

    L = 5/4
    mesh = RectangleMesh(n_el, n_el, L, L, quadrilateral=False)
    n_ver = FacetNormal(mesh)

    # Pp = FiniteElement("DG", tetrahedron, deg - 1)
    # Pq = FiniteElement("N1curl", tetrahedron, deg, variant='point')
    #
    # Pp_d = FiniteElement("CG", tetrahedron, deg)
    # Pq_d = FiniteElement("RT", tetrahedron, deg, variant='point')
    #
    # Vp = FunctionSpace(mesh, Pp)
    # Vq = FunctionSpace(mesh, Pq)
    #
    # Vp_d = FunctionSpace(mesh, Pp_d)
    # Vq_d = FunctionSpace(mesh, Pq_d)

    Vp = FunctionSpace(mesh, 'DG', deg-1)
    Vq = FunctionSpace(mesh, 'N1curl', deg)

    Vp_d = FunctionSpace(mesh, 'CG', deg)
    Vq_d = FunctionSpace(mesh, 'RT', deg)

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

    v0 = om_t*sin(om_x*x)*sin(om_y*y)
    in_cond = project(as_vector([v0, 0, 0, v0, 0, 0]), V)

    p0, q0, p0_d, q0_d = split(in_cond)

    Ep = 0.5 * (inner(p0, p0) * dx + inner(q0, q0) * dx)
    Ed = 0.5 * (inner(p0_d, p0_d) * dx + inner(q0_d, q0_d) * dx)
    Es = 0.5 * (p0 * p0_d * dx + dot(q0, q0_d) * dx)
    # Es = 0.5 * cross_2D(q0, q0_d) * dx

    m_form = inner(vp, Dt(p0)) * dx + inner(vq, Dt(q0)) * dx + inner(vp_d, Dt(p0_d)) * dx + inner(vq_d, Dt(q0_d)) * dx

    # Check for sign in adjoint system
    j_form = dot(vp, div(q0_d)) * dx + dot(vq, grad(p0_d)) * dx - dot(grad(vp_d), q0) * dx - dot(div(vq_d), p0) * dx \
               + vp_d * dot(q0_d, n_ver) * ds + dot(vq_d, n_ver) * p0_d * ds
    # Form defininig the problem
    f_form = m_form - j_form

    t = Constant(0.0)
    w_ex = sin(om_x * x) * sin(om_y * y) * sin(om_t * t)

    v_ex = om_t * sin(om_x * x) * sin(om_y * y) * cos(om_t * t)
    sig_ex = as_vector([om_x * cos(om_x * x) * sin(om_y * y) * sin(om_t * t),
                        om_y * sin(om_x * x) * cos(om_y * y) * sin(om_t * t)])

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
    stepper = TimeStepper(f_form, butcher_tableau, t, dt, in_cond,
                          bcs=bc, solver_parameters=params)

    Ep_vec = np.zeros((1+n_t, ))
    Ed_vec = np.zeros((1+n_t, ))
    Es_vec = np.zeros((1+n_t, ))
    E_res = np.zeros((1+n_t, ))

    Ppoint = (L/2, L/2)

    p_P = np.zeros((1+n_t,))
    p_P[0] = interpolate(v0, Vp).at(Ppoint)

    pd_P = np.zeros((1+n_t, ))
    pd_P[0] = interpolate(v0, Vp_d).at(Ppoint)

    t_vec = np.linspace(0, n_t * float(dt), 1 + n_t)

    Ep_vec[0] = assemble(Ep)
    Ed_vec[0] = assemble(Ed)
    Es_vec[0] = assemble(Es)

    print("Time    Energy")
    print("==============")

    Ep_0 = assemble(Ep)
    Ed_0 = assemble(Ed)
    Es_0 = assemble(Es)

    pn = Function(Vp, name="p primal at t_n")
    pn_d = Function(Vp_d, name="p dual at t_n")

    pn1 = Function(Vp, name="p primal at t_n1")
    pn1_d = Function(Vp_d, name="p dual at t_n1")

    qn = Function(Vq, name="q primal at t_n")
    qn_d = Function(Vq_d, name="q dual at t_n")

    qn1 = Function(Vq, name="q primal at t_n1")
    qn1_d = Function(Vq_d, name="q dual at t_n1")

    err_p = Function(Vp, name="p error primal at t_n")
    err_pd = Function(Vp_d, name="p error dual at t_n")

    for ii in tqdm(range(n_t)):
        pn.assign(interpolate(p0, Vp))
        pn_d.assign(interpolate(p0_d, Vp_d))

        qn.assign(interpolate(q0, Vq))
        qn_d.assign(interpolate(q0_d, Vq_d))

        Delta_res_n = float(dt)/2*assemble(pn_d*dot(qn_d, n_ver) * ds)

        stepper.advance()

        t.assign(float(t) + float(dt))

        pn1.assign(interpolate(p0, Vp))
        pn1_d.assign(interpolate(p0_d, Vp_d))

        qn1.assign(interpolate(q0, Vq))
        qn1_d.assign(interpolate(q0_d, Vq_d))

        Ep_vec[ii+1] = assemble(Ep)
        Ed_vec[ii+1] = assemble(Ed)
        Es_vec[ii+1] = assemble(Es)

        Delta_res_n1 = float(dt)/2*assemble(pn1_d*dot(qn1_d, n_ver) * ds)
        E_res[ii+1] = E_res[ii] + Delta_res_n + Delta_res_n1

        p_P[ii+1] = pn1.at(Ppoint)
        pd_P[ii+1] = pn1_d.at(Ppoint)
        # print("Primal energy")
        # print("{0:1.1e} {1:5e}".format(float(t), Ep_vec[ii]))
        # print("Dual energy")
        # print("{0:1.1e} {1:5e}".format(float(t), Ed_vec[ii]))
        # print("Scattering energy")
        # print("{0:1.1e} {1:5e}".format(float(t), Es_vec[ii]))

    err_p.assign(pn1 - interpolate(v_ex, Vp))
    err_pd.assign(pn1_d - interpolate(v_ex, Vp_d))

    # err_p.assign(interpolate(v_ex, Vp))
    # err_pd.assign(interpolate(v_ex, Vp_d))

    Ep_f = assemble(Ep)
    Ed_f = assemble(Ed)
    Es_f = assemble(Es)

    print(r"Initial and final primal energy:")
    print(r"Inital: ", Ep_0)
    print(r"Final: ", Ep_f)
    print(r"Delta: ", Ep_f - Ep_0)

    print(r"Initial and final dual energy:")
    print(r"Inital: ", Ed_0)
    print(r"Final: ", Ed_f)
    print(r"Delta: ", Ed_f - Ed_0)

    print(r"Initial and final scattering energy:")
    print(r"Inital: ", Es_0)
    print(r"Final: ", Es_f)
    print(r"Delta: ", Es_f - Es_0)

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

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    contours = trisurf(pn, axes=axes, cmap="inferno")
    axes.set_aspect("auto")
    axes.set_title("Primal velocity")
    fig.colorbar(contours)

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    contours = trisurf(pn_d, axes=axes, cmap="inferno")
    axes.set_aspect("auto")
    axes.set_title("Dual velocity")
    fig.colorbar(contours)

    plt.figure()
    plt.plot(t_vec, p_P, 'r-', label=r'primal $p$')
    plt.plot(t_vec, pd_P, 'b-', label=r'dual $p$')
    plt.plot(t_vec, om_t * np.sin(om_x * Ppoint[0]) * np.sin(om_y * Ppoint[1]) * np.cos(om_t * t_vec), 'g-',
             label=r'exact $p$')
    plt.xlabel(r'Time [s]')
    plt.title(r'$p$ at ' + str(Ppoint))
    plt.legend()

    return t_vec, Ep_vec, Ed_vec, Es_vec, E_res


t_vec, Ep_vec, Ed_vec, Es_vec, E_res = compute_sol(10, 100, 2, 1)

plt.figure()
plt.plot(t_vec, 0.5 * (Ep_vec + Ed_vec), 'g', label=r'Both energies')
plt.plot(t_vec, Ep_vec, 'r', label=r'Primal energies')
plt.plot(t_vec, Ed_vec, 'b', label=r'Dual energies')
plt.xlabel(r'Time [s]')
plt.title(r'Energy')
plt.legend()

plt.figure()
plt.plot(t_vec, 0.5 * (Ep_vec + Ed_vec), 'r', label=r'Both energies')
plt.plot(t_vec, Es_vec, 'b', label=r'Scattering energy')
plt.xlabel(r'Time [s]')
plt.title(r'Energy')
plt.legend()

plt.figure()
# plt.plot(t_vec, (Ep_vec - Ep_vec[0]) - E_res, 'r', label=r'Primal En. residual')
plt.plot(t_vec, (Ed_vec - Ed_vec[0]) - E_res, 'g', label=r'Dual En. residual')
# plt.plot(t_vec, (Es_vec - Es_vec[0]) - E_res, 'b', label=r'Scac En. residual')
plt.plot(t_vec, 0.5*(Ep_vec - Ep_vec[0]) + 0.5*(Ed_vec - Ed_vec[0]) - E_res, 'b--', label=r'Both Ens. residual')
plt.xlabel(r'Time [s]')
plt.title(r'Residual')
plt.legend()
plt.show()
