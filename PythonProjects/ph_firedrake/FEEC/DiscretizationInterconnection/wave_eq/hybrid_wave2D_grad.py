## This is a first test to solve the wave equation in 3d domains using the dual field method
## A staggering method is used for the time discretization

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from firedrake import *
import matplotlib.pyplot as plt
from tools_plotting import setup
from tqdm import tqdm
# from time import sleep
from matplotlib.ticker import FormatStrFormatter
import pickle

from FEEC.DiscretizationInterconnection.wave_eq.exact_eigensolution import exact_sol_wave2D


def compute_err(n_el, n_t, deg=1, t_fin=1, bd_cond="D"):
    """Compute the numerical solution of the wave equation with a DG method based on interconnection

        Parameters:
        n_el: number of elements for the discretization
        n_t: number of time instants
        deg: polynomial degree for finite
        Returns:
        some plots
       """

    def m_form10(v_1, u_1, v_0, p_0):
        m_form = inner(v_1, u_1) * dx + inner(v_0, p_0) * dx

        return m_form

    def j_form10(v_1, u_1, v_0, p_0):
        j_form = dot(v_1, grad(p_0)) * dx - dot(grad(v_0), u_1) * dx

        return j_form

    def int_facets_flow0(v_0, p_0_nor):
        form = (v_0('+') * p_0_nor('+') + v_0('-') * p_0_nor('-')) * dS + v_0 * p_0_nor * ds
        return form

    def constr_p0(v_0_nor, p_0, p_0_tan):
        form = -((v_0_nor('+') * p_0('+') + v_0_nor('-') * p_0('-')) * dS + v_0_nor * p_0 * ds) \
                + (v_0_nor('+') * p_0_tan('+') + v_0_nor('-') * p_0_tan('-')) * dS + v_0_nor * p_0_tan * ds
        return form

    def constr_p0_nor(v_0_tan, p0_nor):
        form = -(v_0_tan('+') * p0_nor('+') + v_0_tan('-') * p0_nor('-')) * dS + v_0_tan * p0_nor * ds
        return form

    def neumann_flow0(v_0_tan, neumann_bc):
        return v_0_tan * neumann_bc * ds

    L = 1/2

    mesh = RectangleMesh(n_el, n_el, 1, 1/2)
    n_ver = FacetNormal(mesh)

    # triplot(mesh)
    # plt.show()

    P0 = FiniteElement("CG", triangle, deg)
    P0f = FacetElement(P0)

    P1 = FiniteElement("N1curl", triangle, deg, variant="integral")

    P0_b = BrokenElement(P0)
    P1_b = BrokenElement(P1)

    P0f_b = BrokenElement(P0f)

    W0 = FunctionSpace(mesh, P0_b)
    W1 = FunctionSpace(mesh, P1_b)

    W0_nor = FunctionSpace(mesh, P0f_b)
    V0_tan = FunctionSpace(mesh, P0f)

    V_grad = W0 * W1 * W0_nor * V0_tan

    print(W0.dim())
    print(W1.dim())
    print(W0_nor.dim())
    print(V0_tan.dim())

    v_grad = TestFunction(V_grad)
    v0, v1, v0_nor, v0_tan = split(v_grad)

    e_grad = TrialFunction(V_grad)
    p0, u1, p0_nor, p0_tan = split(e_grad)

    dx = Measure('dx')
    ds = Measure('ds')
    dS = Measure('dS')

    x, y = SpatialCoordinate(mesh)

    dt = Constant(t_fin / n_t)

    params = {"mat_type": "aij",
              "ksp_type": "preonly"}

    # params = {"ksp_type": "gmres"}

    t_vec = np.linspace(0, n_t * float(dt), 1 + n_t)

    t = Constant(0.0)
    t_1 = Constant(dt)

    p_ex, u_ex, p_ex_1, u_ex_1 = exact_sol_wave2D(x, y, t, t_1)

    u_ex_mid = 0.5 * (u_ex + u_ex_1)
    p_ex_mid = 0.5 * (p_ex + p_ex_1)

    if bd_cond == "D":
        bc_D = [DirichletBC(V_grad.sub(3), p_ex_1, "on_boundary")]



        def dirichlet_flow0(v0_nor, u_D):
            return v0_nor * u_D * ds

    elif bd_cond == "N":
        bc_D = []

        def dirichlet_flow0(v0_nor, u_D):
            return v0_nor * Constant(0) * ds
    else:
        bc_D = [DirichletBC(V_grad.sub(3), p_ex_1, 1), \
                DirichletBC(V_grad.sub(3), p_ex_1, 3)]

        def dirichlet_flow0(v0_nor, u_D):
            return v0_nor * u_D * ds(1) + v0_nor * u_D * ds(3)

    bcs = bc_D

    dofs_D = []

    for ii in range(len(bc_D)):
        nodesD = bc_D[ii].nodes

        dofs_D = dofs_D + list(nodesD)

    dofs_D = list(set(dofs_D))
    dofs_N = list(set(np.arange(V0_tan.dim())).difference(set(dofs_D)))

    dofsV0_tan_D = W0.dim() + W1.dim() + W0_nor.dim() + np.array(dofs_D)
    dofsV0_tan_N = W0.dim() + W1.dim() + W0_nor.dim() + np.array(dofs_N)


    p0_0 = project(p_ex, W0)
    u0_1 = project(u_ex, W1)

    # The Lagrange multiplier is computed at half steps

    en_grad = Function(V_grad, name="e n")
    en_grad.sub(0).assign(p0_0)
    en_grad.sub(1).assign(u0_1)
    pn_0, un_1, pn_0_nor, pn_0_tan = en_grad.split()

    enmid_grad = Function(V_grad, name="e n+1/2")
    pnmid_0, unmid_1, pnmid_0_nor, pnmid_0_tan = enmid_grad.split()

    en1_grad = Function(V_grad, name="e n+1")
    pn1_0, un1_1, pn1_0_nor, pn1_0_tan = en1_grad.split()

    Hn_01 = 0.5 * (inner(pn_0, pn_0) * dx + inner(un_1, un_1) * dx)

    Hn_ex = 0.5 * (inner(p_ex, p_ex) * dx(domain=mesh) + inner(u_ex, u_ex) * dx(domain=mesh))

    bdflow_ex_nmid = p_ex_mid * dot(u_ex_mid, n_ver) * ds(domain=mesh)

    bdflow_nmid_D = dirichlet_flow0(pnmid_0_nor, p_ex_mid)

    flow_nmid_N = neumann_flow0(v0_tan, dot(u_ex_mid, n_ver))

    H_01_vec = np.zeros((1 + n_t,))
    H_ex_vec = np.zeros((1 + n_t,))

    bdflow10_mid_vec = np.zeros((n_t,))
    bdflow_ex_mid_vec = np.zeros((n_t,))

    errL2_p_0_vec = np.zeros((1 + n_t,))
    errL2_u_1_vec = np.zeros((1 + n_t,))

    errH1_p_0_vec = np.zeros((1 + n_t,))
    errHcurl_u_1_vec = np.zeros((1 + n_t,))

    errH_01_vec = np.zeros((1 + n_t,))

    H_01_vec[0] = assemble(Hn_01)
    H_ex_vec[0] = assemble(Hn_ex)

    errH_01_vec[0] = np.abs(H_01_vec[0] - H_ex_vec[0])

    errL2_u_1_vec[0] = errornorm(u_ex, u0_1, norm_type="L2")
    errL2_p_0_vec[0] = errornorm(p_ex, p0_0, norm_type="L2")

    errHcurl_u_1_vec[0] = errornorm(u_ex, u0_1, norm_type="Hcurl")
    errH1_p_0_vec[0] = errornorm(p_ex, p0_0, norm_type="H1")

    ## Settings of intermediate variables and matrices for the 2 linear systems

    a_form10 = 1/dt* m_form10(v1, u1, v0, p0) - 0.5 *  j_form10(v1, u1, v0, p0) \
               - int_facets_flow0(v0, p0_nor) - constr_p0(v0_nor, p0, p0_tan) - constr_p0_nor(v0_tan, p0_nor)

    # print("Computation of the solution with n elem " + str(n_el) + " n time " + str(n_t) + " deg " + str(deg))
    # print("==============")

    for ii in tqdm(range(n_t)):

        ## Integration of 10 system (Neumann natural)

        A_10 = assemble(a_form10, bcs=bc_D, mat_type='aij')

        b_form10 = 1/dt * m_form10(v1, un_1, v0, pn_0) + 0.5 * j_form10(v1, un_1, v0, pn_0) \
                    + dirichlet_flow0(v0_nor, p_ex_1) + neumann_flow0(v0_tan, dot(u_ex_mid, n_ver))

        b_vec10 = assemble(b_form10)

        solve(A_10, en1_grad, b_vec10, solver_parameters=params)

        # Computation of energy rate and fluxes

        enmid_grad.assign(0.5 * (en_grad + en1_grad))

        y_neumann = enmid_grad.vector().get_local()[dofsV0_tan_N]
        u_neumann = assemble(neumann_flow0(v0_tan, dot(u_ex_mid, n_ver))).vector().get_local()[dofsV0_tan_N]
        bdflow_neumann = np.dot(y_neumann, u_neumann)

        bdflow10_mid_vec[ii] = bdflow_neumann + assemble(dirichlet_flow0(pnmid_0_nor, p_ex_mid))
        bdflow_ex_mid_vec[ii] = assemble(bdflow_ex_nmid)

        # New assign

        en_grad.assign(en1_grad)

        pn_0, un_1, pn_0_nor, pn_0_tan = en_grad.split()
        H_01_vec[ii + 1] = assemble(Hn_01)

        t.assign(float(t) + float(dt))
        t_1.assign(float(t_1) + float(dt))

        H_ex_vec[ii + 1] = assemble(Hn_ex)

        errH_01_vec[ii + 1] = np.abs(H_01_vec[ii + 1] - H_ex_vec[ii + 1])

        errL2_p_0_vec[ii + 1] = errornorm(p_ex, pn_0, norm_type="L2")
        errL2_u_1_vec[ii + 1] = errornorm(u_ex, un_1, norm_type="L2")

        errHcurl_u_1_vec[ii + 1] = errornorm(u_ex, un_1, norm_type="Hcurl")
        errH1_p_0_vec[ii + 1] = errornorm(p_ex, pn_0, norm_type="H1")

        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        contours = trisurf(project(p_ex, W0), axes=axes, cmap="inferno")
        axes.set_aspect("auto")
        axes.set_title("p0 ex")
        fig.colorbar(contours)
        fig = plt.figure()
        axes = fig.add_subplot(111, projection='3d')
        contours = trisurf(pn_0, axes=axes, cmap="inferno")
        axes.set_aspect("auto")
        axes.set_title("p0 h")
        fig.colorbar(contours)

    # err_p_0 = np.sqrt(np.sum(float(dt) * np.power(err_p_0_vec, 2)))
    # err_u_1 = np.sqrt(np.sum(float(dt) * np.power(err_u_1_vec, 2)))
    #
    # err_p_0 = max(err_p_0_vec)
    # err_u_1 = max(err_u_1_vec)



    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    contours = trisurf(project(p_ex, W0), axes=axes, cmap="inferno")
    axes.set_aspect("auto")
    axes.set_title("Error $p_0$")
    fig.colorbar(contours)

    errL2_p_0 = errL2_p_0_vec[-1]
    errL2_u_1 = errL2_u_1_vec[-1]

    errH1_p_0 = errH1_p_0_vec[-1]
    errHcurl_u_1 = errHcurl_u_1_vec[-1]

    errH_01 = errH_01_vec[-1]

    dict_res = {"t_span": t_vec, "energy_ex": H_ex_vec, "energy_01": H_01_vec, \
                "flow_ex_mid": bdflow_ex_mid_vec, "flow10_mid": bdflow10_mid_vec, \
                "err_u1": [errL2_u_1, errHcurl_u_1], "err_p0": [errL2_p_0, errH1_p_0], "err_H": errH_01}

    return dict_res


bd_cond = 'N' #input("Enter bc: ")

n_elem = 10
pol_deg = 2

n_time = 5
t_fin = 1

dt = t_fin / n_time

results = compute_err(n_elem, n_time, pol_deg, t_fin, bd_cond=bd_cond)

t_vec = results["t_span"]

bdflow10_mid = results["flow10_mid"]

H_01 = results["energy_01"]
H_ex = results["energy_ex"]

bdflow_ex_nmid = results["flow_ex_mid"]

errL2_u1, errHcurl_u1 = results["err_u1"]
errL2_p0, errH1_p0 = results["err_p0"]

err_H01 = results["err_H"]

dt = t_vec[-1] / (len(t_vec)-1)


plt.figure()
plt.plot(t_vec[1:]-dt/2, np.diff(H_01)/dt - bdflow10_mid, 'r-.')
plt.xlabel(r'Time $[\mathrm{s}]$')
plt.title(r'Power balance conservation')

#
# plt.figure()
# plt.plot(t_vec[1:]-dt/2, abs(bdflow_ex_nmid - bdflow10_mid), 'r-.')
# plt.xlabel(r'Time $[\mathrm{s}]$')
# plt.title(r'Power balance conservation')

plt.show()
# dictres_file = open("results_wave.pkl", "wb")
# pickle.dump(results, dictres_file)
# dictres_file.close()
#


