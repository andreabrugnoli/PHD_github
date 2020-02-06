from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)
from math import pi, floor
import matplotlib
import matplotlib.pyplot as plt
import petsc4py

matplotlib.rcParams['text.usetex'] = True
save_res = False
bc_input = 'CSSC_BJT'


def compute_constants():
    A = np.array([[np.cosh(pi), - np.cosh(pi), -np.sinh(pi), np.sinh(pi)],
                  [-pi * np.sinh(pi), pi * np.sinh(pi) + np.cosh(pi), pi * np.cosh(pi),
                   -(np.sinh(pi) + pi * np.cosh(pi))],
                  [np.cosh(pi), np.cosh(pi), np.sinh(pi), np.sinh(pi)],
                  [pi * np.sinh(pi), pi * np.sinh(pi) + np.cosh(pi), pi * np.cosh(pi),
                   np.sinh(pi) + pi * np.cosh(pi)]])

    b = np.array([np.sin(pi), -pi * np.cos(pi), -np.sin(pi), -pi * np.cos(pi)])

    a, b, c, d = np.linalg.solve(A, b)

    return a, b, c, d

def compute_err(n, r):

    h_mesh = 1 / n

    rho = Constant(5600)  # kg/m^3
    h = Constant(0.001)

    Lx = 2
    Ly = 2

    mesh_int = IntervalMesh(n, Lx)
    mesh = ExtrudedMesh(mesh_int, n)

    def bending_curv(momenta):
        kappa = momenta
        return kappa

    def j_operator(v_p, v_q, e_p, e_q):

        # n_ver = FacetNormal(mesh)
        # s_ver = as_vector([-n_ver[1], n_ver[0]])
        #
        # e_mnn = inner(e_q, outer(n_ver, n_ver))
        # v_mnn = inner(v_q, outer(n_ver, n_ver))
        #
        # e_mns = inner(e_q, outer(n_ver, s_ver))
        # v_mns = inner(v_q, outer(n_ver, s_ver))
        #
        # j_graddiv = dot(grad(v_p), div(e_q)) * dx \
        #             + v_p * dot(grad(e_mns), s_ver) * ds_v \
        #             + v_p * dot(grad(e_mns), s_ver) * ds_b \
        #             + v_p * dot(grad(e_mns), s_ver) * ds_t
        # j_divgrad = - dot(div(v_q), grad(e_p)) * dx \
        #             - dot(grad(v_mns), s_ver) * e_p * ds_v \
        #             - dot(grad(v_mns), s_ver) * e_p * ds_b \
        #             - dot(grad(v_mns), s_ver) * e_p * ds_t
        #
        # j_form = j_graddiv + j_divgrad

        j_form = dot(grad(v_p), div(e_q)) * dx - dot(div(v_q), grad(e_p)) * dx

        return j_form


    # Finite element defition

    CG_deg1 = FiniteElement("CG", interval, r)
    DG_deg1 = FiniteElement("DG", interval, r)

    DG_deg = FiniteElement("DG", interval, r - 1)

    P_CG1_DG = TensorProductElement(CG_deg1, DG_deg)
    P_DG_CG1 = TensorProductElement(DG_deg, CG_deg1)

    RT_horiz = HDivElement(P_CG1_DG)
    RT_vert = HDivElement(P_DG_CG1)
    RT_quad = RT_horiz + RT_vert

    P_CG1_DG1 = TensorProductElement(CG_deg1, DG_deg1)
    P_DG1_CG1 = TensorProductElement(DG_deg1, CG_deg1)

    BDM_horiz = HDivElement(P_CG1_DG1)
    BDM_vert = HDivElement(P_DG1_CG1)
    BDM_quad = BDM_horiz + BDM_vert

    Vp = FunctionSpace(mesh, "CG", r)
    VqD = FunctionSpace(mesh, RT_quad)
    Vq12 = FunctionSpace(mesh, "CG", r)

    V = MixedFunctionSpace([Vp, VqD, Vq12])

    n_Vp = V.sub(0).dim()
    n_Vq = V.sub(1).dim()
    n_V = V.dim()
    print(n_V)

    v = TestFunction(V)
    v_p, v_qD, v_q12 = split(v)

    e = TrialFunction(V)
    e_p, e_qD, e_q12 = split(e)

    v_q = as_tensor([[v_qD[0], v_q12],
                     [v_q12, v_qD[1]]
                     ])

    e_q = as_tensor([[e_qD[0], e_q12],
                     [e_q12, e_qD[1]]
                     ])

    al_p = rho * h * e_p
    al_q = bending_curv(e_q)

    dx = Measure('dx')
    ds = Measure('ds')
    dS = Measure("dS")

    m_form = inner(v_p, al_p) * dx + inner(v_q, al_q) * dx

    j_form = j_operator(v_p, v_q, e_p, e_q)

    bcs = []

    # Rivedere l'imposizione delle bcs sulle mesh estruse
    bc_p_l = DirichletBC(V.sub(0), Constant(0.0), 1)
    bc_p_t = DirichletBC(V.sub(0), Constant(0.0), "top")
    bc_p_b = DirichletBC(V.sub(0), Constant(0.0), "bottom")
    bc_p_r = DirichletBC(V.sub(0), Constant(0.0), 2)

    bc_qD_t = DirichletBC(V.sub(1), Constant((0.0, 0.0)), "top")
    bc_qD_b = DirichletBC(V.sub(1), Constant((0.0, 0.0)), "bottom")

    bcs.append(bc_p_l)
    bcs.append(bc_p_t)
    bcs.append(bc_p_b)
    bcs.append(bc_p_r)

    bcs.append(bc_qD_t)
    bcs.append(bc_qD_b)

    t = 0.
    t_ = Constant(t)
    t_1 = Constant(t)
    t_fin = 1  # total simulation time
    x_til, y_til = SpatialCoordinate(mesh)
    x = x_til - 1
    y = y_til - 1

    beta = 1

    a, b, c, d = compute_constants()
    wst = ((a + b * x) * cosh(pi * x) + (c + d * x) * sinh(pi * x) + sin(pi * x)) * sin(pi * y)
    fst = 4 * pi ** 4 * sin(pi * x) * sin(pi * y)

    wst_x = (b * cosh(pi * x) + (a + b * x) * pi * sinh(pi * x) + d * sinh(pi * x) + (c + d * x) * pi * cosh(
        pi * x) + pi * cos(pi * x)) * sin(pi * y)
    wst_y = ((a + b * x) * cosh(pi * x) + (c + d * x) * sinh(pi * x) + sin(pi * x)) * pi * cos(pi * y)

    wst_xx = (2 * b * pi * sinh(pi * x) + (a + b * x) * pi ** 2 * cosh(pi * x) + 2 * d * pi * cosh(pi * x) + (
                c + d * x) * pi ** 2 * sinh(pi * x) - pi ** 2 * sin(pi * x)) * sin(pi * y)
    wst_yy = - ((a + b * x) * cosh(pi * x) + (c + d * x) * sinh(pi * x) + sin(pi * x)) * pi ** 2 * sin(pi * y)
    wst_xy = (b * cosh(pi * x) + (a + b * x) * pi * sinh(pi * x) + d * sinh(pi * x) + (c + d * x) * pi * cosh(
        pi * x) + pi * cos(pi * x)) * pi * cos(pi * y)

    wst_xxx = (3 * b * pi ** 2 * cosh(pi * x) + (a + b * x) * pi ** 3 * sinh(pi * x) + 3 * d * pi ** 2 * sinh(
        pi * x) + (c + d * x) * pi ** 3 * cosh(pi * x) \
               - pi ** 3 * cos(pi * x)) * sin(pi * y)

    wst_xyy = -(b * cosh(pi * x) + (a + b * x) * pi * sinh(pi * x) + d * sinh(pi * x) + (c + d * x) * pi * cosh(
        pi * x) + pi * cos(pi * x)) * pi ** 2 * sin(pi * y)

    # tol = 1e-13
    # print(abs(interpolate(wst, Vp).at(0, Ly/4)))
    # print(abs(interpolate(wst_x, Vp).at(0, Ly/4)))
    # print(abs(interpolate(wst_xx, Vp).at(Lx, Ly/4)))
    # print(abs(interpolate(wst_xxx + 2*wst_xyy, Vp).at(Lx, Ly/4)))
    # assert(abs(interpolate(wst, Vp).at(0, Ly/4))<=tol)
    # assert (abs(interpolate(wst_x, Vp).at(0, Ly/4)) <= tol)
    # assert(abs(interpolate(wst_xx, Vp).at(Lx, Ly/4))<=tol)
    # assert (abs(interpolate(wst_xxx + 2*wst_xyy, Vp).at(Lx, Ly/4)) <=tol)

    wdyn = wst * sin(beta * t_)
    wdyn_xx = wst_xx * sin(beta * t_)
    wdyn_yy = wst_yy * sin(beta * t_)
    wdyn_xy = wst_xy * sin(beta * t_)

    dt_w = beta * wst * cos(beta * t_)
    dt_w_x = beta * wst_x * cos(beta * t_)
    dt_w_y = beta * wst_y * cos(beta * t_)

    v_exact = dt_w
    grad_vex = as_vector([dt_w_x, dt_w_y])

    kappa_ex = as_tensor([[wdyn_xx, wdyn_xy],
                          [wdyn_xy, wdyn_yy]])

    sigma_ex = kappa_ex

    dtt_w = -beta ** 2 * wst * sin(beta * t_)
    dtt_w1 = -beta ** 2 * wst * sin(beta * t_1)

    fdyn = fst * sin(beta * t_) + rho * h * dtt_w
    fdyn1 = fst * sin(beta * t_1) + rho * h * dtt_w1

    f_form = v_p * fdyn * dx
    f_form1 = v_p * fdyn1 * dx

    # J = assemble(j_form)
    # M = assemble(m_form)

    # Apply boundary conditions to M, J
    # [bc.apply(J) for bc in bcs]
    # [bc.apply(M) for bc in bcs]

    dt = 0.1*h_mesh
    theta = 0.5

    lhs = m_form - dt*theta*j_form

    e_n1 = Function(V, name="e next")
    e_n = Function(V,  name="e old")
    w_n1 = Function(Vp, name="w old")
    w_n = Function(Vp, name="w next")

    e_n.sub(0).assign(interpolate(v_exact, Vp))

    ep_n, eqD_n, eq12_n = e_n.split()

    eq_n = as_tensor([[eqD_n[0], eq12_n],
                     [eq12_n, eqD_n[1]]
                     ])

    w_n.assign(Constant(0.0))

    n_t = int(floor(t_fin/dt) + 1)

    # w_err_H1 = np.zeros((n_t,))
    v_err_L2 = np.zeros((n_t,))
    v_err_H1 = np.zeros((n_t,))

    sig_err_L2 = np.zeros((n_t,))
    sig_err_div = np.zeros((n_t,))

    w_atP = np.zeros((n_t,))
    v_atP = np.zeros((n_t,))
    Ppoint = (Lx / 14, Ly / 3)
    v_atP[0] = ep_n.at(Ppoint)

    # w_err_H1[0] = np.sqrt(assemble(dot(w_n-w_exact, w_n-w_exact) *dx
    #                      + dot(grad(w_n) - grad_wex, grad(w_n) - grad_wex) * dx))
    v_err_H1[0] = np.sqrt(assemble(dot(ep_n - v_exact, ep_n - v_exact) * dx
                                   + dot(grad(ep_n) - grad_vex, grad(ep_n) - grad_vex) * dx))
    v_err_L2[0] = np.sqrt(assemble(dot(ep_n - v_exact, ep_n - v_exact) * dx))

    sig_err_div[0] = np.sqrt(assemble(inner(eq_n - sigma_ex, eq_n - sigma_ex) * dx
                                      + inner(div(eq_n - sigma_ex), div(eq_n - sigma_ex)) * dx))
    sig_err_L2[0] = np.sqrt(assemble(inner(eq_n - sigma_ex, eq_n - sigma_ex) * dx))

    t_vec = np.linspace(0, t_fin, num=n_t)

    A = assemble(lhs, bcs=bcs, mat_type='aij')

    # param = {'ksp_converged_reason': None,
    #                      'ksp_monitor_true_residual': None,
    #                      'ksp_view': None}

    param = {"ksp_type": "preonly", "pc_type": "lu"}

    # print(e_n.vector().get_local())
    for i in range(1, n_t):

        t_.assign(t)
        t_1.assign(t+dt)

        ep_n, eqD_n, eq12_n = e_n.split()

        eq_n = as_tensor([[eqD_n[0], eq12_n],
                          [eq12_n, eqD_n[1]]
                          ])

        alp_n = rho * h * ep_n
        alq_n = bending_curv(eq_n)

        # rhs = inner(v_p, alp_n) * dx + inner(v_q, alq_n) * dx \
        #       + dt * (1 - theta) * (- inner(grad(grad(v_p)), eq_n) * dx \
        #       + jump(grad(v_p), n_ver) * dot(dot(eq_n('+'), n_ver('+')), n_ver('+')) * dS \
        #       + dot(grad(v_p), n_ver) * dot(dot(eq_n, n_ver), n_ver) * ds \
        #       + inner(v_q, grad(grad(ep_n))) * dx \
        #       - dot(dot(v_q('+'), n_ver('+')), n_ver('+')) * jump(grad(ep_n), n_ver) * dS \
        #       - dot(dot(v_q, n_ver), n_ver) * dot(grad(ep_n), n_ver) * ds)\
        #       + dt*((1-theta)*f_form + theta*f_form1)

        rhs = inner(v_p, alp_n) * dx + inner(v_q, alq_n) * dx \
              + dt * (1 - theta) * j_operator(v_p, v_q, ep_n, eq_n) \
              + dt * ((1 - theta) * f_form + theta * f_form1)

        b = assemble(rhs, bcs=bcs)

        t += dt
        solve(A, e_n1, b, solver_parameters=param)
        ep_n1, eqD_n1, eq12_n1 = e_n1.split()

        eq_n1 = as_tensor([[eqD_n1[0], eq12_n1],
                          [eq12_n1, eqD_n1[1]]
                          ])

        w_n1.assign(w_n + dt/2*(ep_n + ep_n1))

        e_n.assign(e_n1)
        w_n.assign(w_n1)

        w_atP[i] = w_n1.at(Ppoint)
        v_atP[i] = ep_n1.at(Ppoint)
        t_.assign(t)

        # w_err_H1[i] = np.sqrt(assemble(dot(w_n1-w_exact, w_n1-w_exact) * dx
        #                      + dot(grad(w_n1)-grad_wex, grad(w_n1)-grad_wex) * dx))
        v_err_H1[i] = np.sqrt(assemble(dot(ep_n1 - v_exact, ep_n1 - v_exact) * dx
                                       + dot(grad(ep_n1) - grad_vex, grad(ep_n1) - grad_vex) * dx))
        v_err_L2[i] = np.sqrt(assemble(dot(ep_n1 - v_exact, ep_n1 - v_exact) * dx))

        sig_err_div[i] = np.sqrt(assemble(inner(eq_n1 - sigma_ex, eq_n1 - sigma_ex) * dx
                                          + inner(div(eq_n1 - sigma_ex), div(eq_n1 - sigma_ex)) * dx))
        sig_err_L2[i] = np.sqrt(assemble(inner(eq_n1 - sigma_ex, eq_n1 - sigma_ex) * dx))

    plt.figure()
    wst_atP = interpolate(wst, Vp).at(Ppoint)
    vex_atP = wst_atP * beta * np.cos(beta * t_vec)
    plt.plot(t_vec, v_atP, 'r-', label=r'approx $v$')
    plt.plot(t_vec, vex_atP, 'b-', label=r'exact $v$')
    plt.xlabel(r'Time [s]')
    plt.title(r'Displacement at' + str(Ppoint))
    plt.legend()
    # plt.show()

    # v_err_last = w_err_H1[-1]
    # v_err_max = max(w_err_H1)
    # v_err_quad = np.sqrt(np.sum(dt * np.power(w_err_H1, 2)))

    v_err_last = v_err_H1[-1]
    v_err_max = max(v_err_H1)
    v_err_quad = np.sqrt(np.sum(dt * np.power(v_err_H1, 2)))

    # v_err_last = v_err_L2[-1]
    # v_err_max = max(v_err_L2)
    # v_err_quad = np.sqrt(np.sum(dt * np.power(v_err_L2, 2)))

    # sig_err_last = sig_err_div[-1]
    # sig_err_max = max(sig_err_div)
    # sig_err_quad = np.sqrt(np.sum(dt * np.power(sig_err_div, 2)))

    sig_err_last = sig_err_L2[-1]
    sig_err_max = max(sig_err_L2)
    sig_err_quad = np.sqrt(np.sum(dt * np.power(sig_err_L2, 2)))

    return v_err_last, v_err_max, v_err_quad, sig_err_last, sig_err_max, sig_err_quad


n_h = 2
n1_vec = np.array([2**(i+2) for i in range(n_h)])
n2_vec = np.array([2**(i+1) for i in range(n_h)])
h1_vec = 1./n1_vec
h2_vec = 1./n2_vec

v_err_r1 = np.zeros((n_h,))
v_errInf_r1 = np.zeros((n_h,))
v_errQuad_r1 = np.zeros((n_h,))

v_err_r2 = np.zeros((n_h,))
v_errInf_r2 = np.zeros((n_h,))
v_errQuad_r2 = np.zeros((n_h,))

v_err_r3 = np.zeros((n_h,))
v_errInf_r3 = np.zeros((n_h,))
v_errQuad_r3 = np.zeros((n_h,))

v_r1_atF = np.zeros((n_h-1,))
v_r1_max = np.zeros((n_h-1,))
v_r1_L2 = np.zeros((n_h-1,))

v_r2_atF = np.zeros((n_h-1,))
v_r2_max = np.zeros((n_h-1,))
v_r2_L2 = np.zeros((n_h-1,))

v_r3_atF = np.zeros((n_h-1,))
v_r3_max = np.zeros((n_h-1,))
v_r3_L2 = np.zeros((n_h-1,))

sig_err_r1 = np.zeros((n_h,))
sig_errInf_r1 = np.zeros((n_h,))
sig_errQuad_r1 = np.zeros((n_h,))

sig_err_r2 = np.zeros((n_h,))
sig_errInf_r2 = np.zeros((n_h,))
sig_errQuad_r2 = np.zeros((n_h,))

sig_err_r3 = np.zeros((n_h,))
sig_errInf_r3 = np.zeros((n_h,))
sig_errQuad_r3 = np.zeros((n_h,))

sig_r1_atF = np.zeros((n_h-1,))
sig_r1_max = np.zeros((n_h-1,))
sig_r1_L2 = np.zeros((n_h-1,))

sig_r2_atF = np.zeros((n_h-1,))
sig_r2_max = np.zeros((n_h-1,))
sig_r2_L2 = np.zeros((n_h-1,))

sig_r3_atF = np.zeros((n_h-1,))
sig_r3_max = np.zeros((n_h-1,))
sig_r3_L2 = np.zeros((n_h-1,))


for i in range(n_h):
    v_err_r1[i], v_errInf_r1[i], v_errQuad_r1[i],\
    sig_err_r1[i], sig_errInf_r1[i], sig_errQuad_r1[i] = compute_err(n1_vec[i], 1)
    v_err_r2[i], v_errInf_r2[i], v_errQuad_r2[i], sig_err_r2[i],\
    sig_errInf_r2[i], sig_errQuad_r2[i] = compute_err(n1_vec[i], 2)
    v_err_r3[i], v_errInf_r3[i], v_errQuad_r3[i], sig_err_r3[i],\
    sig_errInf_r3[i], sig_errQuad_r3[i] = compute_err(n2_vec[i], 3)

    if i>0:
        v_r1_atF[i-1] = np.log(v_err_r1[i]/v_err_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
        v_r1_max[i-1] = np.log(v_errInf_r1[i]/v_errInf_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
        v_r1_L2[i-1] = np.log(v_errQuad_r1[i]/v_errQuad_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])

        v_r2_atF[i-1] = np.log(v_err_r2[i]/v_err_r2[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
        v_r2_max[i-1] = np.log(v_errInf_r2[i]/v_errInf_r2[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
        v_r2_L2[i-1] = np.log(v_errQuad_r2[i]/v_errQuad_r2[i-1])/np.log(h1_vec[i]/h1_vec[i-1])

        v_r3_atF[i-1] = np.log(v_err_r3[i]/v_err_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])
        v_r3_max[i-1] = np.log(v_errInf_r3[i]/v_errInf_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])
        v_r3_L2[i-1] = np.log(v_errQuad_r3[i]/v_errQuad_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])

        sig_r1_atF[i - 1] = np.log(sig_err_r1[i] / sig_err_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        sig_r1_max[i - 1] = np.log(sig_errInf_r1[i] / sig_errInf_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        sig_r1_L2[i - 1] = np.log(sig_errQuad_r1[i] / sig_errQuad_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

        sig_r2_atF[i - 1] = np.log(sig_err_r2[i] / sig_err_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        sig_r2_max[i - 1] = np.log(sig_errInf_r2[i] / sig_errInf_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        sig_r2_L2[i - 1] = np.log(sig_errQuad_r2[i] / sig_errQuad_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

        sig_r3_atF[i - 1] = np.log(sig_err_r3[i] / sig_err_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])
        sig_r3_max[i - 1] = np.log(sig_errInf_r3[i] / sig_errInf_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])
        sig_r3_L2[i - 1] = np.log(sig_errQuad_r3[i] / sig_errQuad_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])

path_res = "./convergence_results_kirchhoff/"
if save_res:
    np.save(path_res + bc_input + "_h1", h1_vec)
    np.save(path_res + bc_input + "_h2", h1_vec)
    np.save(path_res + bc_input + "_h3", h2_vec)

    np.save(path_res + bc_input + "_v_errF_r1", v_err_r1)
    np.save(path_res + bc_input + "_v_errInf_r1", v_errInf_r1)
    np.save(path_res + bc_input + "_v_errQuad_r1", v_errQuad_r1)

    np.save(path_res + bc_input + "_v_errF_r2", v_err_r2)
    np.save(path_res + bc_input + "_v_errInf_r2", v_errInf_r2)
    np.save(path_res + bc_input + "_v_errQuad_r2", v_errQuad_r2)

    np.save(path_res + bc_input + "_v_errF_r3", v_err_r3)
    np.save(path_res + bc_input + "_v_errInf_r3", v_errInf_r3)
    np.save(path_res + bc_input + "_v_errQuad_r3", v_errQuad_r3)

    np.save(path_res + bc_input + "_sig_errF_r1", sig_err_r1)
    np.save(path_res + bc_input + "_sig_errInf_r1", sig_errInf_r1)
    np.save(path_res + bc_input + "_sig_errQuad_r1", sig_errQuad_r1)

    np.save(path_res + bc_input + "_sig_errF_r2", sig_err_r2)
    np.save(path_res + bc_input + "_sig_errInf_r2", sig_errInf_r2)
    np.save(path_res + bc_input + "_sig_errQuad_r2", sig_errQuad_r2)

    np.save(path_res + bc_input + "_sig_errF_r3", sig_err_r3)
    np.save(path_res + bc_input + "_sig_errInf_r3", sig_errInf_r3)
    np.save(path_res + bc_input + "_sig_errQuad_r3", sig_errQuad_r3)


v_r1int_atF = np.polyfit(np.log(h1_vec), np.log(v_err_r1), 1)[0]
v_r1int_max = np.polyfit(np.log(h1_vec), np.log(v_errInf_r1), 1)[0]
v_r1int_L2 = np.polyfit(np.log(h1_vec), np.log(v_errQuad_r1), 1)[0]

print("Estimated order of convergence r=1 for v at T fin: " + str(v_r1_atF))
print("Interpolated order of convergence r=1 for v at T fin: " + str(v_r1int_atF))
print("Estimated order of convergence r=1 for v Linf: " + str(v_r1_max))
print("Interpolated order of convergence r=1 for v Linf: " + str(v_r1int_max))
print("Estimated order of convergence r=1 for v L2: " + str(v_r1_L2))
print("Interpolated order of convergence r=1 for v L2: " + str(v_r1int_L2))
print("")

v_r2int = np.polyfit(np.log(h1_vec), np.log(v_err_r2), 1)[0]
v_r2int_max = np.polyfit(np.log(h1_vec), np.log(v_errInf_r2), 1)[0]
v_r2int_L2 = np.polyfit(np.log(h1_vec), np.log(v_errQuad_r2), 1)[0]

print("Estimated order of convergence r=2 for v at T fin: " + str(v_r2_atF))
print("Interpolated order of convergence r=2 for v at T fin: " + str(v_r2int))
print("Estimated order of convergence r=2 for v Linf: " + str(v_r2_max))
print("Interpolated order of convergence r=2 for v Linf: " + str(v_r2int_max))
print("Estimated order of convergence r=2 for v L2: " + str(v_r2_L2))
print("Interpolated order of convergence r=2 for v L2: " + str(v_r2int_L2))
print("")

v_r3int = np.polyfit(np.log(h2_vec), np.log(v_err_r3), 1)[0]
v_r3int_max = np.polyfit(np.log(h2_vec), np.log(v_errInf_r3), 1)[0]
v_r3int_L2 = np.polyfit(np.log(h2_vec), np.log(v_errQuad_r3), 1)[0]

print("Estimated order of convergence r=3 for v at T fin: " + str(v_r3_atF))
print("Interpolated order of convergence r=3 for v at T fin: " + str(v_r3int))
print("Estimated order of convergence r=3 for v Linf: " + str(v_r3_max))
print("Interpolated order of convergence r=3 for v Linf: " + str(v_r3int_max))
print("Estimated order of convergence r=3 for v L2: " + str(v_r3_L2))
print("Interpolated order of convergence r=3 for v L2: " + str(v_r3int_L2))
print("")

plt.figure()

# plt.plot(np.log(h1_vec), np.log(v_r1_atF), ':o', label='HHJ 1')
plt.plot(np.log(h1_vec), np.log(v_errInf_r1), '-.+', label='HHJ 1 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(v_errQuad_r1), '--*', label='HHJ 1 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec), '-v', label=r'$h$')

# plt.plot(np.log(h1_vec), np.log(v_r2_atF), ':o', label='HHJ 2')
plt.plot(np.log(h1_vec), np.log(v_errInf_r2), '-.+', label='HHJ 2 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(v_errQuad_r2), '--*', label='HHJ 2 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec**2), '-v', label=r'$h^2$')

# plt.plot(np.log(h2_vec), np.log(v_r3_atF), ':o', label='HHJ 3')
plt.plot(np.log(h2_vec), np.log(v_errInf_r3), '-.+', label='HHJ 3 $L^\infty$')
plt.plot(np.log(h2_vec), np.log(v_errQuad_r3), '--*', label='HHJ 3 $L^2$')
plt.plot(np.log(h2_vec), np.log(h2_vec**3), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log(Error Velocity)')
plt.title(r'Velocity Error vs Mesh size')
plt.legend()
path_fig = "/home/a.brugnoli/Plots/Python/Plots/Kirchhoff_plots/Convergence/firedrake/"
if save_res:
    plt.savefig(path_fig  + bc_input + "_vel.eps", format="eps")

sig_r1int_atF = np.polyfit(np.log(h1_vec), np.log(sig_err_r1), 1)[0]
sig_r1int_max = np.polyfit(np.log(h1_vec), np.log(sig_errInf_r1), 1)[0]
sig_r1int_L2 = np.polyfit(np.log(h1_vec), np.log(sig_errQuad_r1), 1)[0]

print("Estimated order of convergence r=1 for sigma at T fin: " + str(sig_r1_atF))
print("Interpolated order of convergence r=1 for sigma at T fin: " + str(sig_r1int_atF))
print("Estimated order of convergence r=1 for sigma Linf: " + str(sig_r1_max))
print("Interpolated order of convergence r=1 for sigma Linf: " + str(sig_r1int_max))
print("Estimated order of convergence r=1 for sigma L2: " + str(sig_r1_L2))
print("Interpolated order of convergence r=1 for sigma L2: " + str(sig_r1int_L2))
print("")

sig_r2int = np.polyfit(np.log(h1_vec), np.log(sig_err_r2), 1)[0]
sig_r2int_max = np.polyfit(np.log(h1_vec), np.log(sig_errInf_r2), 1)[0]
sig_r2int_L2 = np.polyfit(np.log(h1_vec), np.log(sig_errQuad_r2), 1)[0]

print("Estimated order of convergence r=2 for sigma at T fin: " + str(sig_r2_atF))
print("Interpolated order of convergence r=2 for sigma at T fin: " + str(sig_r2int))
print("Estimated order of convergence r=2 for sigma Linf: " + str(sig_r2_max))
print("Interpolated order of convergence r=2 for sigma Linf: " + str(sig_r2int_max))
print("Estimated order of convergence r=2 for sigma L2: " + str(sig_r2_L2))
print("Interpolated order of convergence r=2 for sigma L2: " + str(sig_r2int_L2))
print("")

sig_r3int = np.polyfit(np.log(h2_vec), np.log(sig_err_r3), 1)[0]
sig_r3int_max = np.polyfit(np.log(h2_vec), np.log(sig_errInf_r3), 1)[0]
sig_r3int_L2 = np.polyfit(np.log(h2_vec), np.log(sig_errQuad_r3), 1)[0]

print("Estimated order of convergence r=3 for sigma at T fin: " + str(sig_r3_atF))
print("Interpolated order of convergence r=3 for sigma at T fin: " + str(sig_r3int))
print("Estimated order of convergence r=3 for sigma Linf: " + str(sig_r3_max))
print("Interpolated order of convergence r=3 for sigma Linf: " + str(sig_r3int_max))
print("Estimated order of convergence r=3 for sigma L2: " + str(sig_r3_L2))
print("Interpolated order of convergence r=3 for sigma L2: " + str(sig_r3int_L2))
print("")

plt.figure()

# plt.plot(np.log(h1_vec), np.log(sig_r1_atF), ':o', label='HHJ 1')
plt.plot(np.log(h1_vec), np.log(sig_errInf_r1), '-.+', label='HHJ 1 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(sig_errQuad_r1), '--*', label='HHJ 1 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec), '-v', label=r'$h$')

# plt.plot(np.log(h1_vec), np.log(sig_r2_atF), ':o', label='HHJ 2')
plt.plot(np.log(h1_vec), np.log(sig_errInf_r2), '-.+', label='HHJ 2 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(sig_errQuad_r2), '--*', label='HHJ 2 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec**2), '-v', label=r'$h^2$')

# plt.plot(np.log(h2_vec), np.log(sig_r3_atF), ':o', label='HHJ 3')
plt.plot(np.log(h2_vec), np.log(sig_errInf_r3), '-.+', label='HHJ 3 $L^\infty$')
plt.plot(np.log(h2_vec), np.log(sig_errQuad_r3), '--*', label='HHJ 3 $L^2$')
plt.plot(np.log(h2_vec), np.log(h2_vec**3), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log(Error Stress)')
plt.title(r'Stress Error vs Mesh size')
plt.legend()
if save_res:
    plt.savefig(path_fig + bc_input + "_sigma.eps", format="eps")
plt.show()