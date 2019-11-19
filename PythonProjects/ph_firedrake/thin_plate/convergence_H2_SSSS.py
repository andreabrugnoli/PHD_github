# Convergence test for HHJ

from firedrake import *
import numpy as np
import scipy as sp
np.set_printoptions(threshold=np.inf)
from math import pi, floor

import matplotlib
import matplotlib.pyplot as plt

import scipy.linalg as la
import scipy.sparse as spa
import scipy.sparse.linalg as sp_la
matplotlib.rcParams['text.usetex'] = True
save_res = False
case = 'SSSS_H2'
bc_input = 'SSSS'

def compute_err(n, r):

    h_mesh = 1/n

    E = Constant(136 * 10**9) # Pa
    rho = Constant(5600)  # kg/m^3
    nu = Constant(0.3)
    h = Constant(0.001)

    Lx = 1
    Ly = 1

    D = Constant(E * h ** 3 / (1 - nu ** 2) / 12)
    fl_rot = Constant(12 / (E * h ** 3))
    # Useful Matrices

    # Operators and functions
    def gradSym(u):
        return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
        # return sym(nabla_grad(u))

    def bending_curv(momenta):
        kappa = fl_rot * ((1+nu)*momenta - nu * Identity(2) * tr(momenta))
        return kappa

    def j_operator(v_p, v_q, e_p, e_q):

        j_form = inner(v_q, grad(grad(e_p))) * dx \
                -inner(grad(grad(v_p)), e_q) * dx

        return j_form

    # The unit square mesh is divided in :math:`N\times N` quadrilaterals::

    mesh = RectangleMesh(n, n, Lx, Lx, quadrilateral=False)

    # Domain, Subdomains, Boundary, Suboundaries

    # Finite element defition

    name_FEp = 'Bell'
    name_FEq = 'Bell'
    deg_q = 3

    if name_FEp == 'Morley':
        deg_p = 2
    elif name_FEp == 'Hermite':
        deg_p = 3
    elif name_FEp == 'Argyris' or name_FEp == 'Bell':
        deg_p = 5

    if name_FEq == 'Morley':
        deg_q = 2
    elif name_FEq == 'Hermite':
        deg_q = 3
    elif name_FEq == 'Argyris' or name_FEq == 'Bell':
        deg_q = 5

    Vp = FunctionSpace(mesh, name_FEp, deg_p)
    Vq = VectorFunctionSpace(mesh, name_FEq, deg_q, dim=3)
    V = Vp * Vq

    n_V = V.dim()
    print(n_V)

    v = TestFunction(V)
    v_p, vq_vec = split(v)

    e_v = TrialFunction(V)
    e_p, eq_vec = split(e_v)

    v_q = as_tensor([[vq_vec[0], vq_vec[1]],
                     [vq_vec[1], vq_vec[2]]])

    e_q = as_tensor([[eq_vec[0], eq_vec[1]],
                     [eq_vec[1], eq_vec[2]]])

    al_p = rho * h * e_p
    al_q = bending_curv(e_q)

    dx = Measure('dx')
    ds = Measure('ds')
    dS = Measure("dS")

    m_form = inner(v_p, al_p) * dx + inner(v_q, al_q) * dx

    n_ver = FacetNormal(mesh)
    s_ver = as_vector([-n_ver[1], n_ver[0]])

    # e_mnn = inner(e_q, outer(n_ver, n_ver))
    # v_mnn = inner(v_q, outer(n_ver, n_ver))
    #
    # e_mns = inner(e_q, outer(n_ver, s_ver))
    # v_mns = inner(v_q, outer(n_ver, s_ver))

    j_form = j_operator(v_p, v_q, e_p, e_q)

    bc_1, bc_3, bc_2, bc_4 = bc_input

    bc_dict = {1: bc_1, 2: bc_2, 3: bc_3, 4: bc_4}

    n = FacetNormal(mesh)
    # s = as_vector([-n[1], n[0]])

    V_qn = FunctionSpace(mesh, 'Lagrange', 1)
    V_Mnn = FunctionSpace(mesh, 'Lagrange', 1)

    Vu = V_qn * V_Mnn

    q_n, M_nn = TrialFunction(Vu)

    v_omn = dot(grad(v_p), n)
    g_vec = []
    for key, val in bc_dict.items():
        if val == 'C':
            g_vec.append(v_p * q_n * ds(key) + v_omn * M_nn * ds(key))
        elif val == 'S':
            g_vec.append(v_p * q_n * ds(key))

    g_lmb = sum(g_vec)

    assert g_lmb

    petsc_g = assemble(g_lmb, mat_type='aij').M.handle
    G_lmb = sp.sparse.csr_matrix(petsc_g.getValuesCSR()[::-1])
    rows, cols = spa.csr_matrix.nonzero(G_lmb)
    set_cols = np.array(list(set(cols)))

    n_lmb = len(set_cols)
    G = np.zeros((n_V, n_lmb))
    for r, c in zip(rows, cols):
        ind_col = np.where(set_cols == c)[0]
        G[r, ind_col] = G_lmb[r, c]

    G_ortho = la.null_space(G.T).T
    G_ortho = spa.csr_matrix(G_ortho)

    dt = 0.1 * h_mesh
    theta = 0.5

    A_form = m_form - dt * theta * j_form
    A = sp.sparse.csr_matrix(assemble(A_form, mat_type='aij').M.handle.getValuesCSR()[::-1])

    B_form = m_form + dt * (1 - theta) * j_form
    B = sp.sparse.csr_matrix(assemble(B_form, mat_type='aij').M.handle.getValuesCSR()[::-1])

    A_til = G_ortho.dot(A.dot(G_ortho.transpose()))
    B_til = G_ortho.dot(B.dot(G_ortho.transpose()))

    t = 0.
    t_ = Constant(t)
    t_fin = 1        # total simulation time
    x = mesh.coordinates

    beta = 1
    w_exact = sin(pi*x[0]/Lx)*sin(pi*x[1]/Ly)*sin(beta*t_)
    grad_wex = as_vector([pi/Lx*cos(pi*x[0]/Lx)*sin(pi*x[1]/Ly)*sin(beta*t_),
                           pi/Ly*sin(pi*x[0]/Lx)*cos(pi*x[1]/Ly)*sin(beta*t_)])

    v_exact = beta * sin(pi*x[0]/Lx)*sin(pi*x[1]/Ly)*cos(beta*t_)
    grad_vex = as_vector([beta * pi / Lx * cos(pi * x[0] / Lx) * sin(pi * x[1] / Ly) * cos(beta * t_),
                          beta * pi / Ly * sin(pi * x[0] / Lx) * cos(pi * x[1] / Ly) * cos(beta * t_)])

    dxx_wex = - (pi/Lx)**2*sin(pi*x[0]/Lx)*sin(pi*x[1]/Ly)*sin(beta*t_)
    dyy_wex = - (pi/Ly)**2*sin(pi*x[0]/Lx)*sin(pi*x[1]/Ly)*sin(beta*t_)
    dxy_wex = pi**2/(Lx*Ly)*cos(pi*x[0]/Lx)*cos(pi*x[1]/Ly)*sin(beta*t_)

    sigma_ex = as_tensor([[D * (dxx_wex + nu * dyy_wex), D * (1 - nu) * dxy_wex],
                          [D * (1 - nu) * dxy_wex, D * (dyy_wex + nu * dxx_wex)]])

    force_xy = sin(pi*x[0]/Lx)*sin(pi*x[1]/Ly)*(D *((pi/Lx)**2 + (pi/Ly)**2)**2 - rho*h*beta**2)

    f_xy = assemble(v_p*force_xy*dx).vector().get_local()
    fxy_til = G_ortho.dot(f_xy)

    e_n1 = Function(V, name="e next")
    e_n = Function(V,  name="e old")
    w_n1 = Function(Vp, name="w old")
    w_n = Function(Vp, name="w next")

    e_n.sub(0).assign(project(v_exact, Vp))

    ep_n, eq_vec_n = e_n.split()
    eq_n = as_tensor([[eq_vec_n[0], eq_vec_n[1]],
                      [eq_vec_n[1], eq_vec_n[2]]])

    w_n.assign(Constant(0.0))

    en_til = sp_la.lsqr(G_ortho.transpose(), e_n.vector().get_local())[0]

    n_t = int(floor(t_fin/dt) + 1)

    w_err_H1 = np.zeros((n_t,))
    v_err_H1 = np.zeros((n_t,))
    sig_err_L2 = np.zeros((n_t,))

    # Ppoint = (Lx/14, Ly/3)
    # w_atP = np.zeros((n_t,))
    # v_atP = np.zeros((n_t,))
    # v_atP[0] = ep_n.at(Ppoint)

    # w_err_H1[0] = np.sqrt(assemble(dot(w_n-w_exact, w_n-w_exact) *dx
    #                      + dot(grad(w_n) - grad_wex, grad(w_n) - grad_wex) * dx))
    v_err_H1[0] = np.sqrt(assemble(dot(ep_n - v_exact, ep_n - v_exact) * dx
                                   + dot(grad(ep_n) - grad_vex, grad(ep_n) - grad_vex) * dx))

    sig_err_L2[0] = np.sqrt(assemble(inner(eq_n - sigma_ex, eq_n - sigma_ex) * dx))

    t_vec = np.linspace(0, t_fin, num=n_t)

    param = {"ksp_type": "preonly", "pc_type": "lu"}

    # print(e_n.vector().get_local())
    for i in range(1, n_t):

        # t_.assign(t)
        b_til = B_til.dot(en_til) + dt*fxy_til*((1-theta)*np.sin(t) + theta*np.sin(t+dt))

        t += dt
        en1_til = sp_la.spsolve(A_til, b_til)

        e_n1.vector().set_local((G_ortho.transpose()).dot(en1_til))
        ep_n1, eq_vec_n1 = e_n1.split()

        eq_n1 = as_tensor([[eq_vec_n1[0], eq_vec_n1[1]],
                          [eq_vec_n1[1], eq_vec_n1[2]]])

        w_n1.assign(w_n + dt/2*(ep_n + ep_n1))
        w_n.assign(w_n1)

        e_n.assign(e_n1)

        en_til = en1_til

        # w_atP[i] = w_n1.at(Ppoint)
        # v_atP[i] = ep_n1.at(Ppoint)
        t_.assign(t)

        # w_err_H1[i] = np.sqrt(assemble(dot(w_n1-w_exact, w_n1-w_exact) * dx
        #                      + dot(grad(w_n1)-grad_wex, grad(w_n1)-grad_wex) * dx))
        v_err_H1[i] = np.sqrt(assemble(dot(ep_n1 - v_exact, ep_n1 - v_exact) * dx
                                       + dot(grad(ep_n1) - grad_vex, grad(ep_n1) - grad_vex) * dx))

        sig_err_L2[i] = np.sqrt(assemble(inner(eq_n1 - sigma_ex, eq_n1 - sigma_ex) * dx))

    # plt.figure()
    # # plt.plot(t_vec, w_atP, 'r-', label=r'approx $w$')
    # # plt.plot(t_vec, np.sin(pi*Ppoint[0]/Lx)*np.sin(pi*Ppoint[1]/Ly)*np.sin(beta*t_vec), 'b-', label=r'exact $w$')
    # plt.plot(t_vec, v_atP, 'r-', label=r'approx $v$')
    # plt.plot(t_vec, beta * np.sin(pi*Ppoint[0]/Lx)*np.sin(pi*Ppoint[1]/Ly) * np.cos(beta * t_vec), 'b-', label=r'exact $v$')
    # plt.xlabel(r'Time [s]')
    # plt.title(r'Displacement at' + str(Ppoint))
    # plt.legend()
    # plt.show()

    # v_err_last = w_err_H1[-1]
    # v_err_max = max(w_err_H1)
    # v_err_quad = np.sqrt(np.sum(dt * np.power(w_err_H1, 2)))

    v_err_last = v_err_H1[-1]
    v_err_max = max(v_err_H1)
    v_err_quad = np.sqrt(np.sum(dt * np.power(v_err_H1, 2)))

    sig_err_last = sig_err_L2[-1]
    sig_err_max = max(sig_err_L2)
    sig_err_quad = np.sqrt(np.sum(dt * np.power(sig_err_L2, 2)))

    return v_err_last, v_err_max, v_err_quad, sig_err_last, sig_err_max, sig_err_quad


n_h = 5
n1_vec = np.array([2**(i) for i in range(n_h)])
n2_vec = np.array([2**(i) for i in range(n_h)])
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
    # v_err_r2[i], v_errInf_r2[i], v_errQuad_r2[i], sig_err_r2[i],\
    # sig_errInf_r2[i], sig_errQuad_r2[i] = compute_err(n1_vec[i], 2)
    # v_err_r3[i], v_errInf_r3[i], v_errQuad_r3[i], sig_err_r3[i],\
    # sig_errInf_r3[i], sig_errQuad_r3[i] = compute_err(n2_vec[i], 3)

    if i>0:
        v_r1_atF[i-1] = np.log(v_err_r1[i]/v_err_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
        v_r1_max[i-1] = np.log(v_errInf_r1[i]/v_errInf_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
        v_r1_L2[i-1] = np.log(v_errQuad_r1[i]/v_errQuad_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])

        # v_r2_atF[i-1] = np.log(v_err_r2[i]/v_err_r2[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
        # v_r2_max[i-1] = np.log(v_errInf_r2[i]/v_errInf_r2[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
        # v_r2_L2[i-1] = np.log(v_errQuad_r2[i]/v_errQuad_r2[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
        #
        # v_r3_atF[i-1] = np.log(v_err_r3[i]/v_err_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])
        # v_r3_max[i-1] = np.log(v_errInf_r3[i]/v_errInf_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])
        # v_r3_L2[i-1] = np.log(v_errQuad_r3[i]/v_errQuad_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])

        sig_r1_atF[i - 1] = np.log(sig_err_r1[i] / sig_err_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        sig_r1_max[i - 1] = np.log(sig_errInf_r1[i] / sig_errInf_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        sig_r1_L2[i - 1] = np.log(sig_errQuad_r1[i] / sig_errQuad_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

        # sig_r2_atF[i - 1] = np.log(sig_err_r2[i] / sig_err_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        # sig_r2_max[i - 1] = np.log(sig_errInf_r2[i] / sig_errInf_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        # sig_r2_L2[i - 1] = np.log(sig_errQuad_r2[i] / sig_errQuad_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        #
        # sig_r3_atF[i - 1] = np.log(sig_err_r3[i] / sig_err_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])
        # sig_r3_max[i - 1] = np.log(sig_errInf_r3[i] / sig_errInf_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])
        # sig_r3_L2[i - 1] = np.log(sig_errQuad_r3[i] / sig_errQuad_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])

if save_res:
    np.save("./convergence_results/" + case + "_h1", h1_vec)
    np.save("./convergence_results/" + case + "_h2", h1_vec)
    np.save("./convergence_results/" + case + "_h3", h2_vec)

    np.save("./convergence_results/" + case + "_v_errF_r1", v_err_r1)
    np.save("./convergence_results/" + case + "_v_errInf_r1", v_errInf_r1)
    np.save("./convergence_results/" + case + "_v_errQuad_r1", v_errQuad_r1)

    np.save("./convergence_results/" + case + "_v_errF_r2", v_err_r2)
    np.save("./convergence_results/" + case + "_v_errInf_r2", v_errInf_r2)
    np.save("./convergence_results/" + case + "_v_errQuad_r2", v_errQuad_r2)

    np.save("./convergence_results/" + case + "_v_errF_r3", v_err_r3)
    np.save("./convergence_results/" + case + "_v_errInf_r3", v_errInf_r3)
    np.save("./convergence_results/" + case + "_v_errQuad_r3", v_errQuad_r3)

    np.save("./convergence_results/" + case + "_sig_errF_r1", sig_err_r1)
    np.save("./convergence_results/" + case + "_sig_errInf_r1", sig_errInf_r1)
    np.save("./convergence_results/" + case + "_sig_errQuad_r1", sig_errQuad_r1)

    np.save("./convergence_results/" + case + "_sig_errF_r2", sig_err_r2)
    np.save("./convergence_results/" + case + "_sig_errInf_r2", sig_errInf_r2)
    np.save("./convergence_results/" + case + "_sig_errQuad_r2", sig_errQuad_r2)

    np.save("./convergence_results/" + case + "_sig_errF_r3", sig_err_r3)
    np.save("./convergence_results/" + case + "_sig_errInf_r3", sig_errInf_r3)
    np.save("./convergence_results/" + case + "_sig_errQuad_r3", sig_errQuad_r3)

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

# v_r2int = np.polyfit(np.log(h1_vec), np.log(v_err_r2), 1)[0]
# v_r2int_max = np.polyfit(np.log(h1_vec), np.log(v_errInf_r2), 1)[0]
# v_r2int_L2 = np.polyfit(np.log(h1_vec), np.log(v_errQuad_r2), 1)[0]
#
# print("Estimated order of convergence r=2 for v at T fin: " + str(v_r2_atF))
# print("Interpolated order of convergence r=2 for v at T fin: " + str(v_r2int))
# print("Estimated order of convergence r=2 for v Linf: " + str(v_r2_max))
# print("Interpolated order of convergence r=2 for v Linf: " + str(v_r2int_max))
# print("Estimated order of convergence r=2 for v L2: " + str(v_r2_L2))
# print("Interpolated order of convergence r=2 for v L2: " + str(v_r2int_L2))
# print("")
#
# v_r3int = np.polyfit(np.log(h2_vec), np.log(v_err_r3), 1)[0]
# v_r3int_max = np.polyfit(np.log(h2_vec), np.log(v_errInf_r3), 1)[0]
# v_r3int_L2 = np.polyfit(np.log(h2_vec), np.log(v_errQuad_r3), 1)[0]
#
# print("Estimated order of convergence r=3 for v at T fin: " + str(v_r3_atF))
# print("Interpolated order of convergence r=3 for v at T fin: " + str(v_r3int))
# print("Estimated order of convergence r=3 for v Linf: " + str(v_r3_max))
# print("Interpolated order of convergence r=3 for v Linf: " + str(v_r3int_max))
# print("Estimated order of convergence r=3 for v L2: " + str(v_r3_L2))
# print("Interpolated order of convergence r=3 for v L2: " + str(v_r3int_L2))
# print("")

plt.figure()

# plt.plot(np.log(h1_vec), np.log(v_r1_atF), ':o', label='HHJ 1')
plt.plot(np.log(h1_vec), np.log(v_errInf_r1), '-.+', label='HHJ 1 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(v_errQuad_r1), '--*', label='HHJ 1 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec), '-v', label=r'$h$')

# # plt.plot(np.log(h1_vec), np.log(v_r2_atF), ':o', label='HHJ 2')
# plt.plot(np.log(h1_vec), np.log(v_errInf_r2), '-.+', label='HHJ 2 $L^\infty$')
# plt.plot(np.log(h1_vec), np.log(v_errQuad_r2), '--*', label='HHJ 2 $L^2$')
# plt.plot(np.log(h1_vec), np.log(h1_vec**2), '-v', label=r'$h^2$')
#
# # plt.plot(np.log(h2_vec), np.log(v_r3_atF), ':o', label='HHJ 3')
# plt.plot(np.log(h2_vec), np.log(v_errInf_r3), '-.+', label='HHJ 3 $L^\infty$')
# plt.plot(np.log(h2_vec), np.log(v_errQuad_r3), '--*', label='HHJ 3 $L^2$')
# plt.plot(np.log(h2_vec), np.log(h2_vec**3), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log(Error Velocity)')
plt.title(r'Velocity Error vs Mesh size')
plt.legend()
path_fig = "/home/a.brugnoli/Plots_Videos/Python/Plots/Kirchhoff_plots/Convergence/firedrake/"
if save_res:
    plt.savefig(path_fig + case + "_vel.eps", format="eps")

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
#
# sig_r2int = np.polyfit(np.log(h1_vec), np.log(sig_err_r2), 1)[0]
# sig_r2int_max = np.polyfit(np.log(h1_vec), np.log(sig_errInf_r2), 1)[0]
# sig_r2int_L2 = np.polyfit(np.log(h1_vec), np.log(sig_errQuad_r2), 1)[0]
#
# print("Estimated order of convergence r=2 for sigma at T fin: " + str(sig_r2_atF))
# print("Interpolated order of convergence r=2 for sigma at T fin: " + str(sig_r2int))
# print("Estimated order of convergence r=2 for sigma Linf: " + str(sig_r2_max))
# print("Interpolated order of convergence r=2 for sigma Linf: " + str(sig_r2int_max))
# print("Estimated order of convergence r=2 for sigma L2: " + str(sig_r2_L2))
# print("Interpolated order of convergence r=2 for sigma L2: " + str(sig_r2int_L2))
# print("")
#
# sig_r3int = np.polyfit(np.log(h2_vec), np.log(sig_err_r3), 1)[0]
# sig_r3int_max = np.polyfit(np.log(h2_vec), np.log(sig_errInf_r3), 1)[0]
# sig_r3int_L2 = np.polyfit(np.log(h2_vec), np.log(sig_errQuad_r3), 1)[0]
#
# print("Estimated order of convergence r=3 for sigma at T fin: " + str(sig_r3_atF))
# print("Interpolated order of convergence r=3 for sigma at T fin: " + str(sig_r3int))
# print("Estimated order of convergence r=3 for sigma Linf: " + str(sig_r3_max))
# print("Interpolated order of convergence r=3 for sigma Linf: " + str(sig_r3int_max))
# print("Estimated order of convergence r=3 for sigma L2: " + str(sig_r3_L2))
# print("Interpolated order of convergence r=3 for sigma L2: " + str(sig_r3int_L2))
# print("")

plt.figure()

# plt.plot(np.log(h1_vec), np.log(sig_r1_atF), ':o', label='HHJ 1')
plt.plot(np.log(h1_vec), np.log(sig_errInf_r1), '-.+', label='HHJ 1 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(sig_errQuad_r1), '--*', label='HHJ 1 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec), '-v', label=r'$h$')

# # plt.plot(np.log(h1_vec), np.log(sig_r2_atF), ':o', label='HHJ 2')
# plt.plot(np.log(h1_vec), np.log(sig_errInf_r2), '-.+', label='HHJ 2 $L^\infty$')
# plt.plot(np.log(h1_vec), np.log(sig_errQuad_r2), '--*', label='HHJ 2 $L^2$')
# plt.plot(np.log(h1_vec), np.log(h1_vec**2), '-v', label=r'$h^2$')
#
# # plt.plot(np.log(h2_vec), np.log(sig_r3_atF), ':o', label='HHJ 3')
# plt.plot(np.log(h2_vec), np.log(sig_errInf_r3), '-.+', label='HHJ 3 $L^\infty$')
# plt.plot(np.log(h2_vec), np.log(sig_errQuad_r3), '--*', label='HHJ 3 $L^2$')
# plt.plot(np.log(h2_vec), np.log(h2_vec**3), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log(Error Stress)')
plt.title(r'Stress Error vs Mesh size')
plt.legend()
if save_res:
    plt.savefig(path_fig + case + "_sigma.eps", format="eps")
plt.show()