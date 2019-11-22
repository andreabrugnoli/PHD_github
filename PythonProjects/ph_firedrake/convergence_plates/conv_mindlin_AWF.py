# Convergence test for HHJ

from firedrake import *
import numpy as np
np.set_printoptions(threshold=np.inf)
from math import pi, floor

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['text.usetex'] = True

bc_input = 'CCCC_min'
save_res = False
path_mesh = "/home/a.brugnoli/GitProjects/PythonProjects/ph_firedrake/thin_plate/circle_meshes/"

def compute_err(n, r):

    h_mesh = 1/n

    E = Constant(1)
    nu = Constant(0.3)

    rho = Constant(1)
    k = Constant(5/6)
    h = Constant(1)
    L = 1

    D = E * h ** 3 / (1 - nu ** 2) / 12
    fl_rot = 12 / (E * h ** 3)

    G = E / 2 / (1 + nu)
    F = G * h * k

    # Operators and functions

    def bending_curv(momenta):
        kappa = fl_rot * ((1+nu)*momenta - nu * Identity(2) * tr(momenta))
        return kappa

    def m_operator(v_pw, al_pw, v_pth, al_pth, v_qth, al_qth, e_qth, v_qw, al_qw, v_skw, al_skw):

        m_form = v_pw * al_pw * dx \
                 + dot(v_pth, al_pth) * dx \
                 + inner(v_qth, al_qth) * dx + inner(v_qth, al_skw) * dx \
                 + dot(v_qw, al_qw) * dx \
                 + inner(v_skw, e_qth) * dx

        return m_form

    def j_operator(v_pw, e_pw, v_pth, e_pth, v_qth, e_qth, v_qw, e_qw):

        j_div = v_pw * div(e_qw) * dx
        j_divIP = -div(v_qw) * e_pw * dx

        j_divSym = dot(v_pth, div(e_qth)) * dx
        j_divSymIP = -dot(div(v_qth), e_pth) * dx

        j_Id = dot(v_pth, e_qw) * dx
        j_IdIP = -dot(v_qw, e_pth) * dx

        j_form = j_div + j_divIP + j_divSym + j_divSymIP + j_Id + j_IdIP

        return j_form

    # The unit square mesh is divided in :math:`N\times N` quadrilaterals::

    mesh = Mesh(path_mesh + "circle_n" + str(n) + ".msh")

    # plot(mesh);
    # plt.show()


    # Finite element defition

    V_pw = FunctionSpace(mesh, "DG", r-1)
    V_skw = FunctionSpace(mesh, "DG", r-1)
    V_pth = VectorFunctionSpace(mesh, "DG", r-1)

    V_qth1 = FunctionSpace(mesh, "BDM", r)
    V_qth2 = FunctionSpace(mesh, "BDM", r)
    V_qw = FunctionSpace(mesh, "BDM", r)

    V = MixedFunctionSpace([V_pw, V_skw, V_pth, V_qth1, V_qth2, V_qw])

    n_V = V.dim()
    print(n_V)

    v = TestFunction(V)
    v_pw, v_skw, v_pth, v_qth1, v_qth2, v_qw = split(v)

    e = TrialFunction(V)
    e_pw, e_skw, e_pth, e_qth1, e_qth2, e_qw = split(e)

    v_qth = as_tensor([[v_qth1[0], v_qth1[1]],
                       [v_qth2[0], v_qth2[1]]
                       ])

    e_qth = as_tensor([[e_qth1[0], e_qth1[1]],
                       [e_qth2[0], e_qth2[1]]
                       ])

    al_pw = rho * h * e_pw
    al_pth = (rho * h ** 3) / 12. * e_pth
    al_qth = bending_curv(e_qth)
    al_qw = 1. / F * e_qw

    v_skw = as_tensor([[0, v_skw],
                       [-v_skw, 0]])
    al_skw = as_tensor([[0, e_skw],
                        [-e_skw, 0]])

    # v_skw = skew(v_skw)
    # al_skw = skew(e_skw)

    dx = Measure('dx')
    ds = Measure('ds')

    m_form = m_operator(v_pw, al_pw, v_pth, al_pth, v_qth, al_qth, e_qth, v_qw, al_qw, v_skw, al_skw)
    j_form = j_operator(v_pw, e_pw, v_pth, e_pth, v_qth, e_qth, v_qw, e_qw)

    t = 0
    t_fin = 1        # total simulation time

    t_ = Constant(t)
    t_1 = Constant(t)

    beta = 1 # 4*pi/t_fin
    x, y = SpatialCoordinate(mesh)

    w_st = 1/3*x**3*(x-1)**3*y**3*(y-1)**3 - 2 *h**2/(5*(1-nu))*\
             (y**3*(y-1)**3*x*(x-1)*(5*x**2 - 5*x+1) + x**3*(x-1)**3*y*(y-1)*(5*y**2-5*y+1))
    w_dyn = w_st * sin(beta*t_)

    thx_st = y ** 3 * (y - 1) ** 3 * x ** 2 * (x - 1) ** 2 * (2 * x - 1)
    thy_st = x ** 3 * (x - 1) ** 3 * y ** 2 * (y - 1) ** 2 * (2 * y - 1)

    th_st = as_vector([thx_st, thy_st])

    thx_dyn = thx_st * sin(beta * t_)
    thy_dyn = thy_st * sin(beta * t_)

    f_st = E/(12*(1-nu))*(12*y*(y-1)*(5*x**2-5*x+1)*(2*y**2*(y-1)**2+x*(x-1)*(5*y**2-5*y+1)) + \
                          12*x*(x-1)*(5*y**2-5*y+1)*(2*x**2*(x-1)**2+y*(y-1)*(5*x**2-5*x+1)))

    dt_w = beta * w_st * cos(beta*t_)
    dtt_w = -beta**2 * w_st * sin(beta*t_)

    dt_thx = beta * thx_st * cos(beta * t_)
    dt_thy = beta * thy_st * cos(beta * t_)
    dtt_thx = -beta**2 * thx_st * sin(beta * t_)
    dtt_thy = -beta**2 * thy_st * sin(beta * t_)
    dt_th = as_vector([dt_thx, dt_thy])

    # J, M = PETScMatrix(), PETScMatrix()

    dt = 0.1*h_mesh
    theta = 0.5

    lhs = m_form - dt*theta*j_form
    A = assemble(lhs, bcs=bcs, mat_type='aij')

    e_n1 = Function(V)
    e_n = Function(V)
    w_n1 = Function(Vp)
    w_n = Function(Vp)

    e_n.sub(0).assign(interpolate(v_exact, Vp))

    ep_n, eq_n = e_n.split()

    w_n.assign(Constant(0.0))

    n_t = int(floor(t_fin/dt) + 1)

    v_err_L2 = np.zeros((n_t,))
    sig_err_L2 = np.zeros((n_t,))

    w_atP = np.zeros((n_t,))
    v_atP = np.zeros((n_t,))
    Ppoint = (R/3, R/7)
    v_atP[0] = ep_n.at(Ppoint[0], Ppoint[1])

    # w_err_H1[0] = np.sqrt(assemble(dot(w_n-w_exact, w_n-w_exact) *dx
    #                      + dot(grad(w_n) - grad_wex, grad(w_n) - grad_wex) * dx))
    v_err_H1[0] = np.sqrt(assemble(dot(ep_n - v_exact, ep_n - v_exact) * dx
                         + dot(grad(ep_n) - grad_vex, grad(ep_n) - grad_vex) * dx))

    sig_err_L2[0] = np.sqrt(assemble(inner(eq_n - sigma_ex, eq_n - sigma_ex) * dx))

    t_vec = np.linspace(0, t_fin, num=n_t)

    param = {"ksp_type": "preonly", "pc_type": "lu"}

    for i in range(1, n_t):

        t_.assign(t)
        t_1.assign(t + dt)

        ep_n, eq_n = e_n.split()
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

        solve(A, e_n1, b, solver_parameters=param)

        t += dt

        ep_n, eq_n = e_n.split()
        ep_n1, eq_n1 = e_n1.split()

        w_n1.assign(w_n + dt / 2 * (ep_n + ep_n1))
        w_n.assign(w_n1)

        e_n.assign(e_n1)

        w_atP[i] = w_n1.at(Ppoint[0], Ppoint[1])
        v_atP[i] = ep_n1.at(Ppoint[0], Ppoint[1])

        t_.assign(t)


        # w_err_H1[i] = np.sqrt(assemble(dot(w_n1-w_exact, w_n1-w_exact) * dx
        #                      + dot(grad(w_n1)-grad_wex, grad(w_n1)-grad_wex) * dx))
        v_err_H1[i] = np.sqrt(assemble(dot(ep_n1 - v_exact, ep_n1 - v_exact) * dx
                         + dot(grad(ep_n1) - grad_vex, grad(ep_n1) - grad_vex) * dx))

        sig_err_L2[i] = np.sqrt(assemble(inner(eq_n1 - sigma_ex, eq_n1 - sigma_ex) * dx))

    plt.figure()
    # plt.plot(t_vec, w_atP, 'r-', label=r'approx $w$')
    # plt.plot(t_vec, f_0/(64*D)*(R**2-(Ppoint[0]**2+Ppoint[1]**2))**2*np.sin(beta*t_vec), 'b-', label=r'exact $w$')
    plt.plot(t_vec, v_atP, 'r-', label=r'approx $v$')
    plt.plot(t_vec, f_0*beta/(64*D)*(R**2-(Ppoint[0]**2+Ppoint[1]**2))**2*np.cos(beta*t_vec), 'b-', label=r'exact $v$')
    plt.xlabel(r'Time [s]')
    plt.title(r'Displacement at ' +  str(Ppoint))
    plt.legend()

    v_err_last = v_err_H1[-1]
    v_err_max = max(v_err_H1)
    v_err_quad = np.sqrt(np.sum(dt * np.power(v_err_H1, 2)))

    sig_err_last = sig_err_L2[-1]
    sig_err_max = max(sig_err_L2)
    sig_err_quad = np.sqrt(np.sum(dt * np.power(sig_err_L2, 2)))

    return v_err_last, v_err_max, v_err_quad, sig_err_last, sig_err_max, sig_err_quad


n_h = 3
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

if save_res:
    np.save("./convergence_results/" + bc_input + "_h1", h1_vec)
    np.save("./convergence_results/" + bc_input + "_h2", h1_vec)
    np.save("./convergence_results/" + bc_input + "_h3", h2_vec)

    np.save("./convergence_results/" + bc_input + "_v_errF_r1", v_err_r1)
    np.save("./convergence_results/" + bc_input + "_v_errInf_r1", v_errInf_r1)
    np.save("./convergence_results/" + bc_input + "_v_errQuad_r1", v_errQuad_r1)

    np.save("./convergence_results/" + bc_input + "_v_errF_r2", v_err_r2)
    np.save("./convergence_results/" + bc_input + "_v_errInf_r2", v_errInf_r2)
    np.save("./convergence_results/" + bc_input + "_v_errQuad_r2", v_errQuad_r2)

    np.save("./convergence_results/" + bc_input + "_v_errF_r3", v_err_r3)
    np.save("./convergence_results/" + bc_input + "_v_errInf_r3", v_errInf_r3)
    np.save("./convergence_results/" + bc_input + "_v_errQuad_r3", v_errQuad_r3)

    np.save("./convergence_results/" + bc_input + "_sig_errF_r1", sig_err_r1)
    np.save("./convergence_results/" + bc_input + "_sig_errInf_r1", sig_errInf_r1)
    np.save("./convergence_results/" + bc_input + "_sig_errQuad_r1", sig_errQuad_r1)

    np.save("./convergence_results/" + bc_input + "_sig_errF_r2", sig_err_r2)
    np.save("./convergence_results/" + bc_input + "_sig_errInf_r2", sig_errInf_r2)
    np.save("./convergence_results/" + bc_input + "_sig_errQuad_r2", sig_errQuad_r2)

    np.save("./convergence_results/" + bc_input + "_sig_errF_r3", sig_err_r3)
    np.save("./convergence_results/" + bc_input + "_sig_errInf_r3", sig_errInf_r3)
    np.save("./convergence_results/" + bc_input + "_sig_errQuad_r3", sig_errQuad_r3)

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
path_fig = "/home/a.brugnoli/Plots_Videos/Python/Plots/Kirchhoff_plots/Convergence/firedrake/"
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