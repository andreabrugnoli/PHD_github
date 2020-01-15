# Convergence test for HHJ

from firedrake import *
import numpy as np
import scipy as sp

import scipy.linalg as la
import scipy.sparse as spa
import scipy.sparse.linalg as sp_la

np.set_printoptions(threshold=np.inf)
from math import pi, floor

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['text.usetex'] = True

bc_input = 'CCCC'
save_res = False

def compute_err(n, r):

    h_mesh = 1/n

    E = Constant(1)
    nu = Constant(0.3)

    rho = Constant(1)
    k = Constant(5/6)
    h = Constant(0.1)

    D = E * h ** 3 / (1 - nu ** 2) / 12
    fl_rot = 12 / (E * h ** 3)

    G = E / 2 / (1 + nu)
    F = G * h * k

    # Operators and functions

    def gradSym(u):
        return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
        # return sym(nabla_grad(u))

    def gradSkew(u):
        return 0.5 * (nabla_grad(u) - nabla_grad(u).T)

    def bending_mom(kappa):
        momenta = D * ((1-nu)*kappa + nu * Identity(2) * tr(kappa))
        return momenta

    def bending_curv(momenta):
        kappa = fl_rot * ((1+nu)*momenta - nu * Identity(2) * tr(momenta))
        return kappa

    def m_operator(v_pw, al_pw, v_pth, al_pth, v_qth, al_qth, e_qth, v_qw, al_qw, v_skw, e_skw):

        m_form = v_pw * al_pw * dx \
                 + dot(v_pth, al_pth) * dx \
                 + inner(v_qth, al_qth) * dx + inner(v_qth, e_skw) * dx \
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

    Lx, Ly = 1, 1
    mesh = RectangleMesh(n, n, Lx, Lx, quadrilateral=False)

    # plot(mesh);
    # plt.show()


    # Finite element defition

    V_pw = FunctionSpace(mesh, "DG", r-1)
    V_pth = VectorFunctionSpace(mesh, "DG", r-1)
    V_skw = FunctionSpace(mesh, "DG", r-1)

    V_qth1 = FunctionSpace(mesh, "BDM", r)
    V_qth2 = FunctionSpace(mesh, "BDM", r)
    V_qw = FunctionSpace(mesh, "RT", r)

    V = MixedFunctionSpace([V_pw, V_pth, V_qth1, V_qth2, V_qw, V_skw])
    n_V = V.dim()
    print(n_V)

    v = TestFunction(V)
    v_pw, v_pth, v_qth1, v_qth2, v_qw, v_skw = split(v)

    e = TrialFunction(V)
    e_pw, e_pth, e_qth1, e_qth2, e_qw, e_skw = split(e)

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
    e_skw = as_tensor([[0, e_skw],
                        [-e_skw, 0]])

    # v_skw = skew(v_skw)
    # al_skw = skew(e_skw)

    dx = Measure('dx')
    ds = Measure('ds')

    t = 0
    t_fin = 1        # total simulation time

    t_ = Constant(t)
    t_1 = Constant(t)

    beta = 1

    x, y = SpatialCoordinate(mesh)

    w_st = 1/3*x**3*(x-1)**3*y**3*(y-1)**3 \
           - 2 *h**2/(5*(1-nu))*\
             (y**3*(y-1)**3*x*(x-1)*(5*x**2-5*x+1)
             +x**3*(x-1)**3*y*(y-1)*(5*y**2-5*y+1))

    dx_w_st = -(2*h**2*(3*(y-1)*y*(5*y**2-5*y+1)*(x-1)**2*x**3+(y-1)**3*y**3*x*(5*x**2-5*x+1)\
                        +(y-1)**3*y**3*(x-1)*(5*x**2-5*x+1)+3*(y-1)*y*(5*y**2-5*y+1)*(x-1)**3*x**2\
                        +(y-1)**3*y**3*(x-1)*x*(10*x-5)))/(5*(1-nu))+(y-1)**3*y**3*(x-1)**2*x**3\
                        +(y-1)**3*y**3*(x-1)**3*x**2

    dy_w_st = -(2*h**2*(3*(x-1)*x*(5*x**2-5*x+1)*(y-1)**2*y**3+(x-1)**3*x**3*y*(5*y**2-5*y+1)\
                        +(x-1)**3*x**3*(y-1)*(5*y**2-5*y+1)+3*(x-1)*x*(5*x**2-5*x+1)*(y-1)**3 * y**2\
                        +(x-1)**3*x**3*(y-1)*y*(10*y-5)))/(5*(1-nu))+(x-1)**3*x**3*(y-1)**2*y**3\
                        +(x-1)**3*x**3*(y-1)**3*y**2

    thx_st = y ** 3 * (y - 1) ** 3 * x ** 2 * (x - 1) ** 2 * (2 * x - 1)
    thy_st = x ** 3 * (x - 1) ** 3 * y ** 2 * (y - 1) ** 2 * (2 * y - 1)

    dx_thx_st = 2 * (y - 1) ** 3 * y ** 3 * (x - 1) * x * (5 * x ** 2 - 5 * x + 1)
    dy_thx_st = 3 * (x - 1) ** 2 * x ** 2 * (2 * x - 1) * (y - 1) ** 2 * y ** 2 * (2 * y - 1)

    dx_thy_st = 3 * (y - 1) ** 2 * y ** 2 * (2 * y - 1) * (x - 1) ** 2 * x ** 2 * (2 * x - 1)
    dy_thy_st = 2 * (x - 1) ** 3 * x ** 3 * (y - 1) * y * (5 * y ** 2 - 5 * y + 1)

    th_st = as_vector([thx_st, thy_st])

    f_st = E*h**3/(12*(1-nu**2))*(12*y*(y-1)*(5*x**2-5*x+1)*(2*y**2*(y-1)**2+x*(x-1)*(5*y**2-5*y+1)) + \
                                  12*x*(x-1)*(5*y**2-5*y+1)*(2*x**2*(x-1)**2+y*(y-1)*(5*x**2-5*x+1)))

    w_dyn = w_st * sin(beta * t_)
    thx_dyn = thx_st * sin(beta * t_)
    thy_dyn = thy_st * sin(beta * t_)

    th_dyn = as_vector([thx_dyn, thy_dyn])

    dx_thx_dyn = dx_thx_st * sin(beta * t_)
    dy_thx_dyn = dy_thx_st * sin(beta * t_)

    dx_thy_dyn = dx_thy_st * sin(beta * t_)
    dy_thy_dyn = dy_thy_st * sin(beta * t_)

    # sigma_ex = bending_mom(grad(th_dyn))

    kxx = dx_thx_dyn
    kyy = dy_thy_dyn
    kxy = 0.5*(dy_thx_dyn + dx_thy_dyn)

    sigma_ex = as_tensor([[D * (kxx + nu * kyy), D * (1 - nu) * kxy],
                          [D * (1 - nu) * kxy, D * (kyy + nu * kxx)]])

    q_ex = F*(grad(w_dyn) - th_dyn)
    # r_ex = 0.5*(thx_dyn.dx(1) - thy_dyn.dx(0))
    r_ex = -0.5*(dy_thx_dyn - dx_thy_dyn)

    dt_w = beta * w_st * cos(beta*t_)
    dtt_w = -beta**2 * w_st * sin(beta*t_)
    dtt_w1 = -beta ** 2 * w_st * sin(beta * t_1)

    dt_thx = beta * thx_st * cos(beta * t_)
    dt_thy = beta * thy_st * cos(beta * t_)
    dt_th = as_vector([dt_thx, dt_thy])

    dtt_thx = -beta**2 * thx_st * sin(beta * t_)
    dtt_thy = -beta**2 * thy_st * sin(beta * t_)
    dtt_th = as_vector([dtt_thx, dtt_thy])

    dtt_thx1 = -beta ** 2 * thx_st * sin(beta * t_1)
    dtt_thy1 = -beta ** 2 * thy_st * sin(beta * t_1)
    dtt_th1 = as_vector([dtt_thx1, dtt_thy1])

    f_dyn = f_st*sin(beta*t_) + rho*h*dtt_w
    f_dyn1 = f_st*sin(beta*t_1) + rho*h*dtt_w1

    m_dyn = rho*h**3/12*dtt_th
    m_dyn1 = rho*h**3/12*dtt_th1

    # J, M = PETScMatrix(), PETScMatrix()

    dt = 0.1*h_mesh
    theta = 0.5

    m_form = m_operator(v_pw, al_pw, v_pth, al_pth, v_qth, al_qth, e_qth, v_qw, al_qw, v_skw, e_skw)
    j_form = j_operator(v_pw, e_pw, v_pth, e_pth, v_qth, e_qth, v_qw, e_qw)

    f_form = v_pw*f_dyn*dx + dot(v_pth, m_dyn)*dx
    f_form1 = v_pw*f_dyn1*dx + dot(v_pth, m_dyn1)*dx

    # MM = sp.sparse.csr_matrix(assemble(m_form, mat_type='aij').M.handle.getValuesCSR()[::-1])
    # JJ = sp.sparse.csr_matrix(assemble(j_form, mat_type='aij').M.handle.getValuesCSR()[::-1])

    # plt.spy(MM); plt.show()
    # plt.spy(JJ); plt.show()
    lhs = m_form - dt*theta*j_form
    A = assemble(lhs, mat_type='aij')

    e_n1 = Function(V)
    e_n = Function(V)
    w_n1 = Function(V_pw)
    w_n = Function(V_pw)
    th_n1 = Function(V_pth)
    th_n = Function(V_pth)

    e_n.sub(0).assign(interpolate(dt_w, V_pw))
    e_n.sub(1).assign(interpolate(dt_th, V_pth))

    epw_n, epth_n, eqth1_n, eqth2_n, eqw_n, eskw_n = e_n.split()

    eqth_n = as_tensor([[eqth1_n[0], eqth1_n[1]],
                       [eqth2_n[0], eqth2_n[1]]])

    w_n.assign(Constant(0.0))

    n_t = int(floor(t_fin/dt) + 1)

    w_atP = np.zeros((n_t,))
    v_atP = np.zeros((n_t,))

    r_atP = np.zeros((n_t,))

    th_atP = np.zeros((2, n_t))
    om_atP = np.zeros((2, n_t))

    Ppoint = (Lx/3, Ly/7)
    v_atP[0] = epw_n.at(Ppoint)
    r_atP[0] = eskw_n.at(Ppoint)

    v_err_L2 = np.zeros((n_t,))
    om_err_L2 = np.zeros((n_t,))
    sig_err_L2 = np.zeros((n_t,))
    # q_err_Hdiv = np.zeros((n_t,))
    q_err_L2 = np.zeros((n_t,))
    r_err_L2 = np.zeros((n_t,))

    v_err_L2[0] = np.sqrt(assemble((epw_n - dt_w)*(epw_n - dt_w) * dx))
    om_err_L2[0] = np.sqrt(assemble(inner(epth_n - dt_th, epth_n - dt_th) * dx))
    sig_err_L2[0] = np.sqrt(assemble(inner(eqth_n - sigma_ex, eqth_n - sigma_ex) * dx))
    # q_err_Hdiv[0] = np.sqrt(assemble(inner(eqw_n - q_ex, eqw_n - q_ex) * dx \
    #                                + inner(div(eqw_n - q_ex), div(eqw_n - q_ex)) * dx))
    q_err_L2[0] = np.sqrt(assemble(inner(eqw_n - q_ex, eqw_n - q_ex) * dx))
    r_err_L2[0] = np.sqrt(assemble((eskw_n - r_ex)**2 * dx))

    t_vec = np.linspace(0, t_fin, num=n_t)

    param = {"ksp_type": "gmres", "ksp_gmres_restart":100} #, "ksp_atol":1e-40}
    # param = {"ksp_type": "preonly", "pc_type": "lu"}

    for i in range(1, n_t):

        t_.assign(t)
        t_1.assign(t + dt)

        epw_n, epth_n, eqth1_n, eqth2_n, eqw_n, eskw_n = e_n.split()

        eqth_n = as_tensor([[eqth1_n[0], eqth1_n[1]],
                            [eqth2_n[0], eqth2_n[1]]])
        alpw_n = rho * h * epw_n
        alpth_n = (rho * h ** 3) / 12. * epth_n
        alqth_n = bending_curv(eqth_n)
        alqw_n = 1. / F * eqw_n

        eskw_n = as_tensor([[0, eskw_n],
                           [-eskw_n, 0]])

        rhs = m_operator(v_pw, alpw_n, v_pth, alpth_n, v_qth, alqth_n, eqth_n, v_qw, alqw_n, v_skw, eskw_n) \
              + dt * (1 - theta) * j_operator(v_pw, epw_n, v_pth, epth_n, v_qth, eqth_n, v_qw, eqw_n) \
              + dt * ((1 - theta) * f_form + theta * f_form1)

        b = assemble(rhs)

        solve(A, e_n1, b, solver_parameters=param)

        t += dt

        epw_n1, epth_n1, eqth1_n1, eqth2_n1, eqw_n1, eskw_n1 = e_n1.split()

        eqth_n1 = as_tensor([[eqth1_n1[0], eqth1_n1[1]],
                            [eqth2_n1[0], eqth2_n1[1]]])

        w_n1.assign(w_n + dt / 2 * (epw_n + epw_n1))
        w_n.assign(w_n1)

        e_n.assign(e_n1)

        w_atP[i] = w_n1.at(Ppoint)
        v_atP[i] = epw_n1.at(Ppoint)
        r_atP[i] = eskw_n1.at(Ppoint)

        t_.assign(t)

        v_err_L2[i] = np.sqrt(assemble(inner(epw_n1 - dt_w, epw_n1 - dt_w) * dx))
        om_err_L2[i] = np.sqrt(assemble(inner(epth_n1 - dt_th, epth_n1 - dt_th) * dx))
        sig_err_L2[i] = np.sqrt(assemble(inner(eqth_n1 - sigma_ex, eqth_n1 - sigma_ex) * dx))
        # q_err_Hdiv[i] = np.sqrt(assemble(inner(eqw_n1 - q_ex, eqw_n1 - q_ex) * dx  \
        #                                  + inner(div(eqw_n1 - q_ex), div(eqw_n1 - q_ex)) * dx))
        q_err_L2[i] = np.sqrt(assemble(inner(eqw_n1 - q_ex, eqw_n1 - q_ex) * dx))

        r_err_L2[i] = np.sqrt(assemble(inner(eskw_n1 - r_ex, eskw_n1 - r_ex) * dx))

    # plt.figure()
    # plt.plot(t_vec, v_atP, 'r-', label=r'approx $v$')
    # wst_atP = interpolate(w_st, V_pw).at(Ppoint)
    # vex_atP = wst_atP*beta*np.cos(beta*t_vec)
    # plt.plot(t_vec, vex_atP, 'b-', label=r'exact $v$')
    # plt.xlabel(r'Time [s]')
    # plt.title(r'Displacement at ' + str(Ppoint))
    # plt.legend()

    plt.figure()
    plt.plot(t_vec, r_atP, 'r-', label=r'approx $r$')
    # wst_atP = interpolate(w_st, V_pw).at(Ppoint)
    # vex_atP = wst_atP*beta*np.cos(beta*t_vec)
    # plt.plot(t_vec, vex_atP, 'b-', label=r'exact $v$')
    plt.xlabel(r'Time [s]')
    plt.title(r'Displacement at ' + str(Ppoint))
    plt.legend()

    v_err_max = max(v_err_L2)
    v_err_quad = np.sqrt(np.sum(dt * np.power(v_err_L2, 2)))

    om_err_max = max(om_err_L2)
    om_err_quad = np.sqrt(np.sum(dt * np.power(om_err_L2, 2)))

    sig_err_max = max(sig_err_L2)
    sig_err_quad = np.sqrt(np.sum(dt * np.power(sig_err_L2, 2)))

    # q_err_max = max(q_err_Hdiv)
    # q_err_quad = np.sqrt(np.sum(dt * np.power(q_err_Hdiv, 2)))

    q_err_max = max(q_err_L2)
    q_err_quad = np.sqrt(np.sum(dt * np.power(q_err_L2, 2)))

    r_err_max = max(r_err_L2)
    r_err_quad = np.sqrt(np.sum(dt * np.power(r_err_L2, 2)))

    return v_err_max, v_err_quad, om_err_max, om_err_quad, sig_err_max, sig_err_quad, \
           q_err_max, q_err_quad, r_err_max, r_err_quad


n_h = 2
n1_vec = np.array([2**(i+2) for i in range(n_h)])
n2_vec = np.array([2**(i+1) for i in range(n_h)])
h1_vec = 1./n1_vec
h2_vec = 1./n2_vec

v_errInf_r1 = np.zeros((n_h,))
v_errQuad_r1 = np.zeros((n_h,))

v_errInf_r2 = np.zeros((n_h,))
v_errQuad_r2 = np.zeros((n_h,))

v_errInf_r3 = np.zeros((n_h,))
v_errQuad_r3 = np.zeros((n_h,))

v_r1_max = np.zeros((n_h-1,))
v_r1_L2 = np.zeros((n_h-1,))

v_r2_max = np.zeros((n_h-1,))
v_r2_L2 = np.zeros((n_h-1,))

v_r3_max = np.zeros((n_h-1,))
v_r3_L2 = np.zeros((n_h-1,))

om_errInf_r1 = np.zeros((n_h,))
om_errQuad_r1 = np.zeros((n_h,))

om_errInf_r2 = np.zeros((n_h,))
om_errQuad_r2 = np.zeros((n_h,))

om_errInf_r3 = np.zeros((n_h,))
om_errQuad_r3 = np.zeros((n_h,))

om_r1_max = np.zeros((n_h-1,))
om_r1_L2 = np.zeros((n_h-1,))

om_r2_max = np.zeros((n_h-1,))
om_r2_L2 = np.zeros((n_h-1,))

om_r3_max = np.zeros((n_h-1,))
om_r3_L2 = np.zeros((n_h-1,))

sig_errInf_r1 = np.zeros((n_h,))
sig_errQuad_r1 = np.zeros((n_h,))

sig_errInf_r2 = np.zeros((n_h,))
sig_errQuad_r2 = np.zeros((n_h,))

sig_errInf_r3 = np.zeros((n_h,))
sig_errQuad_r3 = np.zeros((n_h,))

sig_r1_max = np.zeros((n_h-1,))
sig_r1_L2 = np.zeros((n_h-1,))

sig_r2_max = np.zeros((n_h-1,))
sig_r2_L2 = np.zeros((n_h-1,))

sig_r3_max = np.zeros((n_h-1,))
sig_r3_L2 = np.zeros((n_h-1,))

q_errInf_r1 = np.zeros((n_h,))
q_errQuad_r1 = np.zeros((n_h,))

q_errInf_r2 = np.zeros((n_h,))
q_errQuad_r2 = np.zeros((n_h,))

q_errInf_r3 = np.zeros((n_h,))
q_errQuad_r3 = np.zeros((n_h,))

q_r1_max = np.zeros((n_h-1,))
q_r1_L2 = np.zeros((n_h-1,))

q_r2_max = np.zeros((n_h-1,))
q_r2_L2 = np.zeros((n_h-1,))

q_r3_max = np.zeros((n_h-1,))
q_r3_L2 = np.zeros((n_h-1,))

r_errInf_r1 = np.zeros((n_h,))
r_errQuad_r1 = np.zeros((n_h,))

r_errInf_r2 = np.zeros((n_h,))
r_errQuad_r2 = np.zeros((n_h,))

r_errInf_r3 = np.zeros((n_h,))
r_errQuad_r3 = np.zeros((n_h,))

r_r1_max = np.zeros((n_h-1,))
r_r1_L2 = np.zeros((n_h-1,))

r_r2_max = np.zeros((n_h-1,))
r_r2_L2 = np.zeros((n_h-1,))

r_r3_max = np.zeros((n_h-1,))
r_r3_L2 = np.zeros((n_h-1,))

for i in range(n_h):
    v_errInf_r1[i], v_errQuad_r1[i], om_errInf_r1[i], om_errQuad_r1[i], \
    sig_errInf_r1[i], sig_errQuad_r1[i], q_errInf_r1[i], q_errQuad_r1[i], \
    r_errInf_r1[i], r_errQuad_r1[i] = compute_err(n1_vec[i], 1)

    v_errInf_r2[i], v_errQuad_r2[i], om_errInf_r2[i], om_errQuad_r2[i], \
    sig_errInf_r2[i], sig_errQuad_r2[i], q_errInf_r2[i], q_errQuad_r2[i], \
    r_errInf_r2[i], r_errQuad_r2[i] = compute_err(n1_vec[i], 2)

    v_errInf_r3[i], v_errQuad_r3[i], om_errInf_r3[i], om_errQuad_r3[i], \
    sig_errInf_r3[i], sig_errQuad_r3[i], q_errInf_r3[i], q_errQuad_r3[i], \
    r_errInf_r3[i], r_errQuad_r3[i] = compute_err(n1_vec[i], 3)

    if i>0:
        v_r1_max[i-1] = np.log(v_errInf_r1[i]/v_errInf_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
        v_r1_L2[i-1] = np.log(v_errQuad_r1[i]/v_errQuad_r1[i-1])/np.log(h1_vec[i]/h1_vec[i-1])

        v_r2_max[i-1] = np.log(v_errInf_r2[i]/v_errInf_r2[i-1])/np.log(h1_vec[i]/h1_vec[i-1])
        v_r2_L2[i-1] = np.log(v_errQuad_r2[i]/v_errQuad_r2[i-1])/np.log(h1_vec[i]/h1_vec[i-1])

        v_r3_max[i-1] = np.log(v_errInf_r3[i]/v_errInf_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])
        v_r3_L2[i-1] = np.log(v_errQuad_r3[i]/v_errQuad_r3[i-1])/np.log(h2_vec[i]/h2_vec[i-1])

        om_r1_max[i - 1] = np.log(om_errInf_r1[i] / om_errInf_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        om_r1_L2[i - 1] = np.log(om_errQuad_r1[i] / om_errQuad_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

        om_r2_max[i - 1] = np.log(om_errInf_r2[i] / om_errInf_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        om_r2_L2[i - 1] = np.log(om_errQuad_r2[i] / om_errQuad_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

        om_r3_max[i - 1] = np.log(om_errInf_r3[i] / om_errInf_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])
        om_r3_L2[i - 1] = np.log(om_errQuad_r3[i] / om_errQuad_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])

        sig_r1_max[i - 1] = np.log(sig_errInf_r1[i] / sig_errInf_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        sig_r1_L2[i - 1] = np.log(sig_errQuad_r1[i] / sig_errQuad_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

        sig_r2_max[i - 1] = np.log(sig_errInf_r2[i] / sig_errInf_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        sig_r2_L2[i - 1] = np.log(sig_errQuad_r2[i] / sig_errQuad_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

        sig_r3_max[i - 1] = np.log(sig_errInf_r3[i] / sig_errInf_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])
        sig_r3_L2[i - 1] = np.log(sig_errQuad_r3[i] / sig_errQuad_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])

        q_r1_max[i - 1] = np.log(q_errInf_r1[i] / q_errInf_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        q_r1_L2[i - 1] = np.log(q_errQuad_r1[i] / q_errQuad_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

        q_r2_max[i - 1] = np.log(q_errInf_r2[i] / q_errInf_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        q_r2_L2[i - 1] = np.log(q_errQuad_r2[i] / q_errQuad_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

        q_r3_max[i - 1] = np.log(q_errInf_r3[i] / q_errInf_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])
        q_r3_L2[i - 1] = np.log(q_errQuad_r3[i] / q_errQuad_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])

        r_r1_max[i - 1] = np.log(r_errInf_r1[i] / r_errInf_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        r_r1_L2[i - 1] = np.log(r_errQuad_r1[i] / r_errQuad_r1[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

        r_r2_max[i - 1] = np.log(r_errInf_r2[i] / r_errInf_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])
        r_r2_L2[i - 1] = np.log(r_errQuad_r2[i] / r_errQuad_r2[i - 1]) / np.log(h1_vec[i] / h1_vec[i - 1])

        r_r3_max[i - 1] = np.log(r_errInf_r3[i] / r_errInf_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])
        r_r3_L2[i - 1] = np.log(r_errQuad_r3[i] / r_errQuad_r3[i - 1]) / np.log(h2_vec[i] / h2_vec[i - 1])

if save_res:
    np.save("./convergence_results_mindlin/" + bc_input + "_h1", h1_vec)
    np.save("./convergence_results_mindlin/" + bc_input + "_h2", h1_vec)
    np.save("./convergence_results_mindlin/" + bc_input + "_h3", h2_vec)

    np.save("./convergence_results_mindlin/" + bc_input + "_v_errInf_r1", v_errInf_r1)
    np.save("./convergence_results_mindlin/" + bc_input + "_v_errQuad_r1", v_errQuad_r1)

    np.save("./convergence_results_mindlin/" + bc_input + "_v_errInf_r2", v_errInf_r2)
    np.save("./convergence_results_mindlin/" + bc_input + "_v_errQuad_r2", v_errQuad_r2)

    np.save("./convergence_results_mindlin/" + bc_input + "_v_errInf_r3", v_errInf_r3)
    np.save("./convergence_results_mindlin/" + bc_input + "_v_errQuad_r3", v_errQuad_r3)

    np.save("./convergence_results_mindlin/" + bc_input + "_om_errInf_r1", om_errInf_r1)
    np.save("./convergence_results_mindlin/" + bc_input + "_om_errQuad_r1", om_errQuad_r1)

    np.save("./convergence_results_mindlin/" + bc_input + "_om_errInf_r2", om_errInf_r2)
    np.save("./convergence_results_mindlin/" + bc_input + "_om_errQuad_r2", om_errQuad_r2)

    np.save("./convergence_results_mindlin/" + bc_input + "_om_errInf_r3", om_errInf_r3)
    np.save("./convergence_results_mindlin/" + bc_input + "_om_errQuad_r3", om_errQuad_r3)

    np.save("./convergence_results_mindlin/" + bc_input + "_sig_errInf_r1", sig_errInf_r1)
    np.save("./convergence_results_mindlin/" + bc_input + "_sig_errQuad_r1", sig_errQuad_r1)

    np.save("./convergence_results_mindlin/" + bc_input + "_sig_errInf_r2", sig_errInf_r2)
    np.save("./convergence_results_mindlin/" + bc_input + "_sig_errQuad_r2", sig_errQuad_r2)

    np.save("./convergence_results_mindlin/" + bc_input + "_sig_errInf_r3", sig_errInf_r3)
    np.save("./convergence_results_mindlin/" + bc_input + "_sig_errQuad_r3", sig_errQuad_r3)

    np.save("./convergence_results_mindlin/" + bc_input + "_q_errInf_r1", q_errInf_r1)
    np.save("./convergence_results_mindlin/" + bc_input + "_q_errQuad_r1", q_errQuad_r1)

    np.save("./convergence_results_mindlin/" + bc_input + "_q_errInf_r2", q_errInf_r2)
    np.save("./convergence_results_mindlin/" + bc_input + "_q_errQuad_r2", q_errQuad_r2)

    np.save("./convergence_results_mindlin/" + bc_input + "_q_errInf_r3", q_errInf_r3)
    np.save("./convergence_results_mindlin/" + bc_input + "_q_errQuad_r3", q_errQuad_r3)

    np.save("./convergence_results_mindlin/" + bc_input + "_r_errInf_r1", r_errInf_r1)
    np.save("./convergence_results_mindlin/" + bc_input + "_r_errQuad_r1", r_errQuad_r1)

    np.save("./convergence_results_mindlin/" + bc_input + "_r_errInf_r2", r_errInf_r2)
    np.save("./convergence_results_mindlin/" + bc_input + "_r_errQuad_r2", r_errQuad_r2)

    np.save("./convergence_results_mindlin/" + bc_input + "_r_errInf_r3", r_errInf_r3)
    np.save("./convergence_results_mindlin/" + bc_input + "_r_errQuad_r3", r_errQuad_r3)

v_r1int_max = np.polyfit(np.log(h1_vec), np.log(v_errInf_r1), 1)[0]
v_r1int_L2 = np.polyfit(np.log(h1_vec), np.log(v_errQuad_r1), 1)[0]

print("Estimated order of convergence r=1 for v Linf: " + str(v_r1_max))
print("Interpolated order of convergence r=1 for v Linf: " + str(v_r1int_max))
print("Estimated order of convergence r=1 for v L2: " + str(v_r1_L2))
print("Interpolated order of convergence r=1 for v L2: " + str(v_r1int_L2))
print("")

v_r2int_max = np.polyfit(np.log(h1_vec), np.log(v_errInf_r2), 1)[0]
v_r2int_L2 = np.polyfit(np.log(h1_vec), np.log(v_errQuad_r2), 1)[0]

print("Estimated order of convergence r=2 for v Linf: " + str(v_r2_max))
print("Interpolated order of convergence r=2 for v Linf: " + str(v_r2int_max))
print("Estimated order of convergence r=2 for v L2: " + str(v_r2_L2))
print("Interpolated order of convergence r=2 for v L2: " + str(v_r2int_L2))
print("")

v_r3int_max = np.polyfit(np.log(h2_vec), np.log(v_errInf_r3), 1)[0]
v_r3int_L2 = np.polyfit(np.log(h2_vec), np.log(v_errQuad_r3), 1)[0]

print("Estimated order of convergence r=3 for v Linf: " + str(v_r3_max))
print("Interpolated order of convergence r=3 for v Linf: " + str(v_r3int_max))
print("Estimated order of convergence r=3 for v L2: " + str(v_r3_L2))
print("Interpolated order of convergence r=3 for v L2: " + str(v_r3int_L2))
print("")

plt.figure()

# plt.plot(np.log(h1_vec), np.log(v_r1_atF), ':o', label='AFW 1')
plt.plot(np.log(h1_vec), np.log(v_errInf_r1), '-.+', label='AFW 1 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(v_errQuad_r1), '--*', label='AFW 1 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec), '-v', label=r'$h$')

# plt.plot(np.log(h1_vec), np.log(v_r2_atF), ':o', label='AFW 2')
plt.plot(np.log(h1_vec), np.log(v_errInf_r2), '-.+', label='AFW 2 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(v_errQuad_r2), '--*', label='AFW 2 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec**2), '-v', label=r'$h^2$')

# plt.plot(np.log(h2_vec), np.log(v_r3_atF), ':o', label='AFW 3')
plt.plot(np.log(h2_vec), np.log(v_errInf_r3), '-.+', label='AFW 3 $L^\infty$')
plt.plot(np.log(h2_vec), np.log(v_errQuad_r3), '--*', label='AFW 3 $L^2$')
plt.plot(np.log(h2_vec), np.log(h2_vec**3), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log(Error Velocity)')
plt.title(r'Velocity Error vs Mesh size')
plt.legend()
path_fig = "/home/a.brugnoli/Plots/Python/Plots/Mindlin_plots/Convergence/firedrake/"
if save_res:
    plt.savefig(path_fig  + bc_input + "_vel.eps", format="eps")

om_r1int_max = np.polyfit(np.log(h1_vec), np.log(om_errInf_r1), 1)[0]
om_r1int_L2 = np.polyfit(np.log(h1_vec), np.log(om_errQuad_r1), 1)[0]

print("Estimated order of convergence r=1 for om Linf: " + str(om_r1_max))
print("Interpolated order of convergence r=1 for om Linf: " + str(om_r1int_max))
print("Estimated order of convergence r=1 for om L2: " + str(om_r1_L2))
print("Interpolated order of convergence r=1 for om L2: " + str(om_r1int_L2))
print("")

om_r2int_max = np.polyfit(np.log(h1_vec), np.log(om_errInf_r2), 1)[0]
om_r2int_L2 = np.polyfit(np.log(h1_vec), np.log(om_errQuad_r2), 1)[0]

print("Estimated order of convergence r=2 for om Linf: " + str(om_r2_max))
print("Interpolated order of convergence r=2 for om Linf: " + str(om_r2int_max))
print("Estimated order of convergence r=2 for om L2: " + str(om_r2_L2))
print("Interpolated order of convergence r=2 for om L2: " + str(om_r2int_L2))
print("")

om_r3int_max = np.polyfit(np.log(h2_vec), np.log(om_errInf_r3), 1)[0]
om_r3int_L2 = np.polyfit(np.log(h2_vec), np.log(om_errQuad_r3), 1)[0]

print("Estimated order of convergence r=3 for om Linf: " + str(om_r3_max))
print("Interpolated order of convergence r=3 for om Linf: " + str(om_r3int_max))
print("Estimated order of convergence r=3 for om L2: " + str(om_r3_L2))
print("Interpolated order of convergence r=3 for om L2: " + str(om_r3int_L2))
print("")

plt.figure()

# plt.plot(np.log(h1_vec), np.log(om_r1_atF), ':o', label='AFW 1')
plt.plot(np.log(h1_vec), np.log(om_errInf_r1), '-.+', label='AFW 1 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(om_errQuad_r1), '--*', label='AFW 1 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec), '-v', label=r'$h$')

# plt.plot(np.log(h1_vec), np.log(om_r2_atF), ':o', label='AFW 2')
plt.plot(np.log(h1_vec), np.log(om_errInf_r2), '-.+', label='AFW 2 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(om_errQuad_r2), '--*', label='AFW 2 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec**2), '-v', label=r'$h^2$')

# plt.plot(np.log(h2_vec), np.log(om_r3_atF), ':o', label='AFW 3')
plt.plot(np.log(h2_vec), np.log(om_errInf_r3), '-.+', label='AFW 3 $L^\infty$')
plt.plot(np.log(h2_vec), np.log(om_errQuad_r3), '--*', label='AFW 3 $L^2$')
plt.plot(np.log(h2_vec), np.log(h2_vec**3), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log(Error omega)')
plt.title(r'Omega Error vs Mesh size')
plt.legend()
if save_res:
    plt.savefig(path_fig + bc_input + "_om.eps", format="eps")

sig_r1int_max = np.polyfit(np.log(h1_vec), np.log(sig_errInf_r1), 1)[0]
sig_r1int_L2 = np.polyfit(np.log(h1_vec), np.log(sig_errQuad_r1), 1)[0]

print("Estimated order of convergence r=1 for sig Linf: " + str(sig_r1_max))
print("Interpolated order of convergence r=1 for sig Linf: " + str(sig_r1int_max))
print("Estimated order of convergence r=1 for sig L2: " + str(sig_r1_L2))
print("Interpolated order of convergence r=1 for sig L2: " + str(sig_r1int_L2))
print("")

sig_r2int_max = np.polyfit(np.log(h1_vec), np.log(sig_errInf_r2), 1)[0]
sig_r2int_L2 = np.polyfit(np.log(h1_vec), np.log(sig_errQuad_r2), 1)[0]

print("Estimated order of convergence r=2 for sig Linf: " + str(sig_r2_max))
print("Interpolated order of convergence r=2 for sig Linf: " + str(sig_r2int_max))
print("Estimated order of convergence r=2 for sig L2: " + str(sig_r2_L2))
print("Interpolated order of convergence r=2 for sig L2: " + str(sig_r2int_L2))
print("")

sig_r3int_max = np.polyfit(np.log(h2_vec), np.log(sig_errInf_r3), 1)[0]
sig_r3int_L2 = np.polyfit(np.log(h2_vec), np.log(sig_errQuad_r3), 1)[0]

print("Estimated order of convergence r=3 for sig Linf: " + str(sig_r3_max))
print("Interpolated order of convergence r=3 for sig Linf: " + str(sig_r3int_max))
print("Estimated order of convergence r=3 for sig L2: " + str(sig_r3_L2))
print("Interpolated order of convergence r=3 for sig L2: " + str(sig_r3int_L2))
print("")

plt.figure()

# plt.plot(np.log(h1_vec), np.log(sig_r1_atF), ':o', label='AFW 1')
plt.plot(np.log(h1_vec), np.log(sig_errInf_r1), '-.+', label='AFW 1 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(sig_errQuad_r1), '--*', label='AFW 1 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec), '-v', label=r'$h$')

# plt.plot(np.log(h1_vec), np.log(sig_r2_atF), ':o', label='AFW 2')
plt.plot(np.log(h1_vec), np.log(sig_errInf_r2), '-.+', label='AFW 2 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(sig_errQuad_r2), '--*', label='AFW 2 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec**2), '-v', label=r'$h^2$')

# plt.plot(np.log(h2_vec), np.log(sig_r3_atF), ':o', label='AFW 3')
plt.plot(np.log(h2_vec), np.log(sig_errInf_r3), '-.+', label='AFW 3 $L^\infty$')
plt.plot(np.log(h2_vec), np.log(sig_errQuad_r3), '--*', label='AFW 3 $L^2$')
plt.plot(np.log(h2_vec), np.log(h2_vec**3), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log(Error sigma)')
plt.title(r'Sigma Error vs Mesh size')
plt.legend()
if save_res:
    plt.savefig(path_fig + bc_input + "_sig.eps", format="eps")


q_r1int_max = np.polyfit(np.log(h1_vec), np.log(q_errInf_r1), 1)[0]
q_r1int_L2 = np.polyfit(np.log(h1_vec), np.log(q_errQuad_r1), 1)[0]

print("Estimated order of convergence r=1 for q Linf: " + str(q_r1_max))
print("Interpolated order of convergence r=1 for q Linf: " + str(q_r1int_max))
print("Estimated order of convergence r=1 for q L2: " + str(q_r1_L2))
print("Interpolated order of convergence r=1 for q L2: " + str(q_r1int_L2))
print("")

q_r2int_max = np.polyfit(np.log(h1_vec), np.log(q_errInf_r2), 1)[0]
q_r2int_L2 = np.polyfit(np.log(h1_vec), np.log(q_errQuad_r2), 1)[0]

print("Estimated order of convergence r=2 for q Linf: " + str(q_r2_max))
print("Interpolated order of convergence r=2 for q Linf: " + str(q_r2int_max))
print("Estimated order of convergence r=2 for q L2: " + str(q_r2_L2))
print("Interpolated order of convergence r=2 for q L2: " + str(q_r2int_L2))
print("")

q_r3int_max = np.polyfit(np.log(h2_vec), np.log(q_errInf_r3), 1)[0]
q_r3int_L2 = np.polyfit(np.log(h2_vec), np.log(q_errQuad_r3), 1)[0]

print("Estimated order of convergence r=3 for q Linf: " + str(q_r3_max))
print("Interpolated order of convergence r=3 for q Linf: " + str(q_r3int_max))
print("Estimated order of convergence r=3 for q L2: " + str(q_r3_L2))
print("Interpolated order of convergence r=3 for q L2: " + str(q_r3int_L2))
print("")

plt.figure()

# plt.plot(np.log(h1_vec), np.log(q_r1_atF), ':o', label='AFW 1')
plt.plot(np.log(h1_vec), np.log(q_errInf_r1), '-.+', label='AFW 1 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(q_errQuad_r1), '--*', label='AFW 1 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec), '-v', label=r'$h$')

# plt.plot(np.log(h1_vec), np.log(q_r2_atF), ':o', label='AFW 2')
plt.plot(np.log(h1_vec), np.log(q_errInf_r2), '-.+', label='AFW 2 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(q_errQuad_r2), '--*', label='AFW 2 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec**2), '-v', label=r'$h^2$')

# plt.plot(np.log(h2_vec), np.log(q_r3_atF), ':o', label='AFW 3')
plt.plot(np.log(h2_vec), np.log(q_errInf_r3), '-.+', label='AFW 3 $L^\infty$')
plt.plot(np.log(h2_vec), np.log(q_errQuad_r3), '--*', label='AFW 3 $L^2$')
plt.plot(np.log(h2_vec), np.log(h2_vec**3), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log(Error q)')
plt.title(r'q Error vs Mesh size')
plt.legend()
if save_res:
    plt.savefig(path_fig + bc_input + "_q.eps", format="eps")

r_r1int_max = np.polyfit(np.log(h1_vec), np.log(r_errInf_r1), 1)[0]
r_r1int_L2 = np.polyfit(np.log(h1_vec), np.log(r_errQuad_r1), 1)[0]

print("Estimated order of convergence r=1 for r Linf: " + str(r_r1_max))
print("Interpolated order of convergence r=1 for r Linf: " + str(r_r1int_max))
print("Estimated order of convergence r=1 for r L2: " + str(r_r1_L2))
print("Interpolated order of convergence r=1 for r L2: " + str(r_r1int_L2))
print("")

r_r2int_max = np.polyfit(np.log(h1_vec), np.log(r_errInf_r2), 1)[0]
r_r2int_L2 = np.polyfit(np.log(h1_vec), np.log(r_errQuad_r2), 1)[0]

print("Estimated order of convergence r=2 for r Linf: " + str(r_r2_max))
print("Interpolated order of convergence r=2 for r Linf: " + str(r_r2int_max))
print("Estimated order of convergence r=2 for r L2: " + str(r_r2_L2))
print("Interpolated order of convergence r=2 for r L2: " + str(r_r2int_L2))
print("")

r_r3int_max = np.polyfit(np.log(h2_vec), np.log(r_errInf_r3), 1)[0]
r_r3int_L2 = np.polyfit(np.log(h2_vec), np.log(r_errQuad_r3), 1)[0]

print("Estimated order of convergence r=3 for r Linf: " + str(r_r3_max))
print("Interpolated order of convergence r=3 for r Linf: " + str(r_r3int_max))
print("Estimated order of convergence r=3 for r L2: " + str(r_r3_L2))
print("Interpolated order of convergence r=3 for r L2: " + str(r_r3int_L2))
print("")

plt.figure()

# plt.plot(np.log(h1_vec), np.log(r_r1_atF), ':o', label='AFW 1')
plt.plot(np.log(h1_vec), np.log(r_errInf_r1), '-.+', label='AFW 1 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(r_errQuad_r1), '--*', label='AFW 1 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec), '-v', label=r'$h$')

# plt.plot(np.log(h1_vec), np.log(r_r2_atF), ':o', label='AFW 2')
plt.plot(np.log(h1_vec), np.log(r_errInf_r2), '-.+', label='AFW 2 $L^\infty$')
plt.plot(np.log(h1_vec), np.log(r_errQuad_r2), '--*', label='AFW 2 $L^2$')
plt.plot(np.log(h1_vec), np.log(h1_vec**2), '-v', label=r'$h^2$')

# plt.plot(np.log(h2_vec), np.log(r_r3_atF), ':o', label='AFW 3')
plt.plot(np.log(h2_vec), np.log(r_errInf_r3), '-.+', label='AFW 3 $L^\infty$')
plt.plot(np.log(h2_vec), np.log(r_errQuad_r3), '--*', label='AFW 3 $L^2$')
plt.plot(np.log(h2_vec), np.log(h2_vec**3), '-v', label=r'$h^3$')

plt.xlabel(r'log(Mesh size $h$)')
plt.ylabel(r'log(Error r)')
plt.title(r'r Error vs Mesh size')
plt.legend()
if save_res:
    plt.savefig(path_fig + bc_input + "_r.eps", format="eps")
plt.show()