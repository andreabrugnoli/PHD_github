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
import mpmath
mpmath.mp.dps = 15
mpmath.mp.pretty = True

def compute_analytical(delta, xi, t_fin):

    gam1 = lambda s: (s/2*((1 + delta + s) + ((1+delta+s)**2-4*s)**(1/2)))**(1/2)
    gam2 = lambda s: (s/2*((1 + delta + s) - ((1+delta+s)**2-4*s)**(1/2)))**(1/2)
    phi = lambda s: 1/s

    n_t = 100
    t_vec = np.linspace(0.00001, t_fin, n_t)
    theta_s = lambda s: phi(s)/(gam1(s)**2-gam2(s)**2) *\
                      ((gam1(s)**2-s**2)*mpmath.exp(-gam1(s)*xi) - (gam2(s)**2-s**2)*mpmath.exp(-gam2(s)*xi))

    disp_s = lambda s: -phi(s)/(gam1(s)**2-gam2(s)**2)*(gam1(s)*mpmath.exp(-gam1(s)*xi) - gam2(s)*mpmath.exp(-gam2(s)*xi))
    stress_s = lambda s: s**2*phi(s)/(gam1(s)**2-gam2(s)**2)*(mpmath.exp(-gam1(s)*xi) - mpmath.exp(-gam2(s)*xi))

    theta_t = np.empty((n_t, ))
    disp_t = np.empty((n_t, ))
    stress_t = np.empty((n_t, ))
    for i in range(n_t):
        theta_t[i] = mpmath.invertlaplace(theta_s, t_vec[i], method='dehoog')
        disp_t[i] = mpmath.invertlaplace(disp_s, t_vec[i], method='dehoog')
        stress_t[i] = mpmath.invertlaplace(stress_s, t_vec[i], method='dehoog')

    return t_vec, theta_t, disp_t, stress_t


def compute_sol(n, r, delta, tfin_hat):

    lamda = 0.8529 * 10**9  # kg cm^-1 s^-2
    mu = 0.5686 * 10**9 # kg cm^-1 s^-2
    rho = 7.82 * 10**(-3)  # kg cm^-3
    c_E = 4.61 * 10**6  # cm^2 K^-1 s^-3
    K = 1.7 * 10**3  # kg cm K^-1 s^-3

    c_1 = np.sqrt((lamda + 2*mu)/rho)

    alpha_T = 9.0375 * 10**(-6)  # K^-1  alpha = K/(rho*c_E)
    T_0 = 300  #  K

    beta = K/(rho*c_E*c_1)
    gamma = alpha_T*(3*lamda+2*mu)

    # C = 0
    # fac = 0
    C = alpha_T**2*(3*lamda+2*mu)**2*T_0/(rho*c_E*(2*mu + lamda))
    fac = 1/C

    # alpha_TCoup = np.sqrt((C*rho*c_E*(2*mu+lamda))/(T_0*(3*lamda + 2*mu)**2))
    # print(alpha_TCoup/alpha_T)
    # Operators and functions

    def gradSym(u):
        return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
        # return sym(nabla_grad(u))

    def gradSkew(u):
        return 0.5 * (nabla_grad(u) - nabla_grad(u).T)

    # E = mu*(3*lamda + 2*mu)/(lamda + mu)
    # nu = lamda/(2*(lamda+mu))

    def rigidity_tensor(epsilon):
        sigma = 2*mu*epsilon + lamda * Identity(2)*tr(epsilon)
        return sigma

    def compliance(stress):
        epsilon = 1./(2*mu) * stress - lamda/(2*mu*(3*lamda + 2*mu))*tr(stress)* Identity(2)
        # epsilon = 1./E * ((1+nu)*stress - nu*tr(stress)*Identity(2))

        return epsilon

    def m_operator(v_pel, al_pel, v_qel, al_qel, v_pt, al_pt):
        m_form = dot(v_pel, al_pel) * dx \
                 + inner(v_qel, al_qel) * dx \
                 + v_pt * al_pt * dx

        return m_form

    def j_operator(v_pel, e_pel, v_qel, e_qel, v_pt, e_pt):

        jel_grad = inner(v_qel, gradSym(e_pel)) * dx
        jel_gradIP = -inner(gradSym(v_pel), e_qel) * dx

        if delta != 0:
            jcoup_div  = - delta*fac*gamma * T_0 * v_pt * div(e_pel) * dx

        jcoup_divIP = gamma * T_0 * div(v_pel) * e_pt * dx


        if delta != 0:
            j_form = jel_grad + jel_gradIP + jcoup_div + jcoup_divIP
        else:
            j_form = jel_grad + jel_gradIP + jcoup_divIP

        return j_form

    def r_operator(v_pt, e_pt):
        r_form = dot(grad(v_pt), K*T_0*grad(e_pt)) * dx

        return r_form

    # The unit square mesh is divided in :math:`N\times N` quadrilaterals::

    L_hat = 10
    L_x = beta*L_hat
    L_y = L_x
    n_x = n
    n_y = n
    mesh = RectangleMesh(n_x, n_y, L_x, L_y, quadrilateral=False)

    # plot(mesh);
    # plt.show()


    # Finite element defition
    V_pel = VectorFunctionSpace(mesh, "CG", r)
    V_qel = VectorFunctionSpace(mesh, "DG", r-1, dim=3)
    V_pt = FunctionSpace(mesh, "CG", r)

    V = MixedFunctionSpace([V_pel, V_qel, V_pt])

    n_V = V.dim()
    print(n_V)

    v = TestFunction(V)
    v_pel, v_qel, v_pt = split(v)

    e = TrialFunction(V)
    e_pel, e_qel, e_pt = split(e)

    v_qel = as_tensor([[v_qel[0], v_qel[1]],
                       [v_qel[1], v_qel[2]]])

    e_qel = as_tensor([[e_qel[0], e_qel[1]],
                       [e_qel[1], e_qel[2]]])

    al_pel = rho * e_pel
    al_qel = compliance(e_qel)
    al_pt = rho * c_E * T_0 * e_pt


    dx = Measure('dx')
    ds = Measure('ds')

    t = 0

    t_fin = beta*tfin_hat/c_1     # total simulation time

    # J, M = PETScMatrix(), PETScMatrix()

    dt = t_fin/200
    theta = 0.5

    m_form = m_operator(v_pel, al_pel, v_qel, al_qel, v_pt, al_pt)
    j_form = j_operator(v_pel, e_pel, v_qel, e_qel, v_pt, e_pt)
    r_form = r_operator(v_pt, e_pt)

    bcs = []

    bc_t = DirichletBC(V.sub(2), Constant(1), 1)
    bcs.append(bc_t)
    lhs = m_form - dt*theta*(j_form -r_form)
    A = assemble(lhs, bcs = bcs, mat_type='aij')

    e_n1 = Function(V)
    e_n = Function(V)
    v_n1 = Function(V_pel)
    v_n = Function(V_pel)


    epel_n, eqel_n, ept_n = e_n.split()

    eqel_n = as_tensor([[eqel_n[0], eqel_n[1]],
                       [eqel_n[1], eqel_n[2]]
                       ])

    v_n.assign(Constant((0.0, 0.0)))

    n_t = int(floor(t_fin/dt) + 1)

    u_atP = np.zeros((n_t,))
    v_atP = np.zeros((n_t,))
    theta_atP = np.zeros((n_t,))

    x_hat = 1
    x_P = x_hat*beta
    y_hat = L_hat/2
    y_P = y_hat*beta


    Ppoint = (x_P, y_P)

    param = {"ksp_type": "gmres", "ksp_gmres_restart":100, "ksp_atol":1e-60}
    # param = {"ksp_type": "preonly", "pc_type": "lu"}

    for i in range(1, n_t):
        epel_n, eqel_n, ept_n = e_n.split()

        eqel_n = as_tensor([[eqel_n[0], eqel_n[1]],
                            [eqel_n[1], eqel_n[2]]
                            ])

        alpel_n = rho * epel_n
        alqel_n = compliance(eqel_n)
        alpt_n = rho * c_E * T_0 * ept_n

        rhs = m_operator(v_pel, alpel_n, v_qel, alqel_n, v_pt, alpt_n) \
              + dt * (1 - theta) * (j_operator(v_pel, epel_n, v_qel, eqel_n, v_pt, ept_n)\
                                    -r_operator(v_pt, ept_n))

        b = assemble(rhs, bcs = bcs)

        solve(A, e_n1, b, solver_parameters=param)

        t += dt

        epel_n1, eqel_n1, ept_n1 = e_n.split()

        # eqel_n1 = as_tensor([[eqel_n1[0], eqel_n1[1]],
        #                     [eqel_n1[1], eqel_n1[2]]
        #                     ])

        v_n1.assign(epel_n1)
        e_n.assign(e_n1)

        u_atP[i] = u_atP[i-1] + dt/2*(v_n.at(Ppoint)[0] + v_n1.at(Ppoint)[0])
        v_n.assign(v_n1)
        theta_atP[i] = ept_n1.at(Ppoint)
        print(i)

    t_vec_hat = np.linspace(0, tfin_hat, num=n_t)
    u_atP_hat = u_atP*(lamda + 2*mu)/(beta*gamma*T_0)

    return t_vec_hat, theta_atP, u_atP_hat


n = 100
r = 1
t_fin_hat = 4

t1, th1, u1 = compute_sol(n, r, 0, t_fin_hat)
t2, th2, u2 = compute_sol(n, r, 1, t_fin_hat)

t_an1, th_an1, disp_an1, stress_an1 = compute_analytical(0, 1, t_fin_hat)
t_an2, th_an2, disp_an2, stress_an2 = compute_analytical(1, 1, t_fin_hat)

path_fig = "/home/a.brugnoli/Plots/Python/Plots/Thermoelasticity/"
save_fig = True
plt.figure()

plt.plot(t1, th1, '-.', label=r'approx $\theta$ $\delta=0$')
plt.plot(t2, th2, '--', label=r'approx $\theta$ $\delta=1$')
plt.plot(t_an1, th_an1, '-', label=r'exact $\theta$ $\delta=0$')
plt.plot(t_an2, th_an2, ':', label=r'exact $\theta$ $\delta=1$')
plt.xlabel(r'Dimensionless Time')
plt.title(r'Dimensionless Temperature at ' + str(1))
plt.legend()

plt.savefig(path_fig + "temp_at1_2D_nodae", format="eps")
plt.figure()

plt.plot(t1, u1, '-.', label=r'approx $u_x$ $\delta=0$')
plt.plot(t2, u2, '-.', label=r'approx $u_x$ $\delta=1$')
plt.plot(t_an1, disp_an1, '-', label=r'exact $\theta$ $\delta=0$')
plt.plot(t_an2, disp_an2, ':', label=r'exact $\theta$ $\delta=1$')
plt.xlabel(r'Dimensionless Time')
plt.title(r'Dimensionless Displacement at ' + str(1))
plt.legend()

plt.savefig(path_fig + "disp_at1_2D_nodae", format="eps")
plt.show()