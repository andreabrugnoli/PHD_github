from fenics import *
from ufl.classes import AndCondition
from ufl.classes import OrCondition

import numpy as np
import scipy.linalg as la
from modules_phdae.classes_phsystem import SysPhdaeRig, check_positive_matrix
from Wave_fenics.ifac_wc.sim_CN.theta_method_wc import theta_method

from math import pi
import time


def computeH_ode(ind):

    q_1 = 0.8163  # m^3 kg ^-1   1/mu_0
    mu_0 = 1 / q_1
    
    q_2 = 1.4161 * 10 ** 5  # Pa   1/xsi
    xi_s = 1 / q_2
    c_0 = 340  # 340 m/s
   
   
    def create_sys(mesh1, mesh2, deg_p1, deg_q1, deg_p2, deg_q2):
        
    #    x1 = Expression("x[0]", degree=2)
    #    r1 = Expression("x[1]", degree=2)
        
        x1, r1 = SpatialCoordinate(mesh1)
    
        R_ext = 1
        L_duct = 2
        tol_geo = 1e-9
    
        Vp1 = FunctionSpace(mesh1, "CG", deg_p1)
        Vq1 = FunctionSpace(mesh1, "RT", deg_q1)
    
        np_1 = Vp1.dim()
        nq_1 = Vq1.dim()
        
        n_1 = np_1 + nq_1
    
        v_p1 = TestFunction(Vp1)
        v_q1 = TestFunction(Vq1)
    
        e_p1 = TrialFunction(Vp1)
        e_q1 = TrialFunction(Vq1)
    
        al_p1 = xi_s * e_p1
        al_q1 = mu_0 * e_q1
    
        dx1 = Measure('dx', domain=mesh1)
        ds1 = Measure('ds', domain=mesh1)
    
        m_p1 = v_p1 * al_p1 * r1 * dx1
        m_q1 = dot(v_q1, al_q1) * r1 * dx1
    
        M_p1 = assemble(m_p1).array()
        M_q1 = assemble(m_q1).array()
        
        M1 = la.block_diag(M_p1, M_q1)
        
        j_div = v_p1 * (div(e_q1) + e_q1[1]/r1) * r1 * dx1
        j_divIP = -(div(v_q1) + v_q1[1]/r1) * e_p1 * r1 * dx1
    
        J_div = assemble(j_div).array()
        
        J1 = np.zeros((n_1, n_1))
        J1[:np_1, np_1:] = J_div
        J1[np_1:, :np_1] = -J_div.T
    
        Vf1 = FunctionSpace(mesh1, "CG", 1)
        f_D = TrialFunction(Vf1)
        v_D = TestFunction(Vf1)
        u_D = TrialFunction(Vf1)
        
        d1 = mesh1.geometry().dim()
        tab_coord1 = Vf1.tabulate_dof_coordinates().reshape((-1, d1))
        x1_cor = tab_coord1[:, 0]
        y1_cor = tab_coord1[:, 1]
    
        n_ver1 = FacetNormal(mesh1)
    
        is_D = conditional(gt(r1, R_ext - tol_geo), 1, 0)
        is_int1 = conditional(lt(r1, R_ext - tol_geo), 1, 0)
        # is_int1 = conditional(ne(r1, R_ext), 1, 0)
    
        is_uD = conditional(AndCondition(gt(r1, R_ext - tol_geo), \
                            AndCondition(gt(x1, L_duct / 3), lt(x1, 2 * L_duct / 3))), 1, 0)
    
        b_D = dot(v_q1, n_ver1) * u_D * is_uD * r1 * ds1
    
        B_D = assemble(b_D).array()
    
        controlD_dofs = np.where(B_D.any(axis=0))[0]
    
        B_D = B_D[:, controlD_dofs]
        M_D = assemble(v_D * u_D * is_uD * r1 * ds1).array()
        M_D = M_D[:, controlD_dofs]
        M_D = M_D[controlD_dofs, :]
    
        nu_D = len(controlD_dofs)
    
        b_int1 = dot(v_q1, n_ver1) * f_D * is_int1 * r1 * ds1
    
        B_int1 = assemble(b_int1).array()
    
        perm_x1 = np.argsort(x1_cor)
        B_int1 = B_int1[:, perm_x1]
        
        m_delta = v_D * f_D * r1 * ds1
    
        Mdelta = assemble(m_delta).array()
    
        Mdelta = Mdelta[:, perm_x1]
        Mdelta = Mdelta[perm_x1, :]
    
        int_dofs1 = np.where(B_int1.any(axis=0))[0]
    
        Mdelta = Mdelta[:, int_dofs1]
        Mdelta = Mdelta[int_dofs1, :]
    
        B_int1 = B_int1[:, int_dofs1]
    
        nint_1 = len(int_dofs1)
    
        Bq1 = np.concatenate((B_int1, B_D), axis=1)
        
        m1 = Bq1.shape[1]
        Bp1 = np.zeros((np_1, m1))
        
        B1 = np.concatenate((Bp1, Bq1))
        sys1 = SysPhdaeRig(n_1, 0, 0, np_1, nq_1, E=M1, J=J1, B=B1)
    
        x2, r2 = SpatialCoordinate(mesh2)
    
        Vp2 = FunctionSpace(mesh2, "CG", deg_p2)
        Vq2 = FunctionSpace(mesh2, "RT", deg_q2)
    
        np_2 = Vp2.dim()
        nq_2 = Vq2.dim()
        n_2 = np_2 + nq_2
    
        v_p2 = TestFunction(Vp2)
        v_q2 = TestFunction(Vq2)
    
        e_p2 = TrialFunction(Vp2)
        e_q2 = TrialFunction(Vq2)
    
        al_p2 = xi_s * e_p2
        al_q2 = mu_0 * e_q2
    
        dx2 = Measure('dx', domain=mesh2)
        ds2 = Measure('ds', domain=mesh2)
    
        m_p2 = v_p2 * al_p2 * r2 * dx2
        m_q2 = dot(v_q2, al_q2) * r2 * dx2
        
        M_p2 = assemble(m_p2).array()
        M_q2 = assemble(m_q2).array()
        
        M2 = la.block_diag(M_p2, M_q2)
        
        j_grad = dot(v_q2, grad(e_p2)) * r2 * dx2
        j_gradIP = -dot(grad(v_p2), e_q2) * r2 * dx2
        
        J_grad = assemble(j_grad).array()
        
        J2 = np.zeros((n_2, n_2))
        J2[np_2:, :np_2] = J_grad
        J2[:np_2, np_2:] = -J_grad.T
    
        Vf2 = FunctionSpace(mesh2, "CG", 1)
        f_N = TrialFunction(Vf2)
        v_N = TestFunction(Vf2)
        u_N = TrialFunction(Vf2)
        
        d2 = mesh2.geometry().dim()
        tab_coord2 = Vf2.tabulate_dof_coordinates().reshape((-1, d2))
        x2_cor = tab_coord2[:, 0]
        y2_cor = tab_coord2[:, 1]
    
        isL_uN = conditional(lt(x2, tol_geo), 1, 0)
        isR_uN = conditional(gt(x2, L_duct - tol_geo), 1, 0)
    
        is_N = conditional(OrCondition(OrCondition(lt(x2, tol_geo),\
                                gt(x2, L_duct - tol_geo)), lt(r2, tol_geo)), 1, 0)
        is_int2 = conditional(AndCondition(AndCondition(gt(x2, tol_geo),\
                        lt(x2, L_duct-tol_geo)), gt(r2, R_ext/2-tol_geo)), 1, 0)
    
        ux_N1 = 1 - r1**2/R_ext**2
        ux_N2 = 1 - r2**2/R_ext**2
    
        uy_N1 = 16*r1**2*(R_ext-r1)**2
        uy_N2 = 16*r2**2*(R_ext-r2)**2
    
        b_N = -v_p2 * ux_N2 * isL_uN * r2 * ds2 + v_p2 * ux_N2 * isR_uN * r2 * ds2
    
        B_N = np.reshape(assemble(b_N).get_local(), (-1, 1))
    
        b_int2 = v_p2 * f_N * is_int2 * r2 * ds2
    
        B_int2 = assemble(b_int2).array()
    
        perm_x2 = np.argsort(x2_cor)
        B_int2 = B_int2[:, perm_x2]
    
        int_dofs2 = np.where(B_int2.any(axis=0))[0]
    
        B_int2 = B_int2[:, int_dofs2]
        #
        # x1_cor = x1_cor[perm_x1]
        # r1_cor = r1_cor[perm_x1]
        # x2_cor = x2_cor[perm_x2]
        # r2_cor = r2_cor[perm_x2]
        #
        # plt.plot(x1_cor[:], r1_cor[:], 'bo')
        # plt.plot(x1_cor[int_dofs1], r1_cor[int_dofs1], 'r*')
        # plt.show()
        #
        # plt.plot(x2_cor[:], r2_cor[:], 'bo')
        # plt.plot(x2_cor[int_dofs2], r2_cor[int_dofs2], 'r*')
        # plt.show()
    
        nint_2 = len(int_dofs2)
    
        Bp2 = np.concatenate((B_int2, B_N), axis=1)
        m2 = Bp2.shape[1]
        Bq2 = np.zeros((nq_2, m2))
        
        B2 = np.concatenate((Bp2, Bq2))
        
        sys2 = SysPhdaeRig(n_2, 0, 0, np_2, nq_2, E=M2, J=J2, B=B2)
    
        m1 = sys1.m
        m2 = sys2.m
        sys_DN = SysPhdaeRig.gyrator_ordered(sys1, sys2, list(range(nint_1)), list(range(nint_2)), la.inv(Mdelta))
    
        ep1_0 = project(Constant(0), Vp1).vector().get_local()
        eq1_0 = project(as_vector([ux_N1, uy_N1]), Vq1).vector().get_local()
    
        ep2_0 = project(Constant(0), Vp2).vector().get_local()
        eq2_0 = project(as_vector([ux_N2, uy_N2]), Vq2).vector().get_local()
    
        ep_0 = np.concatenate((ep1_0, ep2_0))
        eq_0 = np.concatenate((eq1_0, eq2_0))
    
        return sys_DN, Vp1, Vp2, ep_0, eq_0, M_D
    
    
    path_mesh = "/home/a.brugnoli/GitProjects/PythonProjects/ph_fenics/Wave_fenics/meshes_ifacwc/"
    mesh1 = Mesh(path_mesh + "duct_dom1_" + str(ind) + ".xml")
    mesh2 = Mesh(path_mesh + "duct_dom2_" + str(ind) + ".xml")
    
    # figure = plt.figure()
    # ax = figure.add_subplot(111)
    # plot(mesh1, axes=ax)
    # plot(mesh2, axes=ax)
    # plt.show()
    
    degp1 = 1
    degq1 = 2
    
    degp2 = 1
    degq2 = 2
    
    sys_DN, Vp1, Vp2, ep_0, eq_0, M_D = create_sys(mesh1, mesh2, degp1, degq1, degp2, degq2)
    JJ = sys_DN.J
    MM = sys_DN.E
    BB = sys_DN.B
        
    m_sysDN = sys_DN.m
    n_p = len(ep_0)
    n_q = len(eq_0)
    n_e = n_p + n_q    
    
    B_D = BB[:, :-1]
    B_N = BB[:, -1]
    
    t_final = 0.1
    Z = c_0 * mu_0
    
    RR = Z * B_D @ la.inv(M_D) @ B_D.T
    
    y0 = np.concatenate((ep_0, eq_0))
    
    # Simulate
    ti_sim = time.time()
    t_sol, e_sol = theta_method(MM, JJ, RR, B_N, y0)
    tf_sim = time.time()
    
    elapsed_t = tf_sim - ti_sim
    
    n_ev = len(t_sol)
    dt_vec = np.diff(t_sol)
    
    n_p1 = Vp1.dim()
    n_p2 = Vp2.dim()
    
    ep_sol = e_sol[:n_p1 + n_p2, :]
    eq_sol = e_sol[n_p1 + n_p2:, :]
    ep1_sol = ep_sol[:n_p1, :]
    ep2_sol = ep_sol[n_p1:, :]
    
    MMp = MM[:n_p1 + n_p2, :n_p1 + n_p2]
    MMq = MM[n_p1 + n_p2:, n_p1 + n_p2:]
    
    H_vec = np.zeros((n_ev,))
    Hp_vec = np.zeros((n_ev,))
    Hq_vec = np.zeros((n_ev,))
    
    for i in range(n_ev):
        H_vec[i] = 0.5 * (e_sol[:, i].T @ MM @ e_sol[:, i])
        Hp_vec[i] = 0.5 * (ep_sol[:, i].T @ MMp @ ep_sol[:, i])
        Hq_vec[i] = 0.5 * (eq_sol[:, i].T @ MMq @ eq_sol[:, i])
    
    
    path_results = "/home/a.brugnoli/LargeFiles/results_ifacwc_CN2/"
    np.save(path_results + "t_ode_" + str(ind) + ".npy", t_sol)
    np.save(path_results + "H_ode_" + str(ind) + ".npy", H_vec)
    np.save(path_results + "Hp_ode_" + str(ind) + ".npy", Hp_vec)
    np.save(path_results + "Hq_ode_" + str(ind) + ".npy", Hq_vec)
    np.save(path_results + "ep_ode_" + str(ind) + ".npy", ep_sol)
    np.save(path_results + "eq_ode_" + str(ind) + ".npy", eq_sol)
    np.save(path_results + "t_elapsed_ode_" + str(ind) + ".npy", elapsed_t)

    return H_vec, t_sol

