from fenics import *
from ufl.classes import AndCondition
from ufl.classes import OrCondition
import numpy as np
from scipy import linalg as la
np.set_printoptions(threshold=np.inf)
from assimulo.solvers import IDA
from assimulo.implicit_ode import Implicit_Problem

#from matplotlib import pyplot as plt
#from tools_plotting.animate_surf import animate2D
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from matplotlib import animation
#plt.rc('text', usetex=True)
# Finite element defition
import time

def computeH_dae(ind):

    # The unit square mesh is divided in :math:`N\times N` quadrilaterals::
    path_mesh = "/home/a.brugnoli/GitProjects/PythonProjects/ph_fenics/Wave_fenics/meshes_ifacwc/"
    # path_mesh = "./meshes_ifacwc/"
    
    mesh = Mesh(path_mesh + "duct_" + str(ind) + ".xml")
    
    #figure = plt.figure()
    #ax = figure.add_subplot(111)
    #plot(mesh, axes=ax)
    #plt.show()
    
    #path_figs = "/home/a.brugnoli/Plots_Videos/Python/Plots/Waves/IFAC_WC2020/"
    #plt.savefig(path_figs + "Mesh_" + str(ind) + ".eps", format="eps")
    
    q_1 = 0.8163  #   1/mu_0
    mu_0 = 1/q_1  # m^3 kg ^-1
    
    q_2 = 1.4161 * 10**5  # Pa   1/xsi
    xi_s = 1/q_2
    c_0 = 340  # 340 m/s
    
    deg_p = 1
    deg_q = 2
    Vp = FunctionSpace(mesh, "CG", deg_p)
    Vq = FunctionSpace(mesh, "RT", deg_q)
    
    n_p = Vp.dim()
    n_q = Vq.dim()
    n_e = n_p + n_q
    
    v_p = TestFunction(Vp)
    v_q = TestFunction(Vq)
    
    e_p = TrialFunction(Vp)
    e_q = TrialFunction(Vq)
    
    al_p = xi_s * e_p
    al_q = mu_0 * e_q
    
    x, r = SpatialCoordinate(mesh)
    tol_geo = 1e-9
    R_ext = 1
    L_duct = 2
    
    dx = Measure('dx')
    ds = Measure('ds')
    m_p = v_p * al_p * r * dx
    m_q = dot(v_q, al_q) * r * dx
    
    M_p = assemble(m_p).array()
    M_q = assemble(m_q).array()
    
    MM = la.block_diag(M_p, M_q)
    
    j_grad = dot(v_q, grad(e_p)) * r * dx
    j_gradIP = -dot(grad(v_p), e_q) * r * dx
    
    J_grad = assemble(j_grad).array()
    
    JJ = np.zeros((n_e, n_e))
    JJ[n_p:, :n_p] = J_grad
    JJ[:n_p, n_p:] = -J_grad.T
    
    n_ver = FacetNormal(mesh)
    
    isL_uN = conditional(lt(x, tol_geo), 1, 0)
    isR_uN = conditional(gt(x, L_duct - tol_geo), 1, 0)
    isLR_uN = conditional(OrCondition(lt(x, tol_geo), gt(x, L_duct - tol_geo)), 1, 0)
    
    # u_N = (1 - cos(pi*r/R_ext)) * cos(pi*x/(2*L_duct))
    ux_N = 1 - r**2/R_ext**2
    uy_N = 16*r**2*(R_ext-r)**2
    
    b_N = -v_p * isL_uN * ux_N * r * ds + v_p * isR_uN * ux_N * r * ds
    
    Bp_N = assemble(b_N).get_local()
    
    Bq_N = np.zeros((n_q, ))
    
    B_N = np.concatenate((Bp_N, Bq_N))
    
    # controlN_dofs = np.where(B_N)[0]
    
    # B matrices based on Lagrange
    V_bc = FunctionSpace(mesh, 'CG', 1)
    lmb_D = TrialFunction(V_bc)
    v_D = TestFunction(V_bc)
    u_D = TrialFunction(V_bc)
    
    d = mesh.geometry().dim()
    tab_coord = V_bc.tabulate_dof_coordinates().reshape((-1, d))
    
    x_cor = tab_coord[:, 0]
    r_cor = tab_coord[:, 1]
    
    assert max(x_cor) == L_duct
    ind_x = np.where(np.logical_and(np.isclose(x_cor, 0),\
                                    np.isclose(r_cor, 0, rtol=1e-2)))[0][0]
    
    is_lmbD = conditional(gt(r, R_ext - tol_geo), 1, 0)
    g_D = v_p * lmb_D * is_lmbD * r * ds
    
    is_uD = conditional(AndCondition(gt(r, R_ext - tol_geo),\
                                     AndCondition(gt(x, L_duct/3), lt(x, 2*L_duct/3))), 1, 0)
    
    b_D = v_D * u_D * is_uD * r * ds
    
    Gp_D = assemble(g_D).array()
    
    Gq_D = np.zeros((n_q, Gp_D.shape[1]))
    
    G_D = np.concatenate((Gp_D, Gq_D), axis=0)
    
    B_D = assemble(b_D).array()
    
    dirichlet_dofs = np.where(G_D.any(axis=0))[0]
    G_D = G_D[:, dirichlet_dofs]
    
    controlD_dofs = np.where(B_D.any(axis=0))[0]
    B_D = B_D[:, controlD_dofs]
    M_D = B_D[controlD_dofs, :]

    B_D = B_D[dirichlet_dofs, :]
    
    # plt.plot(x_cor[:], r_cor[:], 'bo')
    # plt.plot(x_cor[dirichlet_dofs], r_cor[dirichlet_dofs], 'r*')
    # plt.plot(x_cor[controlD_dofs], r_cor[controlD_dofs], 'g*')
    # plt.show()
    
    n_uD = B_D.shape[1]
    n_lmb = G_D.shape[1]
    
    t_final = 0.1
    
    Z = mu_0 * c_0
    t_diss = 0.2*t_final
    
    tau_imp = t_final/100
    if ind != 15:
        invMM = la.inv(MM)
        
    RR = Z * B_D @ la.inv(M_D) @ B_D.T
    
    def dae_closed_phs(t, y, yd):
    
        print(t/t_final*100)
    
        e_var = y[:n_e]
        lmb_var = y[n_e:]
        ed_var = yd[:n_e]
    
        ft_imp = (t>t_diss) # * (1 - np.exp((t - t_diss)/tau_imp))
        ft_ctrl = 1  #  (t<t_diss)
        res_e = ed_var - invMM @ (JJ @ e_var + G_D @ lmb_var + B_N * ft_ctrl)
        res_lmb = - G_D.T @ e_var - RR @ lmb_var * ft_imp
    
        return np.concatenate((res_e, res_lmb))
    
    
    def dae_closed_phs_ref(t, y, yd):
    
        print(t / t_final * 100)
        
        e_var = y[:n_e]
        lmb_var = y[n_e:]
        ed_var = yd[:n_e]
        
        ft_imp = (t > t_diss)  # * (1 - np.exp((t - t_diss)/tau_imp))
        ft_ctrl = 1  # (t<t_diss)
        res_e = MM @ ed_var - (JJ @ e_var + G_D @ lmb_var + B_N * ft_ctrl)
        res_lmb = - G_D.T @ e_var - Z * B_D @ B_D.T @ lmb_var * ft_imp
    
        return np.concatenate((res_e, res_lmb))
    
    
    order = []
    
    
    def handle_result(solver, t, y, yd):
    
        order.append(solver.get_last_order())
    
        solver.t_sol.extend([t])
        solver.y_sol.extend([y])
        solver.yd_sol.extend([yd])
    
    
    ep_0 = np.zeros(n_p)
    eq_0 = project(as_vector([ux_N, uy_N]), Vq).vector().get_local()
    
    e_0 = np.concatenate((ep_0, eq_0))
    
    lmb_0 = la.solve(- G_D.T @ la.solve(MM, G_D), G_D.T @ la.solve(MM, JJ @ e_0 + B_N))
    
    y0 = np.zeros(n_e + n_lmb)  # Initial conditions
    
    de_0 = la.solve(MM, JJ @ e_0 + G_D @ lmb_0 + B_N)
    dlmb_0 = la.solve(- G_D.T @ la.solve(MM, G_D), G_D.T @ la.solve(MM, JJ @ de_0))
    
    y0[:n_p] = ep_0
    y0[n_p:n_e] = eq_0
    # y0[n_V:] = lmb_0
    #
    yd0 = np.zeros(n_e + n_lmb)  # Initial conditions
    # yd0[:n_V] = de_0
    # yd0[n_V:] = dlmb_0
    
    # Maug = la.block_diag(MM, np.zeros((n_lmb, n_lmb)))
    # Jaug = la.block_diag(JJ, np.zeros((n_lmb, n_lmb)))
    # Jaug[:n_e, n_e:] = G_D
    # Jaug[n_e:, :n_e] = -G_D.T
    
    
    # Create an Assimulo implicit problem
    if ind!=15:
        imp_mod = Implicit_Problem(dae_closed_phs, y0, yd0, name='dae_closed_pHs')
    else:
        imp_mod = Implicit_Problem(dae_closed_phs_ref, y0, yd0, name='dae_closed_pHs')
            
    imp_mod.handle_result = handle_result
    
    # Set the algebraic components
    imp_mod.algvar = list(np.concatenate((np.ones(n_e), np.zeros(n_lmb))))
    
    # Create an Assimulo implicit solver (IDA)
    imp_sim = IDA(imp_mod)  # Create a IDA solver
    
    # Sets the paramters
    imp_sim.atol = 1e-6  # Default 1e-6
    imp_sim.rtol = 1e-6  # Default 1e-6
    imp_sim.suppress_alg = True  # Suppress the algebraic variables on the error test
    imp_sim.report_continuously = True
    # imp_sim.maxh = 1e-6
    
    # Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
    imp_sim.make_consistent('IDA_YA_YDP_INIT')
    
    # Simulate
    n_ev = 500
    t_ev = np.linspace(0, t_final, n_ev)
    ti_sim = time.time()
    t_sol, y_sol, yd_sol = imp_sim.simulate(t_final, 0, t_ev)
    tf_sim = time.time()
    
    elapsed_t = tf_sim - ti_sim
        
    e_sol = y_sol[:, :n_e].T
    lmb_sol = y_sol[:, n_e:].T
    
    ep_sol = e_sol[:n_p, :]
    eq_sol = e_sol[n_p:, :]
    MMp = MM[:n_p, :n_p]
    MMq = MM[n_p:, n_p:]
    
    maxZ = np.max(ep_sol)
    minZ = np.min(ep_sol)
    
    
    H_vec = np.zeros((n_ev,))
    Hp_vec = np.zeros((n_ev,))
    Hq_vec = np.zeros((n_ev,))
    
    for i in range(n_ev):
        H_vec[i] = 0.5 * (e_sol[:, i].T @ MM @ e_sol[:, i])
        Hp_vec[i] = 0.5 * (ep_sol[:, i].T @ MMp @ ep_sol[:, i])
        Hq_vec[i] = 0.5 * (eq_sol[:, i].T @ MMq @ eq_sol[:, i])
    
    
    path_results = "/home/a.brugnoli/GitProjects/PythonProjects/ph_fenics/Wave_fenics/results_ifacwc/"
    np.save(path_results + "t_dae_" + str(ind) + ".npy", t_sol)
    np.save(path_results + "H_dae_" + str(ind) + ".npy", H_vec)
    np.save(path_results + "Hp_dae_" + str(ind) + ".npy", Hp_vec)
    np.save(path_results + "Hq_dae_" + str(ind) + ".npy", Hq_vec)
    np.save(path_results + "ep_dae_" + str(ind) + ".npy", ep_sol)
    np.save(path_results + "eq_dae_" + str(ind) + ".npy", eq_sol)
    np.save(path_results + "t_elapsed_dae_" + str(ind) + ".npy", elapsed_t)
    
#    fntsize = 16
    
#    fig = plt.figure()
#    plt.plot(t_ev, H_vec, 'b-', label= "$H$")
#    plt.plot(t_ev, Hp_vec, 'r-', label= "$H_p$")
#    plt.plot(t_ev, Hq_vec, 'g-', label= "$H_v$")
#    
#    plt.xlabel(r'{Time} (s)', fontsize=fntsize)
#    plt.ylabel(r'{Hamiltonian} (J)', fontsize=fntsize)
#    plt.title(r"Hamiltonian trend for the reference solution",
#              fontsize=fntsize)
#    plt.legend(loc='upper right')
    
    #path_figs = "/home/a.brugnoli/Plots_Videos/Python/Plots/Waves/IFAC_WC2020/"
    #plt.savefig(path_figs + "H_dae" + str(ind) + ".eps", format="eps")
    
    #wfun_vec = []
    #w_fun = Function(Vp)
    #
    #for i in range(n_ev):
    #    w_fun.vector()[:] = ep_sol[:, i]
    #    wfun_vec.append(interpolate(w_fun, Vp))
    #
    #anim = animate2D(minZ, maxZ, wfun_vec, t_ev, xlabel = '$x[m]$', ylabel = '$r [m]$', \
    #                          zlabel='$p$', title='pressure')
    
    #rallenty = 10
    #fps = 20
    #Writer = animation.writers['ffmpeg']
    #writer = Writer(fps=fps, metadata=dict(artist='Me'), bitrate=1800)
    #path_videos = "/home/a.brugnoli/Plots_Videos/Python/Videos/Waves/IFAC_WC2020/"
    #anim.save(path_videos + 'wave_dae' + str(ind) + '.mp4', writer=writer)

#    plt.show()

    return H_vec, t_sol
