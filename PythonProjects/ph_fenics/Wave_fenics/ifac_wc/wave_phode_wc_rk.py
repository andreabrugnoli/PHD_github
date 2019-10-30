from fenics import *
from ufl.classes import AndCondition
from ufl.classes import OrCondition
import numpy as np
import scipy.linalg as la
from modules_phdae.classes_phsystem import SysPhdaeRig, check_positive_matrix
import warnings
np.set_printoptions(threshold=np.inf)
from scipy import integrate

from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
#import matplotlib.animation as animation
import time

from math import pi
plt.rc('text', usetex=True)

q_1 = 0.8163  # m^3 kg ^-1   1/mu_0
mu_0 = 1 / q_1

q_2 = 1.4161 * 10 ** 5  # Pa   1/xsi
xi_s = 1 / q_2
c_0 = 340  # 340 m/s
ind = 4


def print_modes(sys, Vp1, Vp2, n_modes):
    tab_coord1 = Vp1.tabulate_dof_coordinates().reshape((-1, 2))
    x1_cor = tab_coord1[:, 0]
    y1_cor = tab_coord1[:, 1]
    
    tab_coord2 = Vp2.tabulate_dof_coordinates().reshape((-1, 2))
    x2_cor = tab_coord2[:, 0]
    y2_cor = tab_coord2[:, 1]

    eigenvalues, eigvectors = la.eig(sys.J_f, sys.M_f)
    omega_all = np.imag(eigenvalues)

    index = omega_all > 0

    omega = omega_all[index]
    eigvec_omega = eigvectors[:, index]
    perm = np.argsort(omega)
    eigvec_omega = eigvec_omega[:, perm]

    omega.sort()

    fntsize = 15

    n_Vp1 = Vp1.dim()
    n_Vp2 = Vp2.dim()

    for i in range(int(n_modes)):
        print("Eigenvalue num " + str(i+1) + ":" + str(omega[i]))
#        eig_real_p1 = Function(Vp1)
#        eig_imag_p1 = Function(Vp1)

        eig_real_p1 = np.real(eigvec_omega[:n_Vp1, i])
        eig_imag_p1 = np.imag(eigvec_omega[:n_Vp1, i])

#        eig_real_p2 = Function(Vp2)
#        eig_imag_p2 = Function(Vp2)

        eig_real_p2 = np.real(eigvec_omega[n_Vp1:n_Vp1 + n_Vp2, i])
        eig_imag_p2 = np.imag(eigvec_omega[n_Vp1:n_Vp1 + n_Vp2, i])

        norm_real_eig = np.linalg.norm(np.real(eigvec_omega[:n_Vp1+n_Vp2, i]))
        norm_imag_eig = np.linalg.norm(np.imag(eigvec_omega[:n_Vp1+n_Vp2, i]))

        figure = plt.figure()
        ax = figure.add_subplot(111, projection="3d")

        if norm_imag_eig > norm_real_eig:
            tri_surf1 = ax.plot_trisurf(x1_cor, y1_cor, eig_imag_p1, cmap=cm.jet, linewidth=0, antialiased=False)
            tri_surf2 = ax.plot_trisurf(x2_cor, y2_cor, eig_imag_p2, cmap=cm.jet, linewidth=0, antialiased=False)

        else:
            tri_surf1 = ax.plot_trisurf(x1_cor, y1_cor, eig_real_p1, cmap=cm.jet, linewidth=0, antialiased=False)
            tri_surf2 = ax.plot_trisurf(x2_cor, y2_cor, eig_real_p2, cmap=cm.jet, linewidth=0, antialiased=False)
            
        tri_surf1._facecolors2d = tri_surf1._facecolors3d
        tri_surf1._edgecolors2d = tri_surf1._edgecolors3d
        
        tri_surf2._facecolors2d = tri_surf2._facecolors3d
        tri_surf2._edgecolors2d = tri_surf2._edgecolors3d

        ax.set_xlabel('$x [m]$', fontsize=fntsize)
        ax.set_ylabel('$y [m]$', fontsize=fntsize)
        ax.set_title('Eigenvector num ' + str(i + 1), fontsize=fntsize)

        ax.w_zaxis.set_major_locator(LinearLocator(10))
        ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

        ax.legend(("mesh 1", "mesh2"))

#        path_figs = "/home/a.brugnoli/Plots_Videos/Python/Plots/Waves/"
        # plt.savefig(path_figs + "Eig_n" + str(i) + ".eps")

    plt.show()
    

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
    # print(nint_1, nint_2)

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

print_modes(sys_DN, Vp1, Vp2, 5)

m_sysDN = sys_DN.m
n_p = len(ep_0)
n_q = len(eq_0)
n_e = n_p + n_q
print(n_e)


B_D = BB[:, :-1]
B_N = BB[:, -1]
print(m_sysDN, B_D.shape, B_N.shape)

t_final = 0.1
Z = c_0 * mu_0
t_diss = 0.2*t_final
tau_imp = t_final/100

if ind != 15:
    invMM = la.inv(MM)

RR = Z * B_D @ la.inv(M_D) @ B_D.T

def ode_closed_phs(t, y):

    print(t/t_final*100)

    ft_imp = (t>t_diss) # * (1 - np.exp((t - t_diss)/tau_imp))
    ft_ctrl = 1 # (t<t_diss)
    yd = invMM @ ((JJ - RR * ft_imp) @ y + B_N * ft_ctrl)

    return yd


def ode_closed_phs_ref(t, y):

    print(t/t_final*100)

    ft_imp = (t>t_diss) # * (1 - np.exp((t - t_diss)/tau_imp))
    ft_ctrl = 1 # (t<t_diss)
    dote = la.solve(MM, ((JJ - RR * ft_imp) @ y + B_N * ft_ctrl), assume_a='pos')
    yd = dote

    return yd

order = []


y0 = np.concatenate((ep_0, eq_0))

# Simulate
n_ev = 500
t_ev = np.linspace(0, t_final, n_ev)
t_span = [0, t_final]
options = {"atol": 1e-6, "rtol": 1e-6}
if ind!=15:
    prob = ode_closed_phs
else:
    prob = ode_closed_phs_ref

ti_sim = time.time()
sol = integrate.solve_ivp(prob, t_span, y0, method='RK45', vectorized=False, t_eval=t_ev, **options)
tf_sim = time.time()

elapsed_t = tf_sim - ti_sim

t_sol = sol.t
e_sol = sol.y

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


path_results = "/home/a.brugnoli/GitProjects/PythonProjects/ph_fenics/Wave_fenics/results_ifacwc/"
np.save(path_results + "t_ode_" + str(ind) + ".npy", t_sol)
np.save(path_results + "H_ode_" + str(ind) + ".npy", H_vec)
np.save(path_results + "Hp_ode_" + str(ind) + ".npy", Hp_vec)
np.save(path_results + "Hq_ode_" + str(ind) + ".npy", Hq_vec)
np.save(path_results + "ep_ode_" + str(ind) + ".npy", ep_sol)
np.save(path_results + "eq_ode_" + str(ind) + ".npy", eq_sol)
np.save(path_results + "t_elapsed_ode_" + str(ind) + ".npy", elapsed_t)

fntsize = 16

fig = plt.figure()
plt.plot(t_ev, H_vec, 'b-', label= "H")
plt.plot(t_ev, Hp_vec, 'r-', label= "Hp")
plt.plot(t_ev, Hq_vec, 'g-', label= "Hq")

plt.xlabel(r'{Time} (s)', fontsize=fntsize)
plt.ylabel(r'{Hamiltonian} (J)', fontsize=fntsize)
plt.title(r"Hamiltonian trend",
          fontsize=fntsize)
plt.legend(loc='upper right')

#path_figs = "/home/a.brugnoli/Plots_Videos/Python/Plots/Waves/IFAC_WC2020/"

#plt.savefig(path_figs + "H_ode" + str(ind) + ".eps", format="eps")

#dt_vec = np.diff(t_ev)
#
#w1_fun = Function(Vp1)
#w2_fun = Function(Vp2)
#w1fun_vec = []
#w2fun_vec = []
#
#maxZ = np.max(ep_sol)
#minZ = np.min(ep_sol)
#
#for i in range(n_ev):
#    w1_fun.vector()[:] = ep1_sol[:, i]
#    w1fun_vec.append(interpolate(w1_fun, Vp1))
#
#    w2_fun.vector()[:] = ep2_sol[:, i]
#    w2fun_vec.append(interpolate(w2_fun, Vp2))
#
#anim = animate2D(minZ, maxZ, w1fun_vec, w2fun_vec, t_ev, xlabel = '$x[m]$', ylabel = '$r [m]$', \
#                         zlabel = '$p [Pa]$', title = 'Pressure')
#
#Writer = animation.writers['ffmpeg']
#writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
#path_videos = "/home/a.brugnoli/Plots_Videos/Python/Videos/Waves/IFAC_WC2020/"
#anim.save(path_videos + 'wave_ode' + str(ind) + '.mp4', writer=writer)

plt.show()

