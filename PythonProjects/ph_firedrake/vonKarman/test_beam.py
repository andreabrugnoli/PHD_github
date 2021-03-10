# Implementation of mixed finite elements for von Karman beams
from firedrake import *
from firedrake.plot import calculate_one_dim_points

import numpy as np
np.set_printoptions(threshold=np.inf)
from math import pi, floor

# Matplotlib settings
import matplotlib
import matplotlib.pyplot as plt
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

matplotlib.rcParams['text.usetex'] = True


save_res = True
# Physical coefficients

rho = 2700
E = 70 * 10**9

# Geometrical coefficients 

L = 1
height = 0.1
wid = 0.1

A = height*wid
I = wid*height**3/12


def m_operator(v_u, v_eps, v_w, v_kap, v_disp, \
                   e_u, e_eps, e_w, e_kap, e_disp):
        
        al_u = rho * A * e_u
        al_eps = 1/(E*A) * e_eps
        
        al_w = rho * A * e_w
        al_kap = 1/(E*I) * e_kap
        
        m_form = v_u * al_u * dx \
               + v_eps * al_eps * dx \
               + v_w * al_w * dx \
               + v_kap * al_kap * dx \
               + v_disp * e_disp * dx
               
        return m_form
    
    
def j_operator(v_u, v_eps, v_w, v_kap, v_disp, \
               e_u, e_eps, e_w, e_kap, e_disp):
    
    j_axial = v_eps * e_u.dx(0) * dx \
            - v_u.dx(0) * e_eps * dx
            
    j_bend = v_w.dx(0) * e_kap.dx(0) * dx \
           - v_kap.dx(0) * e_w.dx(0) * dx
    
    j_coup = v_eps * e_disp.dx(0) * e_w.dx(0) * dx \
           - v_w.dx(0) * e_disp.dx(0) * e_eps * dx
    
    m_w = v_disp * e_w * dx
    
    j_form = j_axial + j_bend + j_coup + m_w
    
    return j_form 


def compute_err(n_elem, deg):

    mesh = IntervalMesh(n_elem, L)
    
    V_u = FunctionSpace(mesh, "CG", deg)
    V_eps = FunctionSpace(mesh, "DG", deg-1)
    V_w = FunctionSpace(mesh, "CG", deg)
    V_kap = FunctionSpace(mesh, "CG", deg)
    V_disp = FunctionSpace(mesh, "CG", deg)
    
    
    V = V_u * V_eps * V_w * V_kap * V_disp
    
    print("Number of dofs: " +  str(V.dim()))

    v_u, v_eps, v_w, v_kap, v_disp = TestFunctions(V)

    bcs = []
    bc_u = DirichletBC(V.sub(0), Constant(0.0), "on_boundary")
    bc_w = DirichletBC(V.sub(2), Constant(0.0), "on_boundary")
    bc_kap = DirichletBC(V.sub(3), Constant(0.0), "on_boundary")
    
    bcs.append(bc_u)
    bcs.append(bc_w)
    bcs.append(bc_kap)

    t = 0.
    t_fin = 1        # total simulation time

    dt = 1/(2*pi*n_elem)
    t_ = Constant(t)
    t_1 = Constant(t+dt)
    theta = 0.5
    
    x = mesh.coordinates
    
    T_u = t_fin
    omega_u = 2*pi/T_u*t_fin
    T_w = t_fin
    omega_w = 2*pi/T_w*t_fin
    
    # omega_u = 1
    # omega_w = 1
    
    u_ex = x[0]*(1-x[0]/L)*sin(omega_u*t_)    
    e_u_ex = omega_u*x[0]*(1-x[0]/L)*cos(omega_u*t_)
    dtt_u_ex = - omega_u**2 * x[0]*(1-x[0]/L)*sin(omega_u*t_)
    
    w_ex = sin(pi*x[0]/L)*sin(omega_w*t_)
    e_w_ex = omega_w*sin(pi*x[0]/L) * cos(omega_w*t_)
    dtt_w_ex = - omega_w**2*sin(pi*x[0]/L)*sin(omega_w*t_)

    e_eps_ex = E*A*(u_ex.dx(0) + 0.5*(w_ex.dx(0))**2)
    e_kap_ex = E*I*w_ex.dx(0).dx(0)

    f_u = rho*A*dtt_u_ex - e_eps_ex.dx(0)
    f_w = rho*A*dtt_w_ex + e_kap_ex.dx(0).dx(0) - (e_eps_ex*w_ex.dx(0)).dx(0)
    
    f_form = v_u*f_u*dx + v_w*f_w*dx
    
    u_ex1 = x[0]*(1-x[0]/L)*sin(omega_u*t_1)    
    e_u_ex1 = omega_u*x[0]*(1-x[0]/L)*cos(omega_u*t_1)
    dtt_u_ex1 = - omega_u**2*x[0]*(1-x[0]/L)*sin(omega_u*t_1)

    w_ex1 = sin(pi*x[0]/L)*sin(omega_w*t_1)
    e_w_ex1 = omega_w*sin(pi*x[0]/L) * cos(omega_w*t_1)
    dtt_w_ex1 = - omega_w**2*sin(pi*x[0]/L)*sin(omega_w*t_1)

    e_eps_ex1 = E*A*(u_ex1.dx(0) + 0.5*(w_ex1.dx(0))**2)
    e_kap_ex1 = E*I*w_ex1.dx(0).dx(0)

    f_u1 = rho*A*dtt_u_ex1 - e_eps_ex1.dx(0)
    f_w1 = rho*A*dtt_w_ex1 + e_kap_ex1.dx(0).dx(0) - (e_eps_ex1*w_ex1.dx(0)).dx(0)
   
    f_form1 = v_u*f_u1*dx + v_w*f_w1*dx


    e_n = Function(V,  name="e old")
    e_n1 = Function(V,  name="e new")

    e_n.sub(0).assign(project(e_u_ex, V_u))
    e_n.sub(2).assign(project(e_w_ex, V_w))
    
    e_u_n, e_eps_n, e_w_n, e_kap_n, e_disp_n = e_n.split()

    n_t = int(floor(t_fin/dt) + 1)
    t_vec = np.linspace(0, t_fin, num=n_t)

    e_u_err_H1 = np.zeros((n_t,))
    e_eps_err_L2 = np.zeros((n_t,))
    
    e_w_err_H1 = np.zeros((n_t,))
    e_kap_err_H1 = np.zeros((n_t,))
    
    w_err_H1 = np.zeros((n_t,))

    w_atP = np.zeros((n_t,))
    e_u_atP = np.zeros((n_t,))
    
    
    Ppoint = 3*L/4

    
    e_u_err_H1[0] = np.sqrt(assemble(inner(e_u_n - e_u_ex, e_u_n - e_u_ex) * dx
                  + inner(e_u_n.dx(0) - e_u_ex.dx(0), e_u_n.dx(0) - e_u_ex.dx(0)) * dx))
    
    e_eps_err_L2[0] = np.sqrt(assemble(inner(e_eps_n - e_eps_ex, e_eps_n - e_eps_ex) * dx))
    
    e_w_err_H1[0] = np.sqrt(assemble(inner(e_w_n - e_w_ex, e_w_n - e_w_ex) * dx
                  + inner(e_w_n.dx(0) - e_w_ex.dx(0), e_w_n.dx(0) - e_w_ex.dx(0)) * dx))
    
    e_kap_err_H1[0] = np.sqrt(assemble(inner(e_kap_n - e_kap_ex, e_kap_n - e_kap_ex) * dx
                  + inner(e_kap_n.dx(0) - e_kap_ex.dx(0), e_kap_n.dx(0) - e_kap_ex.dx(0))*dx))
    
    w_err_H1[0] = np.sqrt(assemble(inner(e_disp_n-w_ex, e_disp_n-w_ex) * dx
                + inner(e_disp_n.dx(0) - w_ex.dx(0), e_disp_n.dx(0) - w_ex.dx(0)) * dx))
    
    param = {"ksp_type": "preonly", "pc_type": "lu"}
        
    for i in range(1, n_t):


        e_u_n, e_eps_n, e_w_n, e_kap_n, e_disp_n = e_n.split()
        e_u_n1, e_eps_n1, e_w_n1, e_kap_n1, e_disp_n1 = split(e_n1)
       
        left_hs = m_operator(v_u, v_eps, v_w, v_kap, v_disp, \
              e_u_n1, e_eps_n1, e_w_n1, e_kap_n1, e_disp_n1) \
            - dt*theta*j_operator(v_u, v_eps, v_w, v_kap, v_disp, \
                       e_u_n1, e_eps_n1, e_w_n1, e_kap_n1, e_disp_n1)


        right_hs = m_operator(v_u, v_eps, v_w, v_kap, v_disp, \
              e_u_n, e_eps_n, e_w_n, e_kap_n, e_disp_n) \
              + dt * (1 - theta) * j_operator(v_u, v_eps, v_w, v_kap, v_disp, \
                                   e_u_n, e_eps_n, e_w_n, e_kap_n, e_disp_n) \
              + dt * ((1 - theta) * f_form + theta * f_form1)

        
        F = left_hs - right_hs

        e_n1.assign(e_n) #  For initialisation
        solve(F==0, e_n1, bcs=bcs, \
              solver_parameters=param)
        
        e_n.assign(e_n1)
        
        t += dt
        t_.assign(t)
        t_1.assign(t+dt)
  
        e_u_n, e_eps_n, e_w_n, e_kap_n, e_disp_n = e_n.split()
        

        e_u_err_H1[i] = np.sqrt(assemble(inner(e_u_n - e_u_ex, e_u_n - e_u_ex) * dx
                  + inner(e_u_n.dx(0) - e_u_ex.dx(0), e_u_n.dx(0) - e_u_ex.dx(0)) * dx))
    
        e_eps_err_L2[i] = np.sqrt(assemble(inner(e_eps_n - e_eps_ex, e_eps_n - e_eps_ex) * dx))
        
        e_w_err_H1[i] = np.sqrt(assemble(inner(e_w_n - e_w_ex, e_w_n - e_w_ex) * dx
                      + inner(e_w_n.dx(0) - e_w_ex.dx(0), e_w_n.dx(0) - e_w_ex.dx(0)) * dx))
        
        e_kap_err_H1[i] = np.sqrt(assemble(inner(e_kap_n - e_kap_ex, e_kap_n - e_kap_ex) * dx
                      + inner(e_kap_n.dx(0) - e_kap_ex.dx(0), e_kap_n.dx(0) - e_kap_ex.dx(0))*dx))
        
        w_err_H1[i] = np.sqrt(assemble(inner(e_disp_n-w_ex, e_disp_n-w_ex) * dx
                    + inner(e_disp_n.dx(0) - w_ex.dx(0), e_disp_n.dx(0) - w_ex.dx(0)) * dx))
    

        w_atP[i] = e_disp_n.at(Ppoint)
    

    # plt.figure()
    # plt.plot(t_vec, w_atP, 'r-', label=r'approx $w$')
    # plt.plot(t_vec, np.sin(pi*Ppoint/L)*np.sin(omega_w*t_vec), 'b-', label=r'exact $w$')
    # plt.xlabel(r'Time [s]')
    # plt.title(r'Displacement at: ' + str(Ppoint))
    # plt.legend()
     
    # x_num, e_eps_num = calculate_one_dim_points(e_eps_n, 1) 

    # V_plot = FunctionSpace(mesh, "CG", 5)
    # x_an, e_eps_an = calculate_one_dim_points(interpolate(e_eps_ex, V_plot), 10) 
         
    # fig, ax = plt.subplots()
    # ax.scatter(x_num, e_eps_num, c='r', label='Numerical')
    # ax.plot(x_an, e_eps_an, 'b', label="Analytical")
    # ax.legend()

    e_u_err_max = max(e_u_err_H1)
    e_eps_err_max = max(e_eps_err_L2)
    
    e_w_err_max = max(e_w_err_H1)
    e_kap_err_max = max(e_w_err_H1)
    
    w_err_max = max(w_err_H1)
    
    return e_u_err_max, e_eps_err_max, e_w_err_max, e_kap_err_max, w_err_max
    

# e_u_err_max, e_eps_err_max, e_w_err_max, e_kap_err_max, w_err_max = compute_err(10, 1)

n_h = 5
n_vec = np.array([2**(i+3) for i in range(n_h)])
h_vec = 1./n_vec


e_u_err_deg1 = np.zeros((n_h,))
e_u_err_deg2 = np.zeros((n_h,))
e_u_err_deg3 = np.zeros((n_h,))

e_eps_err_deg1 = np.zeros((n_h,))
e_eps_err_deg2 = np.zeros((n_h,))
e_eps_err_deg3 = np.zeros((n_h,))

e_w_err_deg1 = np.zeros((n_h,))
e_w_err_deg2 = np.zeros((n_h,))
e_w_err_deg3 = np.zeros((n_h,))

e_kap_err_deg1 = np.zeros((n_h,))
e_kap_err_deg2 = np.zeros((n_h,))
e_kap_err_deg3 = np.zeros((n_h,))

e_disp_err_deg1 = np.zeros((n_h,))
e_disp_err_deg2 = np.zeros((n_h,))
e_disp_err_deg3 = np.zeros((n_h,))


for i in range(n_h):
    e_u_err_deg1[i], e_eps_err_deg1[i], e_w_err_deg1[i], e_kap_err_deg1[i], \
        e_disp_err_deg1[i] = compute_err(n_vec[i], 1)
    e_u_err_deg2[i], e_eps_err_deg2[i], e_w_err_deg2[i], e_kap_err_deg2[i], \
        e_disp_err_deg2[i] = compute_err(n_vec[i], 2)
    # e_u_err_deg3[i], e_eps_err_deg3[i], e_w_err_deg3[i], e_kap_err_deg3[i], \
    #     e_disp_err_deg3[i] = compute_err(n_vec[i], 3)

        
       
path_res = "./errors_data/"
if save_res:
    np.save(path_res + "h_vec", h_vec)
   
    np.save(path_res + "e_u_err_deg1", e_u_err_deg1)
    np.save(path_res + "e_u_err_deg2", e_u_err_deg2)
    np.save(path_res + "e_u_err_deg3", e_u_err_deg3)

    np.save(path_res + "e_eps_err_deg1", e_eps_err_deg1)
    np.save(path_res + "e_eps_err_deg2", e_eps_err_deg2)
    np.save(path_res + "e_eps_err_deg3", e_eps_err_deg3)
    
    np.save(path_res + "e_w_err_deg1", e_w_err_deg1)
    np.save(path_res + "e_w_err_deg2", e_w_err_deg2)
    np.save(path_res + "e_w_err_deg3", e_w_err_deg3)
    
    np.save(path_res + "e_kap_err_deg1", e_kap_err_deg1)
    np.save(path_res + "e_kap_err_deg2", e_kap_err_deg2)
    np.save(path_res + "e_kap_err_deg3", e_kap_err_deg3)
    
    np.save(path_res + "e_disp_err_deg1", e_disp_err_deg1)
    np.save(path_res + "e_disp_err_deg2", e_disp_err_deg2)
    np.save(path_res + "e_disp_err_deg3", e_disp_err_deg3)



