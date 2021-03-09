# Implementation of mixed finite elements for von Karman beams
from firedrake import *
from firedrake.plot import calculate_one_dim_points

import numpy as np
np.set_printoptions(threshold=np.inf)
from math import pi, floor


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

# Physical coefficients

rho = 1 
E = 1

# Geometrical coefficients 

L = 1
A = 1
I = 1

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

    t_ = Constant(t)
    t_1 = Constant(t)
    x = mesh.coordinates
    
    u_ex = x[0]*(1-x[0]/L)*sin(t_)    
    e_u_ex = x[0]*(1-x[0]/L)*cos(t_)
    dtt_u_ex = - x[0]*(1-x[0]/L)*sin(t_)

    w_ex = sin(pi*x[0]/L)*sin(t_)
    e_w_ex = sin(pi*x[0]/L) * cos(t_)
    dtt_w_ex = - sin(pi*x[0]/L)*sin(t_)

    e_eps_ex = E*A*(u_ex.dx(0) + 0.5*(w_ex.dx(0))**2)
    e_kap_ex = E*I*w_ex.dx(0).dx(0)

    f_u = rho*A*dtt_u_ex - e_eps_ex.dx(0)
    f_w = rho*A*dtt_w_ex + e_kap_ex.dx(0).dx(0) - (e_eps_ex*w_ex.dx(0)).dx(0)
    
    f_form = v_u*f_u*dx + v_w*f_w*dx
    
    u_ex1 = x[0]*(1-x[0]/L)*sin(t_1)    
    e_u_ex1 = x[0]*(1-x[0]/L)*cos(t_1)
    dtt_u_ex1 = - x[0]*(1-x[0]/L)*sin(t_1)

    w_ex1 = sin(pi*x[0]/L)*sin(t_1)
    e_w_ex1 = sin(pi*x[0]/L) * cos(t_1)
    dtt_w_ex1 = - sin(pi*x[0]/L)*sin(t_1)

    e_eps_ex1 = E*A*(u_ex1.dx(0) + 0.5*(w_ex1.dx(0))**2)
    e_kap_ex1 = E*I*w_ex1.dx(0).dx(0)

    f_u1 = rho*A*dtt_u_ex1 - e_eps_ex1.dx(0)
    f_w1 = rho*A*dtt_w_ex1 + e_kap_ex1.dx(0).dx(0) - (e_eps_ex1*w_ex1.dx(0)).dx(0)
   
    f_form1 = v_u*f_u1*dx + v_w*f_w1*dx

    dt = 0.1/n_elem
    theta = 0.5

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
    
    
    for i in range(1, n_t):

        t_.assign(t)
        t_1.assign(t+dt)

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
        solve(F==0, e_n1, bcs=bcs)
        
        e_n.assign(e_n1)
        
        t += dt
        t_.assign(t)
        
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
    # plt.plot(t_vec, np.sin(pi*Ppoint/L)*np.sin(t_vec), 'b-', label=r'exact $w$')
    # # plt.plot(t_vec, v_atP, 'r-', label=r'approx $v$')
    # # plt.plot(t_vec, beta * np.sin(pi*Ppoint[0]/Lx)*np.sin(pi*Ppoint[1]/Ly) * np.cos(beta * t_vec), 'b-', label=r'exact $v$')
    # plt.xlabel(r'Time [s]')
    # plt.title(r'Displacement at: ' + str(Ppoint))
    # plt.legend()
    # # plt.show()
     
    x_num, e_eps_num = calculate_one_dim_points(e_eps_n, 1) 

    V_plot = FunctionSpace(mesh, "CG", 5)
    x_an, e_eps_an = calculate_one_dim_points(interpolate(e_eps_ex, V_plot), 10) 
         
    plt.figure()
    plt.scatter(x_num, e_eps_num, 'r', label="Numerical")
    plt.plot(x_an, e_eps_an, 'b', label="Analytical")
    
    return 1
    
compute_err(100, 1)




