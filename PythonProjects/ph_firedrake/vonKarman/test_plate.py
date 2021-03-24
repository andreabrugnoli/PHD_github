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
# Geometrical coefficients 

nu = Constant(0.3)
h = Constant(0.1)

L= 1

# Physical coefficients

rho = Constant(2700)
E = Constant(70 * 10**3)

D_bend = Constant(E * h ** 3 / (1 - nu ** 2) / 12)
C_bend = Constant(12 / (E * h ** 3))

D_men = Constant(E * h / (1 - nu ** 2))
C_men = Constant(1 / (E * h))


# Operators and functions
def gradSym(u):
    return 0.5 * (nabla_grad(u) + nabla_grad(u).T)
    # return sym(nabla_grad(u))

def bending_stiff(curv):
    m_stress = D_bend * ((1 - nu) * curv + nu * Identity(2) * tr(curv))
    return m_stress

def traction_stiff(eps_0):
    n_stress = D_men * ((1 - nu) * eps_0 + nu * Identity(2) * tr(eps_0))
    return n_stress

def bending_comp(m_stress):
    curv = C_bend * ((1+nu)*m_stress - nu * Identity(2) * tr(m_stress))
    return curv

def traction_comp(n_stress):
    eps_0 = C_men * ((1+nu)*n_stress- nu * Identity(2) * tr(n_stress))
    return eps_0


    
def m_operator(v_u, v_eps, v_w, v_kap, v_disp, \
                   e_u, e_eps, e_w, e_kap, e_disp):
        
        al_u = rho * h * e_u
        al_eps = traction_comp(e_eps)
        
        al_w = rho * h * e_w
        al_kap = bending_comp(e_kap)
        
        m_form = inner(v_u, al_u) * dx \
               + inner(v_eps, al_eps) * dx \
               + inner(v_w, al_w) * dx \
               + inner(v_kap, al_kap) * dx \
               + inner(v_disp, e_disp) * dx
               
        return m_form
    
    
def j_operator(v_u, v_eps, v_w, v_kap, v_disp, \
               e_u, e_eps, e_w, e_kap, e_disp, mesh):
    
    n_ver = FacetNormal(mesh)
    s_ver = as_vector([-n_ver[1], n_ver[0]])
    
    j_axial = inner(v_eps, gradSym(e_u)) * dx \
            - inner(gradSym(v_u), e_eps) * dx
            
    j_bend = - inner(grad(grad(v_w)), e_kap) * dx \
        + jump(grad(v_w), n_ver) * dot(dot(e_kap('+'), n_ver('+')), n_ver('+')) * dS \
        + dot(grad(v_w), n_ver) * dot(dot(e_kap, n_ver), n_ver) * ds \
        + inner(v_kap, grad(grad(e_w))) * dx \
        - dot(dot(v_kap('+'), n_ver('+')), n_ver('+')) * jump(grad(e_w), n_ver) * dS \
        - dot(dot(v_kap, n_ver), n_ver) * dot(grad(e_w), n_ver) * ds
    
    j_coup = inner(v_eps, sym(outer(grad(e_disp), grad(e_w)))) * dx \
           - inner(sym(outer(grad(v_w), grad(e_disp))),  e_eps) * dx
    
    m_w = inner(v_disp, e_w) * dx
    
    j_form = j_axial + j_bend + j_coup + m_w
    
    return j_form 



def compute_err(n_elem, deg):

    mesh = RectangleMesh(n_elem, n_elem, L, L, quadrilateral=False)
    
    deg_eps = 2*(deg-1)

    # V_u = VectorFunctionSpace(mesh, "CG", deg_eps+1)
    # V_epsD = VectorFunctionSpace(mesh, "DG", deg_eps)
    # V_eps12 = FunctionSpace(mesh, "DG", deg_eps)

    V_u = VectorFunctionSpace(mesh, "CG", deg)
    V_epsD = VectorFunctionSpace(mesh, "DG", deg-1)
    V_eps12 = FunctionSpace(mesh, "DG", deg-1)
    
    V_w = FunctionSpace(mesh, "CG", deg)
    V_kap = FunctionSpace(mesh, "HHJ", deg-1)
    V_disp = FunctionSpace(mesh, "CG", deg)
    
    
    V = V_u * V_epsD * V_eps12 * V_w * V_kap * V_disp
    
    print("Number of dofs: " +  str(V.dim()))
    
    v_u, v_epsD, v_eps12, v_w, v_kap, v_disp = TestFunctions(V)
    
    
    v_eps = as_tensor([[v_epsD[0], v_eps12],
                       [v_eps12, v_epsD[1]]
                       ])
    
    dx = Measure('dx')
    ds = Measure('ds')
    dS = Measure("dS")    
    
    
    bcs = []
    bc_u = DirichletBC(V.sub(0), Constant((0.0, 0.0)), "on_boundary")
    bc_w = DirichletBC(V.sub(3), Constant(0.0), "on_boundary")
    bc_kap = DirichletBC(V.sub(4), Constant(((0.0, 0.0), (0.0, 0.0))), "on_boundary")
    
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
    
    u_st = as_vector([x[0]**3*(1-(x[0]/L)**3)*sin(pi*x[1]/L)**2,
                      sin(pi*x[0]/L)**2*x[1]**3*(1-(x[1]/L)**3)])
    
    u_ex = u_st*sin(omega_u*t_)    
    e_u_ex = omega_u*u_st*cos(omega_u*t_)
    dtt_u_ex = - omega_u**2 * u_st*sin(omega_u*t_)
    
    w_st = sin(pi*x[0]/L)*sin(pi*x[1]/L)
    
    w_ex = w_st * sin(omega_w*t_)
    e_w_ex = omega_w * w_st * cos(omega_w*t_)
    dtt_w_ex = - omega_w**2*w_st*sin(omega_w*t_)
    
    e_eps_ex = traction_stiff(gradSym(u_ex) + 0.5 * outer(grad(w_ex), grad(w_ex)))
    e_kap_ex = bending_stiff(grad(grad(w_ex)))
    
    f_u = rho*h*dtt_u_ex - div(e_eps_ex)
    f_w = rho*h*dtt_w_ex + div(div(e_kap_ex)) - div(dot(e_eps_ex, grad(w_ex)))
    
    f_form = inner(v_u, f_u)*dx + inner(v_w, f_w)*dx
    
    u_ex1 = u_st*sin(omega_u*t_1)    
    e_u_ex1 = omega_u*u_st*cos(omega_u*t_1)
    dtt_u_ex1 = - omega_u**2*u_st*sin(omega_u*t_1)
    
    w_ex1 = w_st*sin(omega_w*t_1)
    e_w_ex1 = omega_w* w_st * cos(omega_w*t_1)
    dtt_w_ex1 = - omega_w**2* w_st *sin(omega_w*t_1)
    
    e_eps_ex1 = traction_stiff(gradSym(u_ex1) + 0.5 * outer(grad(w_ex1), grad(w_ex1)))
    e_kap_ex1 = bending_stiff(grad(grad(w_ex1)))
    
    f_u1 = rho*h*dtt_u_ex1 - div(e_eps_ex1)
    f_w1 = rho*h*dtt_w_ex1 + div(div(e_kap_ex1)) - div(dot(e_eps_ex1, grad(w_ex1)))
       
    f_form1 = inner(v_u, f_u1)*dx + inner(v_w, f_w1)*dx
    
    
    e_n = Function(V,  name="e old")
    e_n1 = Function(V,  name="e new")
    
    e_n.sub(0).assign(project(e_u_ex, V_u))
    e_n.sub(3).assign(project(e_w_ex, V_w))
    
    e_u_n, e_epsD_n, e_eps12_n, e_w_n, e_kap_n, e_disp_n = e_n.split()
    
    e_eps_n = as_tensor([[e_epsD_n[0], e_eps12_n],
                         [e_eps12_n, e_epsD_n[1]]
                         ])
    
    n_t = int(floor(t_fin/dt) + 1)
    t_vec = np.arange(start=0, stop=n_t*dt, step=dt)
    
    e_u_err_H1 = np.zeros((n_t,))
    e_eps_err_L2 = np.zeros((n_t,))
    
    e_w_err_H1 = np.zeros((n_t,))
    e_kap_err_L2 = np.zeros((n_t,))
    
    w_err_H1 = np.zeros((n_t,))

    w_atP = np.zeros((n_t,))
    e_u_atP = np.zeros((n_t,))
    
    
    Ppoint = 3*L/4

    
    e_u_err_H1[0] = np.sqrt(assemble(inner(e_u_n - e_u_ex, e_u_n - e_u_ex) * dx
                  + inner(gradSym(e_u_n) - gradSym(e_u_ex), gradSym(e_u_n) - gradSym(e_u_ex)) * dx))
    
    e_eps_err_L2[0] = np.sqrt(assemble(inner(e_eps_n - e_eps_ex, e_eps_n - e_eps_ex) * dx))
    
    e_w_err_H1[0] = np.sqrt(assemble(inner(e_w_n - e_w_ex, e_w_n - e_w_ex) * dx
                  + inner(grad(e_w_n) - grad(e_w_ex), grad(e_w_n) - grad(e_w_ex)) * dx))
    
    e_kap_err_L2[0] = np.sqrt(assemble(inner(e_kap_n - e_kap_ex, e_kap_n - e_kap_ex) * dx))
    
    w_err_H1[0] = np.sqrt(assemble(inner(e_disp_n-w_ex, e_disp_n-w_ex) * dx
                + inner(grad(e_disp_n) - grad(w_ex), grad(e_disp_n) - grad(w_ex)) * dx))
    
    param = {'snes_type': 'newtonls', 'ksp_type': 'preonly', 'pc_type': 'lu'} 
             # 'snes_rtol': '1e+1', 'snes_atol': '1e-10','snes_stol': '1e+1', 
             # 'snes_max_it': '50', 'ksp_rtol': '1e+1', 'ksp_atol': '1e-10', 
             # 'ksp_rtol': '1e+1', 'ksp_atol': '1e-10', 'ksp_divtol': '1e15'}
        
    for i in range(1, n_t):


        e_u_n, e_epsD_n, e_eps12_n, e_w_n, e_kap_n, e_disp_n = e_n.split()

        e_eps_n = as_tensor([[e_epsD_n[0], e_eps12_n],
                             [e_eps12_n, e_epsD_n[1]]
                             ])
                
        e_u_n1, e_epsD_n1, e_eps12_n1, e_w_n1, e_kap_n1, e_disp_n1 = split(e_n1)
       
        e_eps_n1 = as_tensor([[e_epsD_n1[0], e_eps12_n1],
                              [e_eps12_n1, e_epsD_n1[1]]
                              ])
       
        left_hs = m_operator(v_u, v_eps, v_w, v_kap, v_disp, \
              e_u_n1, e_eps_n1, e_w_n1, e_kap_n1, e_disp_n1) \
            - dt*theta*j_operator(v_u, v_eps, v_w, v_kap, v_disp, \
                       e_u_n1, e_eps_n1, e_w_n1, e_kap_n1, e_disp_n1, mesh)


        right_hs = m_operator(v_u, v_eps, v_w, v_kap, v_disp, \
              e_u_n, e_eps_n, e_w_n, e_kap_n, e_disp_n) \
              + dt * (1 - theta) * j_operator(v_u, v_eps, v_w, v_kap, v_disp, \
                                   e_u_n, e_eps_n, e_w_n, e_kap_n, e_disp_n, mesh) \
              + dt * ((1 - theta) * f_form + theta * f_form1)

        
        F = left_hs - right_hs

        e_n1.assign(e_n) #  For initialisation
        solve(F==0, e_n1, bcs=bcs, \
              solver_parameters=param)
        
        e_n.assign(e_n1)
        
        t += dt
        t_.assign(t)
        t_1.assign(t+dt)
  
        e_u_n, e_epsD_n, e_eps12_n, e_w_n, e_kap_n, e_disp_n = e_n.split()

        e_eps_n = as_tensor([[e_epsD_n[0], e_eps12_n],
                             [e_eps12_n, e_epsD_n[1]]
                             ])
        

        e_u_err_H1[i] = np.sqrt(assemble(inner(e_u_n - e_u_ex, e_u_n - e_u_ex) * dx
                  + inner(gradSym(e_u_n) - gradSym(e_u_ex), gradSym(e_u_n) - gradSym(e_u_ex)) * dx))
    
        e_eps_err_L2[i] = np.sqrt(assemble(inner(e_eps_n - e_eps_ex, e_eps_n - e_eps_ex) * dx))
        
        e_w_err_H1[i] = np.sqrt(assemble(inner(e_w_n - e_w_ex, e_w_n - e_w_ex) * dx
                      + inner(grad(e_w_n) - grad(e_w_ex), grad(e_w_n) - grad(e_w_ex)) * dx))
        
        e_kap_err_L2[i] = np.sqrt(assemble(inner(e_kap_n - e_kap_ex, e_kap_n - e_kap_ex) * dx))
        
        w_err_H1[i] = np.sqrt(assemble(inner(e_disp_n-w_ex, e_disp_n-w_ex) * dx
                + inner(grad(e_disp_n) - grad(w_ex), grad(e_disp_n) - grad(w_ex)) * dx))
    

        # w_atP[i] = e_disp_n.at(Ppoint)
    

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
    e_kap_err_max = max(e_kap_err_L2)
    
    w_err_max = max(w_err_H1)
    
    return e_u_err_max, e_eps_err_max, e_w_err_max, e_kap_err_max, w_err_max
    


n_h = 3
n_vec = np.array([2**(i+2) for i in range(n_h)])
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

        
       
path_res = "./errors_data_plate3/"
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



