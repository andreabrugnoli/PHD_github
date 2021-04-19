# Implementation of mixed finite elements for von Karman beams
from firedrake import *
from firedrake.plot import calculate_one_dim_points

import numpy as np
np.set_printoptions(threshold=np.inf)
from math import pi, floor

# Matplotlib settings
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

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

plt.rcParams["text.usetex"] =True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'figure.autolayout': True})



save_res = True
# Physical coefficients

rho = 2700
E = 70 * 10**3

# Geometrical coefficients 

L = 1
height = 0.1
wid = 0.1

A = height*wid
I = wid*height**3/12

n_elem= int(input("Enter the number of elements: "))
deg = int(input("Enter the degree of the approximation: "))


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





mesh = IntervalMesh(n_elem, L)

deg_eps = 2*(deg-1)
    
V_u = FunctionSpace(mesh, "CG", deg_eps+1)
V_eps = FunctionSpace(mesh, "DG", deg_eps)
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

u_st = x[0]**3*(1-(x[0]/L)**3)

u_ex = u_st*sin(omega_u*t_)    
e_u_ex = omega_u*u_st*cos(omega_u*t_)
dtt_u_ex = - omega_u**2 * u_st*sin(omega_u*t_)

w_ex = sin(pi*x[0]/L)*sin(omega_w*t_)
e_w_ex = omega_w*sin(pi*x[0]/L) * cos(omega_w*t_)
dtt_w_ex = - omega_w**2*sin(pi*x[0]/L)*sin(omega_w*t_)

e_eps_ex = E*A*(u_ex.dx(0) + 0.5*(w_ex.dx(0))**2)
e_kap_ex = E*I*w_ex.dx(0).dx(0)

f_u = rho*A*dtt_u_ex - e_eps_ex.dx(0)
f_w = rho*A*dtt_w_ex + e_kap_ex.dx(0).dx(0) - (e_eps_ex*w_ex.dx(0)).dx(0)

f_form = v_u*f_u*dx + v_w*f_w*dx

u_ex1 = u_st*sin(omega_u*t_1)    
e_u_ex1 = omega_u*u_st*cos(omega_u*t_1)
dtt_u_ex1 = - omega_u**2*u_st*sin(omega_u*t_1)

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
t_vec = np.arange(start=0, stop=n_t*dt, step=dt)

x_e_u, e_u0_vec = calculate_one_dim_points(e_u_n, 5) 
x_e_eps, e_eps0_vec = calculate_one_dim_points(e_eps_n, 5) 
x_e_w, e_w0_vec = calculate_one_dim_points(e_w_n, 5) 
x_e_kap, e_kap0_vec = calculate_one_dim_points(e_kap_n, 5) 
x_w, w0_vec = calculate_one_dim_points(e_disp_n, 5) 

e_u_sol = np.zeros((n_t, len(e_u0_vec)))
e_eps_sol = np.zeros((n_t, len(e_eps0_vec)))
e_w_sol = np.zeros((n_t, len(e_w0_vec)))
e_kap_sol = np.zeros((n_t, len(e_kap0_vec)))
w_sol = np.zeros((n_t, len(w0_vec)))

e_u_sol[0] = e_u0_vec
e_eps_sol[0] = e_eps0_vec
e_w_sol[0] = e_w0_vec
e_kap_sol[0] = e_kap0_vec
w_sol[0] = w0_vec

w_atP = np.zeros((n_t,))
e_u_atP = np.zeros((n_t,))


Ppoint = 3*L/4

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
    
    e_u_i_vec = calculate_one_dim_points(e_u_n, 5)[1]
    e_eps_i_vec = calculate_one_dim_points(e_eps_n, 5)[1]
    e_w_i_vec = calculate_one_dim_points(e_w_n, 5)[1]
    e_kap_i_vec = calculate_one_dim_points(e_kap_n, 5)[1]
    w_i_vec = calculate_one_dim_points(e_disp_n, 5)[1]

    e_u_sol[i] = e_u_i_vec
    e_eps_sol[i] = e_eps_i_vec
    e_w_sol[i] = e_w_i_vec
    e_kap_sol[i] = e_kap_i_vec
    w_sol[i] = w_i_vec
    


xx_plot_eu, tt_plot = np.meshgrid(x_e_u, t_vec)
xx_plot_eeps, tt_plot = np.meshgrid(x_e_eps, t_vec)
xx_plot_ew, tt_plot = np.meshgrid(x_e_w, t_vec)
xx_plot_ekap, tt_plot = np.meshgrid(x_e_kap, t_vec)
xx_plot_w, tt_plot = np.meshgrid(x_w, t_vec)


path_fig = "/home/andrea/Plots/Python/VonKarman/"
save_fig = True

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('$x \; \mathrm{[m]}$')
ax.set_ylabel('$t \; \mathrm{[s]}$')
ax.set_zlabel('$e_u\; \mathrm{[m/s]}$')

ax.set_title(r'Axial velocity', loc='center')


surf_eu = ax.plot_surface(xx_plot_eu, tt_plot, e_u_sol, cmap=cm.jet, linewidth=0, antialiased=False)

if save_fig:
    plt.savefig(path_fig + "plot_e_u.eps", format="eps")

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('$x \; \mathrm{[m]}$')
ax.set_ylabel('$t \; \mathrm{[s]}$')
ax.set_zlabel(r'$e_{\varepsilon} \; \mathrm{[N]}$')

ax.set_title(r'Axial stress', loc='center')


surf_eeps = ax.plot_surface(xx_plot_eeps, tt_plot, e_eps_sol, cmap=cm.jet, linewidth=0, antialiased=False)  
if save_fig:
    plt.savefig(path_fig + "plot_e_eps.eps", format="eps")


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('$x \; \mathrm{[m]}$')
ax.set_ylabel('$t \; \mathrm{[s]}$')
ax.set_zlabel('$e_w\; \mathrm{[m/s]}$')

ax.set_title(r'Vertical velocity', loc='center')


surf_ew = ax.plot_surface(xx_plot_ew, tt_plot, e_w_sol, cmap=cm.jet, linewidth=0, antialiased=False)   
if save_fig:
    plt.savefig(path_fig + "plot_e_w.eps", format="eps")


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('$x \; \mathrm{[m]}$')
ax.set_ylabel('$t \; \mathrm{[s]}$')
ax.set_zlabel('$e_\kappa\; \mathrm{[Nm]}$')

ax.set_title(r'Bending stress', loc='center')


surf_ekap = ax.plot_surface(xx_plot_ekap, tt_plot, e_kap_sol, cmap=cm.jet, linewidth=0, antialiased=False)   
if save_fig:
    plt.savefig(path_fig + "plot_e_kap.eps", format="eps")


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('$x \; \mathrm{[m]}$')
ax.set_ylabel('$t \; \mathrm{[s]}$')
ax.set_zlabel('$w\; \mathrm{[m]}$')

ax.set_title(r'Vertical displacement', loc='center')


surf_w= ax.plot_surface(xx_plot_w, tt_plot, w_sol, cmap=cm.jet, linewidth=0, antialiased=False)  
ax.view_init(azim=-30)
if save_fig:
    plt.savefig(path_fig + "plot_w.eps", format="eps")
    