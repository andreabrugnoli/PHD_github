
from Waves import Wave_2D
import numpy as np
from scipy.sparse import csc_matrix, block_diag
from math import *
import matplotlib.pyplot as plt
import sys, timeit


#%% Constructor
Wtest = Wave_2D()
Wtest()
self = Wtest
#%%


#%% Problem definition
### Geometric rectangular domain
x0, xL = 0, 2
y0, yL = 0, 1
Wtest.Set_Rectangular_Domain(x0, xL, y0, yL)

### Physical paramters
## Mass density
Rho = 'cos(x[0])+1' #5 * x[0] * (xL - x[0]) * x[1] * (yL - x[1]) + 1' # '1' # 

## Young's modulus
T11 = 'x[0]+2' 
T12 = 'x[1]' # '0' # 
T22 = 'x[0]+x[1]+1' # '1' # 

Wtest.Set_Physical_Parameters(Rho, T11, T12, T22)

### Damping
## Impedance coefficient
#Z = '''( abs(x[0]) <= DOLFIN_EPS ? 1 : 0 )
#        + ( abs(x[1]) <= DOLFIN_EPS ? 1 : 0 )
#        + ( abs(xL - x[0]) <= DOLFIN_EPS ? .1 : 0 )
#        + ( abs(yL - x[1]) <= DOLFIN_EPS ? 10 : 0 )'''
Z   = '1'
Y   = '''( abs(x[0]) <= DOLFIN_EPS ? x[1]*(1-x[1]) : 0 )
        + ( abs(x[1]) <= DOLFIN_EPS ? sin(x[0]*(xL-x[0])) : 0 )
        + ( abs(xL - x[0]) <= DOLFIN_EPS ? 5*x[1]*(1-x[1]) : 0 )
        + ( abs(yL - x[1]) <= DOLFIN_EPS ? 0 : 0 )'''

## Internal damping coefficient
eps = '4 * x[0] * (xL - x[0]) * x[1] * (yL - x[1])'

Wtest.Set_Damping(damp=['internal'], \
              Z=Z, Y=Y, eps=eps)

### Final time 
tf = 6
Wtest.Set_Initial_Final_Time(initial_time=0, final_time=tf)
#%%


#%% Space discretization
### Mesh
Wtest.Set_Gmsh_Mesh('rectangle.xml', rfn_num=1)

### Finite elements spaces
Wtest.Set_Finite_Elements_Spaces(family_q='RT', family_p='P', family_b='P', rq=1, rp=2, rb=2)

### Assembly    
#Wtest.Assembly_Mixed_BC() 
Wtest.Assembly(formulation='Grad')
#%%

#%% Environement
### Boundary control
def Ub_tm0(t):
    if t <= 2:
        return np.sin( 2 * 2*pi/tf *t) * 100 
    else: return 0
Ub_tm0 = lambda t:  np.sin( 2 * 2*pi/tf *t) * 100  
Wtest.Set_Boundary_Control(Ub_tm0=Ub_tm0 ,\
                           Ub_sp0='x[0] * x[1] * (1-x[1])')

# Gaussian initial datas
ampl, sX, sY, X0, Y0  = 10, Wtest.xL/6, Wtest.yL/6, Wtest.xL/2, Wtest.yL/2 
gau_Aq_0_1 = '-ampl * 2 * (x[0]-X0)/sX * exp(- pow( (x[0]-X0)/sX, 2) - pow( (x[1]-Y0)/sY, 2) )'
gau_Aq_0_2 = '-ampl * 2 * (x[0]-Y0)/sY * exp(- pow( (x[0]-X0)/sX, 2) - pow( (x[1]-Y0)/sY, 2) )'
gau_Ap_0 = 'rho * ampl * exp(- pow( (x[0]-X0)/sX, 2) - pow( (x[1]-Y0)/sY, 2) )'
gau_W_0 = 'ampl * exp(- pow( (x[0]-X0)/sX, 2) - pow( (x[1]-Y0)/sY, 2) )'
Wtest.Set_Initial_Data(Aq_0_1=gau_Aq_0_1, Aq_0_2=gau_Aq_0_2, Ap_0=gau_Ap_0, W_0=gau_W_0,\
                       ampl=ampl, sX=sX, sY=sY, X0=X0, Y0=Y0, rho=Wtest.rho)   

#Wtest.Set_Initial_Data(Aq_0_1='0', Aq_0_2='0', Ap_0='0', W_0='0')
#%%

#%% Projections
### Project boundary controls
Wtest.Project_Boundary_Control()

### Project initial datas
Wtest.Project_Initial_Data()
#%%


#%% Time discretization setting
### Step
Wtest.Set_Time_Setting(time_step=1e-3)

### Method
method = 'ODE:RK4'
#%%

#%% Time-stepping
### Start
A, Ham = Wtest.Time_Integration(method)  
Wtest.Plot_Hamiltonian(Wtest.tspan, Ham, linewidth=3)





#%% Post-processing analysis

### Get explicitly energies
Aq = A[:Wtest.Nq,:]
Ap = A[Wtest.Nq:,:]    

### Get deflection
w = Wtest.Get_Deflection(A)

### Simulation time

### Animations
anime = False
step = 50
if anime :
    Wtest.Set_Video_Writer()
    Wtest.Moving_Trisurf(w, step=step, title='Deflection', figsize=(18,7.5), cmap=plt.cm.plasma)#, save=True)
    Wtest.Moving_Trisurf(Ap, step=step, title='Linear momentum', cmap=plt.cm.plasma)#, figsize=(20,7.5) )#, save=True)
    Wtest.Moving_Quiver(Aq, step=step, title='Strain', figsize=(15,7.5), cmap=plt.cm.plasma)#, save=True)               
    Wtest.Moving_Plot(Ham, Wtest.tspan,  step=step, title='Hamiltonian', figsize=(8,7.5), save=True)
End = timeit.default_timer()
