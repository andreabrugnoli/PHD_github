
from Mindlin_class import Mindlin
import numpy as np
from scipy.sparse import csc_matrix, block_diag
from math import *
import matplotlib.pyplot as plt
import sys, timeit


#%% Constructor
Mindlin_test = Mindlin()
Mindlin_test()
self = Mindlin_test
#%%


#%% Problem definition
### Geometric rectangular domain
x0, xL = 0, 1
y0, yL = 0, 1
Mindlin_test.Set_Rectangular_Domain(x0, xL, y0, yL)

### Physical paramters
#rho = '2600' #  'cos(x[0])+1'
#h = '0.1'
#E = 'pow(10,12)'
#nu = '0.3'
#k  = '5/6'
#
#
#Mindlin_test.Set_Physical_Parameters(rho, h, E, nu, k, init_by_value=False)
# 
E = 10**12
nu = 0.3
rho = 2600
h=0.1
k=5/6

Mindlin_test.Set_Physical_Parameters(rho, h, E, nu, k, init_by_value=True)

## Internal damping coefficient
#eps = '4 * x[0] * (xL - x[0]) * x[1] * (yL - x[1])'
eps = '10000'

Mindlin_test.Set_Damping(damp=['internal'], eps=eps)

### Final time 
tf = 0.01
Mindlin_test.Set_Initial_Final_Time(initial_time=0, final_time=tf)
#%%


#%% Space discretization
### Mesh
#Mindlin_test.Set_Gmsh_Mesh('rectangle.xml', rfn_num=1)
Mindlin_test.Generate_Mesh(5, structured_mesh=True)


### Finite elements spaces
Mindlin_test.Set_Finite_Elements_Spaces(r=1)

### Assembly    
Mindlin_test.Set_Mixed_Boundaries(Dir=['G1'], Nor=['G2', 'G3', 'G4'])
Mindlin_test.Assembly_Mixed_BC() 

#from scikits import umfpack
#facto = umfpack.UmfpackLU(self.M)
##%% Environement
#### Boundary control
#def Ub_tm0(t):
#    if t <= 2:
#        return np.sin( 2 * 2*pi/tf *t) * 100 
#    else: return 0
Ub_tm0 = lambda t:  np.array([(1-np.exp(-t/tf))*(t<=0.1*tf), 0, 0])
Mindlin_test.Set_Mixed_BC_Normal(Ub_tm0=Ub_tm0 ,\
                           Ub_sp0=('1.', '0','0'))

Mindlin_test.Set_Mixed_BC_Dirichlet(Ub_tm0=np.array([0,0,0]), Ub_sp0=('0', '0','0'))

# Gaussian initial datas
ampl, sX, sY, X0, Y0  = 1, Mindlin_test.xL/6, Mindlin_test.yL/6,\
                     Mindlin_test.xL/2, Mindlin_test.yL/2 
                     
#gau_W_0 = 'ampl * exp(- pow( (x[0]-X0)/sX, 2) - pow( (x[1]-Y0)/sY, 2) )'
#Th_0_1 = '0'
#Th_0_2 = '0'
#gau_Apw_0 = 'rho * ampl * exp(- pow( (x[0]-X0)/sX, 2) - pow( (x[1]-Y0)/sY, 2) )'
#gau_Aqw_0_1 = '-ampl * 2 * (x[0]-X0)/sX * exp(- pow( (x[0]-X0)/sX, 2) - pow( (x[1]-Y0)/sY, 2) )'
#gau_Aqw_0_2 = '-ampl * 2 * (x[0]-Y0)/sY * exp(- pow( (x[0]-X0)/sX, 2) - pow( (x[1]-Y0)/sY, 2) )'

Mindlin_test.Set_Initial_Data(W_0='0', Th_0_1='0', Th_0_2='0',\
                         Apw_0='0', Apth1_0='0', Apth2_0='0',\
                         Aqth11_0='0', Aqth12_0='0', Aqth22_0='0',\
                         Aqw1_0='0', Aqw2_0='0',\
                         ampl=ampl, sX=sX, sY=sY, X0=X0, Y0=Y0, rho=Mindlin_test.rho)

#Mindlin_test.Set_Initial_Data(Aq_0_1='0', Aq_0_2='0', Ap_0='0', W_0='0')
#%%

#%% Projections
### Project boundary controls
Mindlin_test.Project_Boundary_Control()

### Project initial datas
Mindlin_test.Project_Initial_Data()
#%%


#%% Time discretization setting
### Step
Mindlin_test.Set_Time_Setting(time_step=1e-6)

### Method
#method = 'DAE:Assimulo'
method = 'DAE:RK4Augmented'
#method = 'ODE:RK4'

#%%

#%% Time-stepping
### Start
A, Ham = Mindlin_test.Time_Integration(method)  
Mindlin_test.Plot_Hamiltonian(Mindlin_test.tspan, Ham, linewidth=3)





#%% Post-processing analysis

### Get explicitly energies

### Get deflection
w = Mindlin_test.Get_Deflection(A)

### Simulation time

### Animations
anime = True
step = 50
if anime :
    Mindlin_test.Set_Video_Writer()
    Mindlin_test.Moving_Trisurf(w, step=step, title='Deflection', figsize=(18,7.5), cmap=plt.cm.plasma)#, save=True)
#    Mindlin_test.Moving_Plot(Ham, Mindlin_test.tspan,  step=step, title='Hamiltonian', figsize=(8,7.5), save=True)
End = timeit.default_timer()
