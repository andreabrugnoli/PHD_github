#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 15:52:50 2018

@author: vasseur
"""    
#
# Import here the standard libraries 
#
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode

#
# Import here the classes related to SCRIMP
#
from phdae_system_manager import System_DAE
from phdae_subsystem_manager import Subsystem_DAE
from linear_oscillator import Linear_Oscillator


#
# Part: Construction of the coupled system
#


#
# Define the variables of the subsystem related to the linear oscillator 
#

# We consider a d-degrees of freedom linear oscillator 

d = 2

# ns: dimension of the state space of the linear oscillator
# Here ns is equal to 2d since we consider a d-dimensional linear oscillator

ns = 2*d 

# nr: dimension of the resistive space

nr = 0

# nc: dimension of the input/control space

nc = 1

# ni: dimension of the interconnection space 

ni = 4 

# nm: dimension of the constraint space (unconstrained case so nm = 0)

nm = 0

#
#  Constants related to the two linear oscillators that can be changed
#

mass_1     = 1.0            #[kg] mass  
omega_1    = 0.5            #[rad.s^-1] angular velocity
mass_2     = 2.0            #[kg] mass  
omega_2    = 0.025          #[rad.s^-1] angular velocity
  
#
# Create the Lo_1 linear oscillator object from the Linear_Oscillator class
#
# See the __init__ method of the Linear_Oscillator class for creating Lo_1
#

# C should be symmetric
# G should be skew-symmetric

M_1 = np.zeros(shape=(d,d))
C_1 = np.zeros(shape=(d,d)) 
G_1 = np.zeros(shape=(d,d))
K_1 = np.zeros(shape=(d,d))
    
M_1 = mass_1 * np.eye(d)
K_1 = omega_1*omega_1* np.eye(d)

Lo_1 = Linear_Oscillator(M_1,C_1,G_1,K_1,nr,ni,nc)

#
# We define the structure, dissipation and global structure of the pHs
#

Lo_1.Define_structure_matrix()
Lo_1.Define_dissipation_matrix()
Lo_1.Define_Hamiltonian_quadratic_component()
Lo_1.Set_Subsystem()

#
# We check that the object has been created properly
#

print("The Lo_1 object has been created successfully: ", bool(Lo_1.status))

#
# Create the Lo_2 linear oscillator object following the steps above
#

# C should be symmetric
# G should be skew-symmetric


M_2 = np.zeros(shape=(d,d))
C_2 = np.zeros(shape=(d,d)) 
G_2 = np.zeros(shape=(d,d))
K_2 = np.zeros(shape=(d,d))
    
M_2 = mass_2 * np.eye(d)
K_2 = omega_2*omega_2 * np.eye(d)

Lo_2 = Linear_Oscillator(M_2,C_2,G_2,K_2,nr,ni,nc)

#
# We define the structure, dissipation and global structure of the pHs
#

Lo_2.Define_structure_matrix()
Lo_2.Define_dissipation_matrix()
Lo_2.Define_Hamiltonian_quadratic_component()
Lo_2.Set_Subsystem()

#
# We check that the object has been created properly
#

print("The Lo_2 object has been created successfully: ", bool(Lo_2.status))


#
# Create the two subsystems related to the two linear oscillators
# 


#
# Define B and D matrices here
#

# B (input matrix)    
B = np.random.rand(ns,nr+nc+ni)

# D (feedthrough matrix) 
D = np.random.rand(nr+nc+ni,nr+nc+ni)
D = 0.5*(D+np.transpose(D))   


#
# First subsystem related to Lo_1 
#

Sub_L1 = Subsystem_DAE(ns,nr,nc,ni)

Sub_L1.Set_J_matrix(Lo_1.J)
    
Sub_L1.Set_R_matrix(Lo_1.R)
    
Sub_L1.Set_B_matrix(B)
    
Sub_L1.Set_D_matrix(D)
 
Sub_L1.Set_Q_matrix(Lo_1.Q)    
 
Sub_L1.Set_S_matrix()

#
# Second subsystem related to Lo_2
#

Sub_L2 = Subsystem_DAE(ns,nr,nc,ni)

Sub_L2.Set_J_matrix(Lo_2.J)
    
Sub_L2.Set_R_matrix(Lo_2.R)
    
Sub_L2.Set_B_matrix(B)
    
Sub_L2.Set_D_matrix(D)
 
Sub_L2.Set_Q_matrix(Lo_2.Q)    
 
Sub_L2.Set_S_matrix()

#
# We check that the two pHDAEs have been built successfully
#

print("The Sub_L1 object has been created successfully: ", bool(Sub_L1.istatus_S))

print("The Sub_L2 object has been created successfully: ", bool(Sub_L2.istatus_S))

#
# We are now ready to interconnect the two pHDAEs.
#

#
# Create a Python list of subsystems defining the global system
#

Subsystem_List = []
Subsystem_List.append(Sub_L1)
Subsystem_List.append(Sub_L2)

#
# Define the coupling matrix related to the two subsystems
#

Coupling_Lo = np.eye(Sub_L1.ni)

#
# Specify that the global port-Hamiltonian system named pHs_Lo is
# made of the two pHs Sub_L1 and Sub_L2
#

pHs_Lo = System_DAE(Subsystem_List,Coupling_Lo,connection_type="Gyrator")

#
# Define the global PHS object
#

pHs_Lo.Define_PHS_DAE()

#
# Print the global matrix related to the global port-Hamiltonian DAE system
#

print("\n The global S matrix of the interconnected pHDAE object is: \n")

print(pHs_Lo.S.shape)
print(pHs_Lo.S)   

print("\n The global Q matrix of the interconnected pHDAE object is: \n")

print(pHs_Lo.Q.shape)
print(pHs_Lo.Q)   

#
# Check that everything is fine at this stage
#

print("\n The interconnected pHs object has been created successfully: ", bool(pHs_Lo.istatus_S))

#
# The answer should be True at this stage ! 
#


#
# Part: Simulation
#


#
# We impose the forcing function in the Input_function method of the System_DAE class.
# The following function is considered:
# w(t) = 0.1*sin(pi*t/4) for each component of the u_ext vector.
#


#
# We perform the time integration to deduce the state vector and output vector of the 
# dynamical system with standard methods from the scipy.integrate module.
# We set the initial conditions on the state vector and 
# specify the parameters related to the time integration.
#

#x0 = np.random.rand(pHs_Lo.ns)
x0 = np.ones(pHs_Lo.ns)
t0 = 0. 
t1 = 10.
dt = 0.05

#
# The dynamic equation of the interconnected pHs is implemented in the Define_dynamics_ode method of the 
# System_DAE class.
#
#
# We select here the integration method that we plan to use.
# We choose a non-symplectic method from the ODE library of SciPy.
#

time_integrator = ode(pHs_Lo.Define_dynamics_ode).set_integrator('vode', method='bdf')

# Specify the initial conditions
time_integrator.set_initial_value(x0, t0)

#
# Time integration is now performed
#

#
# The output vector is obtained through the Define_dynamics_output() method of the System_DAE class.
#
#


nsteps        = int((t1-t0)/dt)+1
time          = np.zeros(nsteps)
state         = np.zeros((pHs_Lo.ns,nsteps))
H             = np.zeros(nsteps)
out           = np.zeros((pHs_Lo.ncc,nsteps))
loop          = 0
time[loop]    = t0
state[:,loop] = x0
out[:,loop]   = pHs_Lo.Define_dynamics_output(t0,x0)

while time_integrator.successful() and time_integrator.t < t1:
    
        # Compute the state variable at the next discrete time 
        state_variable  = time_integrator.integrate(time_integrator.t+dt)
        
        # Compute the output variables at the same discrete time 
        # The time_integrator.t variable now corresponds to the same discrete time 
        # since it has been updated due to the previous call to the integrate method.
        output_variable = pHs_Lo.Define_dynamics_output(time_integrator.t,state_variable)
        
        # Store information for later postprocessing
        loop            = loop + 1 
        time[loop]      = time_integrator.t
        state[:,loop]   = state_variable
        out[:,loop]     = output_variable 
        H[loop]         = pHs_Lo.Get_Hamiltonian(state[:,loop])
#
# Final check 
#

print("The ODE integration has been performed successfully: ",time_integrator.successful())

#
#
# Plot the evolution of the Hamiltonian at the state vector versus time
#

plt.figure()
plt.plot(time[1:],H[1:],'-o')
plt.title('Time variation of the Hamiltonian of the coupled system')
plt.xlabel('Time [s]')
plt.ylabel('Hamiltonian at the state vector')
plt.grid(True)
plt.show()



#
# Done
#


