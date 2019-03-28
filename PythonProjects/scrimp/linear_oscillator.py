# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:33:34 2017

@author: vasseur
"""
## @package linear_oscillator
#  Documentation for this module.
#
#  Class to define the linear oscillator as a subsystem of PHS type. 

## Documentation for a class.
#
#  The Linear_Oscillator class implements the specific features 
#  of this port-Hamiltonian system. 
#  

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from subsystem_manager import Subsystem


class Linear_Oscillator(Subsystem):
    """ Linear oscillator as a subsystem of port-Hamiltonian type """

    ## The constructor of the Linear_Oscillator class.
    #  @param self The object pointer.
    #  @param M Mass matrix of the linear oscillator [Numpy array]
    #  @param C Damping matrix of the linear oscillator [Numpy array]
    #  @param G Gyroscopic matrix of the linear oscillator [Numpy array]
    #  @param K Stiffness matrix of the linear oscillator [Numpy array]
    #  @param nr  Dimension of the resistive space with port variables (fr, er)
    #  @param ni  Dimension of the interconnection space with port variables (fi, ei)
    #  @param nc  Dimension of the input/output control space with port variables (fc, ec)
    def __init__(self,M,C,G,K,nr=0,ni=0,nc=0):
        """
        Constructor for the Linear_Oscillator subsystem.
        """
        
        assert   M.shape[0] == M.shape[1], "The matrix M must be square."       
        assert   C.shape[0] == C.shape[1], "The matrix C must be square."       
        assert   G.shape[0] == G.shape[1], "The matrix G must be square."       
        assert   K.shape[0] == K.shape[1], "The matrix K must be square."
        
        # Get the dimension of the block 
        
        self.n  = M.shape[0]
        
        assert   self.n == C.shape[0], "The numbers of rows of M and C do not match."             
        assert   self.n == G.shape[0], "The numbers of rows of M and G do not match."       
        assert   self.n == K.shape[0], "The numbers of rows of M and K do not match."      
               
        # Get the global dimension of the system 
               
        ns      = M.shape[0] + K.shape[0] 
        
        # Call the constructor to define a classical Hamiltonian system
        
        Subsystem.__init__(self,ns,nr,ni,nc)
        
        # Store the matrices in the object
        self.M = M
        self.C = C
        self.G = G
        self.K = K
        self.I = np.eye(self.n)
        
        # Cholesky factorization of M 
        self.L = np.linalg.cholesky(self.M)
        
    ## Method.
    #  @param self The object pointer. 
    def Define_dissipation_matrix(self):
        """
        Define the dissipation matrix of the system 
        """
        
        self.R[self.n:self.nt,self.n:self.nt] = self.C 
        
        self.istatus_R = 1
        
        return self.istatus_R

    ## Method.
    #  @param self The object pointer.     
    def Define_structure_matrix(self):
        """
        Define the structure matrix of the system
        """
        self.J[0:self.n,self.n:self.nt]       = self.I
        self.J[self.n:self.nt,0:self.n]       = - self.I
        self.J[self.n:self.nt,self.n:self.nt] = - self.G  
        
        self.istatus_J = 1
        
        return self.istatus_J

    ## Method.
    #  @param self The object pointer.        
    def Define_Hamiltonian_quadratic_component(self):
        """
        Define the quadratic component of the Hamiltonian of the system.
        """
        
        self.Q[0:self.n,0:self.n] = self.K  
        
        self.Q[self.n:self.ns,self.n:self.ns] = np.linalg.inv(self.M)
        
        self.istatus_Q = 1
        
        return self.istatus_Q
        

    ## Method.
    #  @param self The object pointer. 
    #  @state_vector Global current state vector.        
    def Apply_gradient_Hamiltonian(self,state_vector):
        """
        Apply the gradient of the Hamiltonian to a state vector. 
        This method overrides the same method implemented in the parent class. 
        """
        assert self.nt == state_vector.shape[0], "Wrong dimension in the state vector"
        
        vector = np.zeros(self.ns)
        
        vector[0:self.n] = np.matmul(self.K,state_vector[0:self.n])
        
        vector[self.n:self.nt] = np.linalg.solve(self.M,state_vector[self.n:self.ns])

        #vector[self.n:self.nt] = np.linalg.solve(self.L,state_vector[self.n:self.ns])
        
        #vector[self.n:self.nt] = np.linalg.solve(numpy.transpose(self.L),vector[self.n:self.ns])
         
        return vector
        
    ## Method.
    #  @param self The object pointer.  
    #  @param energy_vector Energy vector. 
    #  @param component_string String specifying which component must be computed.   
    def Get_coenergy_variable(self,energy_vector,component_string):
        """
        Get the coenergy variable corresponding to a given component of the global system
        """
        assert self.nt == energy_vector.shape[0], "Wrong dimension in the vector"
        
        assert component_string in ["First", "Second"], "Wrong component" 
        
        vector = np.zeros(self.n)
        
        if component_string == "First":
            vector[0:self.n] = np.matmul(self.K,energy_vector[0:self.n])
        
        elif component_string == "Second":
            #vector[0:self.n] = np.linalg.solve(self.M,energy_vector[0:self.n])
            vector[0:self.n] = np.linalg.solve(self.L,energy_vector[self.n:self.nt])
            vector[0:self.n] = np.linalg.solve(np.transpose(self.L),vector[0:self.n])
                         
        return vector       
     

    ## Method.
    #  @param self The object pointer. 
    #  @param dissipation_vector Vector of dissipation variables.    
    def Apply_dissipation_function(self,dissipation_vector):
        """
        ?
        """
        pass
    
    
    ## Method.
    #  @param self The object pointer. 
    #  @param state_vector  Vector of state variables.    
    def Get_Hamiltonian(self,state_vector):
        """
        Compute the Hamiltonian of the system at the state vector
        """        
        assert self.nt == state_vector.shape[0], "Wrong dimension in the vector"
        
        # We exploit the quadratic feature of the Hamiltonian    
        
        return 0.5 * np.dot(state_vector,self.Apply_gradient_Hamiltonian(state_vector))
        
    ## Method.
    #   
    #  @param self The object pointer. 
    #  @param state_vector  Vector of state variables.    
    def Set_Subsystem(self):
        """
        Set the complete subsystem
        """        
        
        assert self.Define_structure_matrix() == 1, "The structure matrix is not defined."
    
        assert self.Define_dissipation_matrix() == 1, "The dissipation matrix is not defined."
    
        assert self.Define_Hamiltonian_quadratic_component() == 1, "The quadratic part of the Hamiltonian is not defined."
      
        self.status = 1
        
        return self.status
        
if __name__ == '__main__':
    
    import numpy
    import math 
    import os
    from dynamics_manager import Dynamical
    
    def command_line_options():
        """
        Command line parsing for the linear oscillator problem
        """
        
        import argparse
        import shutil
        
        parser = argparse.ArgumentParser()
        
        parser.add_argument('--ma', '--mass',         type=float, default=1.0,           help='Mass of the oscillator [kg]')
        parser.add_argument('--om', '--omega',        type=float, default=2.0,           help='Pulsation of the oscillator [rad/s]')
        parser.add_argument('--am', '--amplitude',    type=float, default=1.0,           help='Amplitude of the solution [m]')
        parser.add_argument('--ph', '--phase',        type=float, default=0.0,           help='Phase of the solution [rad]')
        parser.add_argument('--ti', '--time_initial', type=float, default=0.,            help='Initial time [s]')
        parser.add_argument('--tf', '--time_final',   type=float, default=100.,          help='Final time')
        parser.add_argument('--ts', '--time_step',    type=float, default=0.01,          help='Time step [s]')
        parser.add_argument('--mo', '--mode',         type=str,   default="Interactive", help='Mode of execution (use Interactive for screen plots)')
        parser.add_argument('--sc', '--scheme',       action='store', dest='mylist',
                            type=str, nargs='*',      default=['Stoermer_Verlet','Stoermer_Verlet_inverse','Asymmetric_Euler_B'],
                            help='Integration schemes (use any of the following schemes: Exponential,Exponential_integrator,Stoermer_Verlet,\
                            Stoermer_Verlet_inverse,Asymmetric_Euler_B)')  
        parser.add_argument('--sd', '--store_directory',  type=str,   default="Current", help='Location of the store directory')                  
        parser.add_argument('--of', '--output_file',  type=str,   default="Default", help='Name of the simulation output file') 
        
        
        args = parser.parse_args()
        
        mass         = args.ma
        omega        = args.om
        qm           = args.am
        phi          = args.ph 
        time_initial = args.ti
        time_final   = args.tf
        time_step    = args.ts
        mode         = args.mo
        scheme_list  = args.mylist
        store_directory = args.sd
        output_file  = args.of
 
        if store_directory == "Current":
           store_directory = os.getcwd()
        else:
            store_directory =  os.path.join(os.getcwd(),store_directory) 
            if not (os.path.isdir(store_directory)):  
               os.mkdir(store_directory)
      
        if output_file == "Default":
           output_file = 'linear_oscillator' 
                       
        output_file = os.path.join(store_directory,output_file) 
           
        return mass, omega, qm, phi, time_initial, time_final, time_step, mode, scheme_list, store_directory, output_file
        
    # Retrieve command line arguments if any
        
    mass,omega,qm,phi,time_initial,time_final,time_step, mode, scheme_list, store_directory, output_file = command_line_options()
    #print(mode)
    #print(output_file)
    #print(store_directory)
    
    # Initial state vector
    
    initial_state    = numpy.zeros(2)
    initial_state[0] = qm*math.cos(time_initial*omega + phi)
    initial_state[1] =-qm*omega*math.sin(time_initial*omega + phi)
  
    # Set the matrices of the linear oscillator
  
    M = numpy.zeros(shape=(1,1))
    C = numpy.zeros(shape=(1,1)) 
    G = numpy.zeros(shape=(1,1))
    K = numpy.zeros(shape=(1,1))
    
    M[0,0] = mass
    K[0,0] = omega*omega
    
    # Call the constructor for the Linear Oscillator
    
    Lo = Linear_Oscillator(M,C,G,K)
        
    # Set the subsystem
    
    Lo.Set_Subsystem() 
        
    #print Lo.J
    #print Lo.R
    #v    = numpy.zeros(2)
    #Lo.Set_source_term(v)
    
    # Define the dynamical system
    
    D = Dynamical(time_initial,time_final,time_step,mode)
    
    # Plot the relative energy error
    
    D.Energy_error_plot(Lo,initial_state,scheme_list)

    # Store output files 

    cmd = 'mv energy.res ' + output_file +'.res'
    os.system(cmd)
    
    cmd = 'mv energy.pdf ' + output_file +'.pdf'
    os.system(cmd)
  
    #print(Lo.B)
    
    #print(Lo.D)
    
    #print(Lo.J)
    
    #print(Lo.K)
    
    #print(Lo.Q)

    Lo.Set_system_matrix()
    
    #print(Lo.S) 
    
    #print(Lo.ncc)
