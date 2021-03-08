#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors   : A. Serhani & X. Vasseur
Template  : Wave_2D as a port-Hamiltonian system        
Project   : INFIDHEM https://websites.isae-supaero.fr/infidhem/
Openforge : https://openforge.isae.fr/projects/scrimp
Contact   : anass.serhani@isae.fr & xavier.vasseur@isae.fr
"""

from dolfin import *
from mshr import *

import numpy as np
import scipy.linalg as linalg
import scipy.integrate as integrate
from scipy.sparse import csc_matrix, csr_matrix, save_npz, bmat, block_diag
from scipy.sparse import issparse
from scipy.sparse.linalg import factorized, spsolve, eigs
from scikits import umfpack

from assimulo.problem import Explicit_Problem, Implicit_Problem
from assimulo.solvers.sundials import IDA, CVode 

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as manimation

import sys
import time


        
class Wave_2D:
    #%% CLASS INSTANTIATION
    def __init__(self):
        """
        Constructor for the Wave_2D class
        """
        # Information related to the problem definition
        self.problem_definition               = 0                  
        self.set_boundary_control             = 0
        self.set_damping                      = 0
        self.damp                             = []    
        self.dynamic_R                        = False
        self.set_initial_final_time           = 0
        self.set_mixed_boundaries             = 0
        self.set_dirichlet_boundary_control   = 0
        self.set_normal_boundary_control      = 0
        self.set_physical_parameters          = 0
        self.set_rectangular_domain           = 0
        
        # Information related to the space-time discretization
        self.assembly                         = 0
        self.generate_mesh                    = 0
        self.project_boundary_control         = 0
        self.project_initial_data             = 0
        self.set_finite_elements_spaces       = 0

        
        # Information related to the post-processing
        self.docker                           = False
        self.interactive                      = False
        self.notebook                         = False
        
        # Information related to printing
        self.verbose                          = True
        
        # Information related to the memory 
        self.memory_constrained               = False
        
    ## Method.
    #  @param self The object pointer.        
    def __call__(self):
        if self.verbose: print('Wave class of Scrimp')
        if self.verbose: print(40*'-', '\n')
        
    #%% PROBLEM DEFINITION
    
    ## Method.
    #  @param self The object pointer.
    def Check_Problem_Definition(self):
        """
        Check if the problem definition has been performed correctly
        """             
        assert self.set_rectangular_domain == 1, \
            'The domain has not been defined properly'

        assert self.set_physical_parameters == 1, \
            'The physical parameters have not been set properly'
        
        assert self.set_boundary_control == 1 or \
        ( self.set_dirichlet_boundary_control == 1 and self.set_normal_boundary_control  == 1), \
            'Boundary control has not been set properly'

        assert self.set_initial_final_time == 1, \
            'The initial and final times have not been set' 
            
        self.problem_definition = 1
        
        return self.problem_definition
    
    ## Method.
    #  @param self The object pointer.
    def Set_Boundary_Control(self, Ub_tm0, Ub_sp0, Ub_tm1=lambda t : 0, Ub_sp1='0', **kwargs):
        """
        Set the boundary control as a callable time function and regular FeniCS expression
        u_\partial(t,x) = Ub_tm0(t) * Ub_sp0(x) + Ub_tm1(t) + Ub_sp1(x)
        """
        assert self.set_rectangular_domain == 1, 'The domain must be defined before.'
        
        self.Ub_tm0 = Ub_tm0

        self.Ub_sp0_Expression = Expression(Ub_sp0, degree=2,\
                                            #x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                            **kwargs)
        
        self.Ub_tm1 = Ub_tm1

        self.Ub_sp1_Expression = Expression(Ub_sp1, degree=2,\
                                            #x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                            **kwargs)
        
        if self.verbose: print(40*'-')
        if self.verbose: print('Boundary control: OK')
        if self.verbose: print(40*'-')
        
        self.set_boundary_control = 1
        
        return self.set_boundary_control
    
    ## Method.
    #  @param self The object pointer.
    def Set_Damping(self, damp=[], Rtime_func=None, Z=None, Y=None, eps=None, k11=None, k12=None, k22=None, uKV_included=False, **kwargs):
        """
        Set damping parameters in the definition of the PDE
        """
        
        assert self.set_rectangular_domain == 1, 'The domain must be defined before.'
        
        if Rtime_func is not None :
            self.dynamic_R = True
            self.Rtime = Rtime_func
        else : self.dynamic_R = False
                
        self.damp = damp
        
        if 'impedance' in self.damp or 'impedance_mbc' in self.damp :
            self.Z = Expression(Z, degree=2,\
                                x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                **kwargs)

        if 'admittance' in self.damp :
            self.Y = Expression(Y, degree=2,\
                                x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                **kwargs)

        if 'fluid' in self.damp :
            self.eps = Expression(eps, degree=2,\
                                  x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                  **kwargs)
            
        if 'viscoelastic' in self.damp :
            self.kappa = Expression( ( (k11, k12), (k12, k22) ), degree=2, \
                                x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                **kwargs)
            self.uKV_included = uKV_included
        
        
        if self.verbose: print(40*'-')
        if self.verbose: print('Damping:', ((len(damp)*'%1s,  ') % tuple(damp))[:-3])
        if self.verbose: print('Damping: OK')
        if self.verbose: print(40*'-')

        self.set_damping = 1
        
        return self.set_damping

    ## Method.
    #  @param self The object pointer. 
    def Set_Initial_Data(self, W_0=None, Aq_0_1=None, Aq_0_2=None, Ap_0=None,\
                         init_by_vector=False, W0=None, Aq0=None, Ap0=None, **kwargs):
        """
        Set initial data for the variables of the pHS
        """
        
        assert self.set_rectangular_domain == 1, 'The domain must be defined before.'
        
        # Expressions
        if not init_by_vector :
            self.W_0  = Expression(W_0, degree=2, x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL, **kwargs)
            self.Aq_0 = Expression((Aq_0_1, Aq_0_2), degree=2, x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL, **kwargs) 
            self.Ap_0 = Expression(Ap_0, degree=2, x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL, **kwargs) 
            self.init_by_vector = False
        else :
            # Vectors
            self.W0             = W0
            self.Aq0            = Aq0
            self.Ap0            = Ap0
            self.A0             = np.concatenate((self.Aq0,self.Ap0))
            self.init_by_vector = True

        self.set_initial_data = 1

        return self.set_initial_data
   
    
    ## Method.
    #  @param self The object pointer.
    def Set_Initial_Final_Time(self, initial_time, final_time):
        """
        Set the initial and final times for defining the time domain
        """
        self.tinit  = initial_time 
        self.tfinal = final_time
        
        self.set_initial_final_time = 1
        
        return self.set_initial_final_time
    
    ## Method.
    #  @param self The object pointer.    
    def Set_Mixed_Boundaries(self, Dir=[], Nor=[], Imp=[]):
        """
        Set the type of spatial boundary conditions on each part of the domain
        """
        self.Dir = Dir
        self.Nor = Nor
        self.Imp = Imp
        
        
        if self.verbose: print(40*'-')
        if self.verbose: print('Dirichlet boundary:', (len(Dir)*'%s,  ' % tuple(Dir))[:-3])
        if self.verbose: print('Normal boundary:  ', (len(Nor)*'%s,  ' % tuple(Nor))[:-3])
        if self.verbose: print('Impedance boundary:', (len(Imp)*'%s,  ' % tuple(Imp))[:-3])
        if self.verbose: print('Mixed Boundaries: OK')
        if self.verbose: print(40*'-', '\n')
        
        self.set_mixed_boundaries = 1
        
        return self.set_mixed_boundaries
    
    ## Method.
    #  @param self The object pointer.
    def Set_Mixed_BC_Dirichlet(self, Ub_tm0, Ub_sp0, Ub_tm1=lambda t : 0, Ub_sp1='0',\
                               Ub_tm0_dir=lambda t : 0, Ub_tm1_dir=lambda t : 0, **kwargs):
        """
        Set boundary control as callable time functions and regular FeniCS expression
        u_\partial(t,x) = Ub_tm0(t) * Ub_sp0(x) + Ub_tm1(t) + Ub_sp1(x)
        """
        assert self.set_rectangular_domain == 1, 'The domain must be defined before.'
        
        self.Ub_tm0_D = Ub_tm0

        self.Ub_sp0_D_Expression = Expression(Ub_sp0, degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub_tm1_D = Ub_tm1

        self.Ub_sp1_D_Expression = Expression(Ub_sp1, degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        self.Ub_tm0_D_dir = Ub_tm0_dir
        self.Ub_tm1_D_dir = Ub_tm1_dir
        
        
        self.set_dirichlet_boundary_control = 1
        
        return self.set_dirichlet_boundary_control
    
    
    ## Method.
    #  @param self The object pointer.
    def Set_Mixed_BC_Normal(self, Ub_tm0, Ub_sp0, Ub_tm1=lambda t : 0, Ub_sp1='0', **kwargs):
        """
        Set boundary control as callable time functions and regular FeniCS expression
        u_\partial(t,x) = Ub_tm0(t) * Ub_sp0(x) + Ub_tm1(t) + Ub_sp1(x)
        """
        
        assert self.set_rectangular_domain == 1, 'The domain must be defined before.'
        
        self.Ub_tm0_N = Ub_tm0

        self.Ub_sp0_N_Expression = Expression(Ub_sp0, degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        
        self.Ub_tm1_N = Ub_tm1

        self.Ub_sp1_N_Expression = Expression(Ub_sp1, degree=2,\
                                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                                              **kwargs)
        self.set_normal_boundary_control = 1

        return self.set_normal_boundary_control
    
    ## Method.
    #  @param self The object pointer.
    def Set_Physical_Parameters(self, rho, T11, T12, T22, **kwargs):
        """
        Set the physical parameters as a FeniCS expression related to the PDE
        """
        
        assert self.set_rectangular_domain == 1, 'The domain must be defined before.'
        
        self.rho = Expression(rho, degree=2,\
                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                              **kwargs)
        self.T   = Expression( ( (T11, T12), (T12, T22) ), degree=2,\
                              x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                              **kwargs)
        
        Tdet = '1./('+'('+T11+')'+'*'+'('+T22+')' + '-' + '('+T12+')'+'*'+'('+T12+'))'
        
        self.Tinv = Expression( ( (Tdet+'*'+'('+T22+')', Tdet+'*'+'(-'+T12+')'),\
                                 (Tdet+'*'+'(-'+T12+')', Tdet+'*'+'('+T11+')') ),\
                               degree=2,\
                               x0=self.x0, xL=self.xL, y0=self.y0, yL=self.yL,\
                               **kwargs)
        
        self.set_physical_parameters = 1
        
        return self.set_physical_parameters
    
    ## Method.
    # @param self The object pointer.
    def Set_Rectangular_Domain(self, x0, xL, y0, yL):
        """
        Set the dimensions of the rectangular domain
        """
        self.x0 = x0
        self.xL = xL
        self.y0 = y0
        self.yL = yL
        
        self.set_rectangular_domain = 1

        return self.set_rectangular_domain
    
        
    #%% SPACE-TIME DISCRETIZATION
 
        ## Method.
    #  @param self The object pointer.   
    def Assembly(self, formulation = 'Grad'):
        """
        Perform the matrix assembly related to the PFEM formulation
        """
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
        
        assert self.set_finite_elements_spaces == 1, \
                "The finite element spaces must be selected first"
        
        # Functions
        aq_tl, ap_tl = TrialFunction(self.Vq), TrialFunction(self.Vp)
        vq_tl, vp_tl = TestFunction(self.Vq), TestFunction(self.Vp)
        
        ab, vb = TrialFunction(self.Vb), TestFunction(self.Vb)
        
        # Mass matrices
        Mq = assemble( dot(self.Tinv * aq_tl, vq_tl) * dx).array()
        Mp = assemble( self.rho * ap_tl * vp_tl * dx).array()

        self.Mq = Mq
        self.Mp = Mp

        self.M = linalg.block_diag(Mq, Mp)
        
        # Stiffness matrices
        if formulation == 'Div' :
            D = assemble(- ap_tl * div(vq_tl) * dx).array()
        elif formulation == 'Grad' :
            D = assemble( dot(grad(ap_tl), vq_tl) * dx).array()
        self.D = D
        Zeroq, Zerop = np.zeros((self.Nq,self.Nq)), np.zeros((self.Np,self.Np))
        self.J = np.block([ [ Zeroq,   D  ],  
                            [ -D.T, Zerop ] ]) 
            
        # Boundary matrices
        self.Mb = assemble( ab * vb * ds).array()[self.b_ind,:][:,self.b_ind]
        
        if formulation == 'Div' :
            self.B    = assemble( ab * dot(vq_tl, self.norext) * ds).array()[:,self.b_ind] 
            self.Bext = np.concatenate((self.B,np.zeros((self.Np,self.Nb))))
        if formulation == 'Grad' :
            self.B    = assemble( ab * vp_tl * ds).array()[:,self.b_ind] 
            self.Bext = np.concatenate((np.zeros((self.Nq,self.Nb)), self.B))

        # Dissipation matrix
        self.R = np.zeros((self.Nsys, self.Nsys))
    
        if 'impedance' in self.damp :
            self.Mz = assemble( self.Z * ab * vb * ds).array()[self.b_ind,:][:,self.b_ind]
            self.Rz = self.B @ linalg.solve(self.Mb, self.Mz) @ linalg.solve(self.Mb, self.B.T)
            self.R += linalg.block_diag(self.Rz, Zerop)
            
        if 'admittance' in self.damp :
            self.My = assemble( self.Y * ab * vb * ds).array()[self.b_ind,:][:,self.b_ind]
            self.Ry = self.B @ linalg.solve(self.Mb, self.My) @ linalg.solve(self.Mb, self.B.T)
            self.R += linalg.block_diag(Zeroq, self.Ry)

        if 'fluid' in self.damp :
            self.Ri = assemble( self.eps * ap_tl * vp_tl * dx).array()
            self.R += linalg.block_diag(Zeroq, self.Ri)

        if 'viscoelastic' in self.damp :
            kappa_itrp = interpolate(self.kappa, TensorFunctionSpace(self.Msh, 'P', 6))
            self.Rkv = assemble( dot(self.kappa * grad(ap_tl), grad(vp_tl)) * dx).array() 
            if not self.uKV_included :
                self.Rkv += - assemble(dot(self.kappa * grad(ap_tl), self.norext) * vp_tl * ds).array()
            self.R += linalg.block_diag(Zeroq, self.Rkv)
            
            self.MR = assemble( dot(aq_tl, vq_tl) * dx).array()
            self.Mkappa = assemble( dot(self.kappa * aq_tl, vq_tl) * dx).array()
            self.G = assemble( - dot(aq_tl, grad(vp_tl)) * dx).array()
            self.C = assemble( dot(aq_tl, self.norext) * vp_tl * ds).array()
            
    
        if self.dynamic_R :
            def Rtime(t):
                return self.R * self.Rtime(t)
            self.Rdyn = Rtime 
    
        self.assembly = 1
        
        return self.assembly

    ## Method.
    # @param self The object pointer
    def Assembly_Mixed_BC(self):
        """
                   Gamma_4
                 —— —— —— ——
                |           | 
        Gamma_1 |           | Gamma_3
                |           |
                 —— —— —— —— 
                   Gamma_2 
        Perform the matrix assembly related to the PFEM formulation with the 
        definition given above for the boundaries
        """
        
        self.Assembly(formulation='Grad')

        
        self.Gamma_1, self.Gamma_2, self.Gamma_3, self.Gamma_4 = [], [], [], [] 
        for i in range(len(self.coord_b_full)) :
            if np.abs(self.coord_b_full[i,0] - self.x0) <= DOLFIN_EPS :
                self.Gamma_1.append(i)
            if np.abs(self.coord_b_full[i,1] - self.y0) <= DOLFIN_EPS :
                self.Gamma_2.append(i)
            if np.abs(self.coord_b_full[i,0] - self.xL) <= DOLFIN_EPS :
                self.Gamma_3.append(i)
            if np.abs(self.coord_b_full[i,1] - self.yL) <= DOLFIN_EPS :
                self.Gamma_4.append(i)
        
        # Indexes of each boundary
        self.N_index = []
        self.D_index = []
        self.Z_index = []
        
        if 'G1' in self.Dir :
            self.D_index += self.Gamma_1
        elif 'G1' in self.Nor :
            self.N_index += self.Gamma_1
        elif 'G1' in self.Imp:
            self.Z_index += self.Gamma_1

        if 'G2' in self.Dir :
            self.D_index += self.Gamma_2
        elif 'G2' in self.Nor :
            self.N_index += self.Gamma_2
        elif 'G2' in self.Imp:
            self.Z_index += self.Gamma_2
            
        if 'G3' in self.Dir :
            self.D_index += self.Gamma_3
        elif 'G3' in self.Nor :
            self.N_index += self.Gamma_3
        elif 'G3' in self.Imp:
            self.Z_index += self.Gamma_3
            
        if 'G4' in self.Dir :
            self.D_index += self.Gamma_4
        elif 'G4' in self.Nor :
            self.N_index += self.Gamma_4
        elif 'G4' in self.Imp:
            self.Z_index += self.Gamma_4

        if 'Inside' in self.Dir :
            self.D_index += self.Ins_index
        elif 'Inside' in self.Nor :
            self.N_index += self.Ins_indxx
        elif 'Inside' in self.Imp:
            self.Z_index += self.Ins_index

            
        # Remove common indexes
        for i in self.D_index :
           if i in self.N_index :
               self.N_index.remove(i)
           if i in self.Z_index :
               self.Z_index.remove(i)
        for i in self.Z_index :
           if i in self.N_index :
               self.N_index.remove(i)       
                
        self.D_index = list(np.unique(np.array(self.D_index)))     
        self.N_index = list(np.unique(np.array(self.N_index)))
        self.Z_index = list(np.unique(np.array(self.Z_index)))      

        self.Nb_D, self.Nb_N, self.Nb_Z = len(self.D_index), len(self.N_index), len(self.Z_index)
        
        # New Boundary Matrices
        aq_tl, vp_tl = TrialFunction(self.Vq), TestFunction(self.Vp)
        ab, vb = TrialFunction(self.Vb), TestFunction(self.Vb)
        
        self.Mb_D = assemble( ab * vb * ds ).array()[self.D_index, :][:, self.D_index]
        self.Mb_N = assemble( ab * vb * ds ).array()[self.N_index, :][:, self.N_index]
        self.Mb_Z = assemble( ab * vb * ds ).array()[self.Z_index, :][:, self.Z_index]

        self.B_D = assemble( ab * vp_tl * ds ).array()[:, self.D_index]
        self.B_Dext = np.concatenate(([np.zeros((self.Nq, self.Nb_D)), self.B_D]))    

        self.B_N = assemble( ab * vp_tl * ds ).array()[:, self.N_index]
        self.B_Next = np.concatenate(([np.zeros((self.Nq, self.Nb_N)), self.B_N]))
        
        self.B_Z = assemble( ab * vp_tl * ds ).array()[:, self.Z_index]
        
        try : self.MZ = assemble( ab * self.Z * vb * ds ).array()[self.Z_index, :][:, self.Z_index]
        except : self.MZ = assemble( ab * vb * ds ).array()[self.Z_index, :][:, self.Z_index]
        self.Rz = self.B_Z @ linalg.solve(self.MZ, self.B_Z.T)
        self.R += linalg.block_diag(np.zeros((self.Nq,self.Nq)), self.Rz)
    
        # Lagrange multiplier initial data
        self.B_normal_D = assemble( dot(aq_tl, self.norext) * vb * ds ).array()[self.D_index, :]

        if self.verbose: print(40*'-')
        if self.verbose: print('Nb_D=', self.Nb_D, ',\t Nb_N=', self.Nb_N, ',\t Nb_Z=', self.Nb_Z)
        if self.verbose: print('DOFsysDAE=', self.Nsys + self.Nb_D)
        if self.verbose: print('DAE system: OK')
        if self.verbose: print(40*'-')
        
        self.assembly = 1
        
        return self.assembly
    
    ## Method.
    # @param self The object pointer.         
    def Assign_Mesh(self, Msh):
        """
        Assign an already generated mesh as an object
        """
        # Mesh
        self.Msh      = Msh
        self.norext   = FacetNormal(self.Msh)
        
        # Explicit coordinates
        self.xv = Msh.coordinates()[:,0]
        self.yv = Msh.coordinates()[:,1]
        self.Nv = self.Msh.num_vertices()


        # Domain limits
        self.x0, self.xL = self.xv.min(), self.xv.max() 
        self.y0, self.yL = self.yv.min(), self.yv.max()
        
        if self.verbose: print(40*'-')
        if self.verbose: print('Mesh:', 'hmax=', round(self.Msh.hmax(),3),'Nv=', self.Msh.num_vertices() )
        if self.verbose: print('Mesh: OK')
        if self.verbose: print(40*'-')
    
        self.generate_mesh = 1
        
        return self.generate_mesh   
    
    ## Method.
    #  @param self The object pointer.
    def Check_Space_Time_Discretization(self):
        """
        Check if the space and time discretizations have been performed correctly
        """
        
        assert self.generate_mesh == 1, \
            'The finite element mesh must be generated first'
        
        assert self.set_finite_elements_spaces == 1, \
            'The FE approximation spaces must be selected first'
            
        assert self.assembly == 1, \
            'The PFEM formulation has to be applied' 
    
        assert self.project_boundary_control == 1, \
            'The BC must be interpolated on the FE spaces' 
        
        assert self.project_initial_data == 1, \
            'The initial data must be interpolated on the FE spaces' 

        assert self.set_time_setting == 1,\
            'The parameters for the time discretization must be set'
    
        self.space_time_discretization = 1
        
        return self.space_time_discretization
      
    ## Method.
    #  @param self The object pointer.   
    def Generate_Mesh(self, rfn, structured_mesh=False):
        """
        Perform the mesh generation through the Fenics meshing functionalities
        """
        self.rfn = rfn  
    
        rfny =  int (self.rfn * (self.yL - self.y0) / (self.xL - self.x0) ) 
        
        if structured_mesh: 
            self.Msh = RectangleMesh(Point(self.x0,self.y0), Point(self.xL,self.yL),\
                                     self.rfn, rfny, 'crossed')
        else:
            self.Msh = generate_mesh(Rectangle(Point(self.x0,self.y0), Point(self.xL,self.yL)),\
                                     self.rfn)
        
        # Explicit coordinates
        
        self.xv = self.Msh.coordinates()[:,0]
        self.yv = self.Msh.coordinates()[:,1]
        self.Nv = self.Msh.num_vertices()
        
        self.norext   = FacetNormal(self.Msh)
        
        if self.verbose: print(40*'-')
        if self.verbose: print('Mesh:', 'hmax=', round(self.Msh.hmax(),3),'Nv=', self.Msh.num_vertices())
        if self.verbose: print('Mesh: OK')
        if self.verbose: print(40*'-')
        
        self.generate_mesh = 1
        
        return self.generate_mesh
   
        ## Method.
    #  @param self The object pointer.
    def Project_Boundary_Control(self):
        """
        Project boundary control on the FE spaces 
        """        
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
    
        assert self.set_finite_elements_spaces == 1, \
                "The FE approximation spaces must be selected first"
       
        if self.set_mixed_boundaries == 0 : 
            self.Ub_sp0 = interpolate(self.Ub_sp0_Expression, self.Vb).vector()[self.b_ind]
            self.Ub_sp1 = interpolate(self.Ub_sp1_Expression, self.Vb).vector()[self.b_ind]
            self.Ub = lambda t : self.Ub_sp0 * self.Ub_tm0(t) + self.Ub_sp1 + self.Ub_tm1(t) * np.ones(self.Nb)
        
        if self.set_mixed_boundaries == 1 :
            self.Ub_sp0_D = interpolate(self.Ub_sp0_D_Expression, self.Vb).vector()[self.D_index]
            self.Ub_sp1_D = interpolate(self.Ub_sp1_D_Expression, self.Vb).vector()[self.D_index]
            self.Ub_D = lambda t : self.Ub_sp0_D * self.Ub_tm0_D(t) + self.Ub_sp1_D + self.Ub_tm1_D(t) * np.ones(self.Nb_D)
            self.Ub_D_dir = lambda t : self.Ub_sp0_D * self.Ub_tm0_D_dir(t) + self.Ub_tm1_D_dir(t) * np.ones(self.Nb_D)

            self.Ub_sp0_N = interpolate(self.Ub_sp0_N_Expression, self.Vb).vector()[self.N_index]
            self.Ub_sp1_N = interpolate(self.Ub_sp1_N_Expression, self.Vb).vector()[self.N_index]
            self.Ub_N = lambda t : self.Ub_sp0_N * self.Ub_tm0_N(t) + self.Ub_sp1_N + self.Ub_tm1_N(t) * np.ones(self.Nb_N)            

        self.project_boundary_control = 1
        
        return self.project_boundary_control

    ## Method.
    #  @param self The object pointer. 
    def Project_Initial_Data(self) :
        """
        Project initial data on the FE spaces 
        """
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
    
        assert self.set_finite_elements_spaces == 1, \
                "The FE approximation spaces must be selected first"
       
        if not self.init_by_vector :
            self.W0  = interpolate(self.W_0, self.Vp).vector()[:]
            self.Aq0 = interpolate(self.Aq_0, self.Vq).vector()[:]
            self.Ap0 = interpolate(self.Ap_0, self.Vp).vector()[:]
            self.A0  = np.concatenate((self.Aq0, self.Ap0))

        if self.verbose: print(40*'-')
        if self.project_boundary_control == 1 and self.verbose: print('Project BC: OK')
        if self.verbose: print('Project initial data: OK')
        if self.verbose: print(40*'-')
        
        self.project_initial_data = 1
        
        return self.project_initial_data
    
    ## Method.
    #  @param self The object pointer.   
    def Set_Finite_Element_Spaces(self, family_q, family_p, family_b, rq, rp, rb):        
        """
        Set the finite element approximation spaces related to the space discretization of the PDE
        """
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
        
        # Spaces
        if family_q == 'P':
            if rq == 0 :  
                self.Vq = VectorFunctionSpace(self.Msh, 'DG', 0, dim=2) 
            else :  
                self.Vq = VectorFunctionSpace(self.Msh, 'P', rq, dim=2) 
        elif family_q == 'RT' :
            self.Vq = FunctionSpace(self.Msh, 'RT', rq+1)
        else :
            self.Vq = FunctionSpace(self.Msh, family_q, rq)
  
        if rp == 0 :
            self.Vp = FunctionSpace(self.Msh, 'DG', 0)
        else :
            self.Vp = FunctionSpace(self.Msh, family_p, rp)

        if rb == 0 :
            self.Vb = FunctionSpace(self.Msh, 'CR', 1)
        else :
            self.Vb = FunctionSpace(self.Msh, family_b, rb)
            
        # Orders
        self.rq = rq
        self.rp = rp
        self.rb = rb
        
        # DOFs
        self.Nq = self.Vq.dim()
        self.Np = self.Vp.dim()
        self.Nsys = self.Nq + self.Np
        coord_q = self.Vq.tabulate_dof_coordinates()
        coord_p = self.Vp.tabulate_dof_coordinates()
        
        # Explicit coordinates
        self.xp = coord_p[:,0]
        self.yp = coord_p[:,1]
        
        self.xq = coord_q[:,0]
        self.yq = coord_q[:,1] 
        
        # Boundary DOFs for control and observation: necessary for B matrix        
        coord_b  = self.Vb.tabulate_dof_coordinates()
        xb       = coord_b[:,0]
        yb       = coord_b[:,1]
        
        # Boundary DOFs for control and observation: necessary for B matrix        
        coord_b = self.Vb.tabulate_dof_coordinates()
        def Get_Associated_DOFs(SubDomain, Space):
            BC = DirichletBC(Space, '0', SubDomain)
            bndr_dofs = []
            for key in BC.get_boundary_values().keys() :
                bndr_dofs.append(key)
            bndr_dofs = sorted(list(set(bndr_dofs)))
            return bndr_dofs
        b_ind = Get_Associated_DOFs('on_boundary', self.Vb)
        self.b_ind = b_ind

        # Exlpicit information about boundary DOFs 
        self.coord_b_full = coord_b
        coord_b      = coord_b[self.b_ind,:]
        self.Nb      = len(self.b_ind)
        self.xb      = xb[b_ind]
        self.yb      = yb[b_ind]
           
        # Corner indices (boundary DOFs)
        self.Corner_indices = []
        for i in range(self.Nb) :
            if  ( np.abs(self.xb[i] - self.x0) <= 1e-16 and np.abs(self.yb[i] - self.y0) <= 1e-16 ) \
            or ( np.abs(self.xb[i] - self.x0) <= 1e-16 and np.abs(self.yb[i] - self.yL) <= 1e-16 ) \
            or ( np.abs(self.xb[i] - self.xL) <= 1e-16 and np.abs(self.yb[i] - self.y0) <= 1e-16 ) \
            or ( np.abs(self.xb[i] - self.xL) <= 1e-16 and np.abs(self.yb[i] - self.yL) <= 1e-16 ) : 
                 self.Corner_indices.append(i)
         
        if self.verbose: print(40*'-')    
        if self.verbose: print('Vq=', family_q+'_'+str(rq), ',\t Vp=', family_p+'_'+str(rp), ',\t Vb=', family_b+'_'+str(rb))
        if self.verbose: print('Nq=', self.Nq, ',\t Np=', self.Np, ',\t Nb=', self.Nb)
        if self.verbose: print('DOFsys=', self.Nq+self.Np)
        if self.verbose: print('FE spaces: OK')
        if self.verbose: print(40*'-')
        
        self.set_finite_elements_spaces = 1
        
        return self.set_finite_elements_spaces
  
    ## Method.
    # @param self The object pointer. 
    def Set_Gmsh_Mesh(self, xmlfile, rfn_num=0):
        """
        Set a mesh generated by Gmsh using the "xml" format
        """
        # Create FENCIS mesh
        Msh = Mesh(xmlfile)

        num_vertices = Msh.num_vertices()
        num_cells = Msh.num_cells()
        vertices = Msh.coordinates()[:, :2]
        cells = Msh.cells()
        
        Msh = Mesh()
        editor = MeshEditor()
        editor.open(Msh, 'triangle', 2, 2) 
        editor.init_vertices(num_vertices)  
        editor.init_cells(num_cells)
        for i in range(num_vertices) :
            editor.add_vertex(i, vertices[i])    
        for i in range(num_cells) :
            editor.add_cell(i, cells[i])    
        editor.close()
        
        # Refinement
        Msh_list = [Msh]
        rfn, rfn_num = 0, rfn_num
        while rfn < rfn_num : 
            Msh = refine(Msh)
            Msh_list.append(Msh)
            rfn += 1
            
        # Mesh
        self.Msh = Msh
        self.norext   = FacetNormal(self.Msh)
        
        # Explicit coordinates
        self.xv = Msh.coordinates()[:,0]
        self.yv = Msh.coordinates()[:,1]
        self.Nv = self.Msh.num_vertices()

        # Domain limits 
        self.x0, self.xL = self.xv.min(), self.xv.max() 
        self.y0, self.yL = self.yv.min(), self.yv.max()

        if self.verbose: print(40*'-')
        if self.verbose: print('Mesh:', 'hmax=', round(self.Msh.hmax(),3),'Nv=', self.Msh.num_vertices() )
        if self.verbose: print('Mesh: OK')
        if self.verbose: print(40*'-')
        
        self.generate_mesh = 1
        
        return self.generate_mesh

    
    ## Method.
    #  @param self The object pointer.           
    def Set_Time_Setting(self, time_step, tf=None):
        """
        Specify the parameters related to the time integration
        """
        if tf is not None: 
            self.tfinal = tf 
            
        self.dt     = time_step
        self.Nt     = int( np.floor(self.tfinal/self.dt) )
        self.tspan  = np.linspace(0,self.tfinal,self.Nt+1)
        
        self.set_time_setting = 1
        
        return self.set_time_setting

    
   
    
    #%% TIME INTEGRATION  
    
    ## Method.
    #  @param self The object pointer.      
    def Time_Integration(self, string_mode, **kwargs):
        """
        Wrapper method for the time integration
        """
        
        if string_mode == 'DAE:Assimulo':
            A, Ham = self.Integration_DAE_Assimulo(**kwargs)
            done   = 1
            
        if string_mode == 'DAE:RK4_Augmented':
            A, Ham = self.Integration_DAE_RK4_Augmented(**kwargs)
            done   = 1        

        if string_mode == 'DAE:SV_Augmented':
            A, Ham = self.Integration_DAE_SV2_Augmented(**kwargs)
            done   = 1        
        
        if string_mode == 'ODE:SV': 
            A, Ham = self.Integration_ODE_SV()
            done   = 1
            
        if string_mode == 'ODE:CN': 
            A, Ham = self.Integration_ODE_CN()
            done   = 1
        
        if string_mode == 'ODE:RK4': 
            A, Ham = self.Integration_ODE_RK4(**kwargs)
            done   = 1  
            
        if string_mode == 'ODE:Scipy': 
            A, Ham = self.Integration_ODE_Scipy(**kwargs)
            done   = 1     
                 
        if string_mode == 'ODE:Assimulo': 
            A, Ham = self.Integration_ODE_Assimulo(**kwargs)
            done   = 1  
           
        assert done == 1, "Unknown time discretization method in Time_integration"
        
        if self.verbose: print(40*'-', '\n')
        
        return A, Ham
  
    ## Method.
    #  @param self The object pointer.    
    def Integration_ODE_Assimulo(self, **kwargs):
        """
        Perform time integration for ODEs with the assimulo package
        """
        assert self.set_time_setting == 1, 'Time discretization must be specified first'

        self.M_sparse  = csc_matrix(self.M)
        self.J         = csr_matrix(self.J)
        self.R         = csr_matrix(self.R)
        self.Bext      = csr_matrix(self.Bext)
        self.JR        = csr_matrix(self.J-self.R)
        
        
        Mass_solver    = umfpack.UmfpackLU(self.M_sparse)
        #Mass_solver   = factorized(self.M_sparse)    
        #my_jac        = csr_matrix(facto((self.J - self.R)))
                
        # Definition of the rhs function required in scipy assimulo
        #Sys = facto(self.J-self.R)
        #Sys_ctrl = facto(self.Bext)
        
        def rhs(t,y):
            """
            Definition of the rhs function required in the ODE part of IVP
            """ 
            if not self.dynamic_R:  
                return Mass_solver.solve(self.my_mult(self.JR,y) + self.my_mult(self.Bext,self.Ub(t)))
            else:
                return Mass_solver.solve(self.my_mult(self.J-self.Rdyn(t),y) + self.my_mult(self.Bext,self.Ub(t)))
            
        #def jacobian(t,y):
        #    """
        #    Jacobian matrix related to the ODE
        #    """
        #    if not self.dynamic_R:
        #        return facto(self.J - self.R)        
        #    else:
        #        return facto(self.J - self.Rdyn(t))    
        
        def jacv(t,y,fy,v):
            """
            Jacobian matrix-vector product related to the ODE formulation
            """
            if not self.dynamic_R:
                #z = (self.J - self.R) @ v  
                z  = self.my_mult(self.JR,v) 
            else:
                #z = (self.J - self.Rdyn(t)) @ v   
                z  = self.my_mult(self.J-self.Rdyn(t),v) 
            return Mass_solver.solve(z)
           
        print('ODE Integration using assimulo built-in functions:')

#
# https://jmodelica.org/assimulo/_modules/assimulo/examples/cvode_with_preconditioning.html#run_example
#
        
        model                     = Explicit_Problem(rhs,self.A0,self.tinit)
        #model.jac                 = jacobian
        model.jacv                = jacv
        sim                       = CVode(model,**kwargs)
        sim.atol                  = 1e-5 
        sim.rtol                  = 1e-5 
        sim.linear_solver         = 'SPGMR' 
        sim.maxord                = 3
        #sim.usejac                = True
        #sim                       = RungeKutta34(model,**kwargs)
        time_span, ODE_solution   = sim.simulate(self.tfinal)
        
        A = ODE_solution.transpose()
        
        # Hamiltonian
        self.Nt    = A.shape[1]-1
        self.tspan = np.array(time_span)
        
        Ham = np.zeros(self.Nt+1)
        
        for k in range(self.Nt+1):
            #Ham[k] = 1/2 * A[:,k].T @ self.M_sparse.dot(A[:,k])
            Ham[k] = 1/2 * self.my_mult(A[:,k].T, self.my_mult(self.M_sparse, A[:,k]))

        self.Ham = Ham
        
        if self.verbose: print('\n')
        if self.verbose: print('Time integration completed !')

    
        return A, Ham
    
    ## Method.
    #  @param self The object pointer.       
    def Integration_ODE_CN(self, theta=0.5):
        """
        $\theta$-scheme for the numerical integration of the ODE system
        """
        assert self.set_time_setting == 1, 'Time discretization must be specified first'
        assert self.dynamic_R        == 0, 'This integration scheme has rather to be used for a fixed dissipation matrix'
        
        self.M_sparse  = csc_matrix(self.M)
        
        self.J         = csr_matrix(self.J)
        self.R         = csr_matrix(self.R)
        self.Bext      = csr_matrix(self.Bext)
        self.JR        = csr_matrix(self.J-self.R)
        
        # Solution and Hamiltonian versus time
        A   = np.zeros((self.Nsys, self.Nt+1))
        Ham = np.zeros(self.Nt+1)
        
        # Initialization
        A[:,0] = self.A0
        Ham[0] = 1/2 * self.my_mult(A[:,0].T, self.my_mult(self.M_sparse, A[:,0]))

        # Factorization
        Mass_solver = umfpack.UmfpackLU(csc_matrix(self.M_sparse - self.dt*theta*(self.JR)))
        
        if not self.memory_constrained:
            Sys      = Mass_solver.solve(self.M + self.dt * (1-theta) * (self.J-self.R))          
            Sys_ctrl = Mass_solver.solve(self.dt* self.Bext)
        
        # Time loop
        for n in range(self.Nt): 
            if not self.memory_constrained:
                A[:,n+1] = Sys @ A[:,n] + Sys_ctrl @ (theta*self.Ub(self.tspan[n+1])+(1.-theta)*self.Ub(self.tspan[n]))
            else:
                A[:,n+1] = Mass_solver.solve(self.my_mult(self.M_sparse + self.dt * (1-theta) * (self.JR), A[:,n]) + \
                           self.my_mult(self.dt* self.Bext, theta*self.Ub(self.tspan[n+1])+\
                                        (1.-theta)*self.Ub(self.tspan[n])))
            
            # Progress bar
            perct = int(n/(self.Nt-1) * 100)  
            bar   = ('Time-stepping : |' + '#' * int(perct/2) + ' ' + str(perct) + '%' + '|')
            sys.stdout.write('\r' + bar)

            #Ham = np.array([ 1/2 * A[:,n] @ self.M_sparse @ A[:,n] for n in range(self.Nt+1) ])
        
            Ham[n+1] = 1/2 * self.my_mult(A[:,n+1].T, self.my_mult(self.M_sparse, A[:,n+1]))
        
        
        if self.verbose: print('\n')
        if self.verbose: print('Time integration completed !')

        self.Ham = Ham
        
        return A, Ham
    
    ## Method.
    #  @param self The object pointer.       
    def Integration_ODE_RK4(self):
        """
        4th order Runge-Kutta scheme for the numerical integration of the ODE system
        """
        assert self.set_time_setting == 1, 'Time discretization must be specified first'
        
        tspan = self.tspan
        dt    = self.dt
        
        # Factorization of the mass matrix with UMFPACK
        
        self.M_sparse  = csc_matrix(self.M)
        Mass_solver    = umfpack.UmfpackLU(self.M_sparse)
        
        self.J         = csr_matrix(self.J)
        self.R         = csr_matrix(self.R)
        self.Bext      = csr_matrix(self.Bext)
        self.JR        = csr_matrix(self.J-self.R)
        
        #if not self.dynamic_R :
            #Sys = facto.solve(self.J-self.R)
            
        #Sys_ctrl = facto.solve(self.Bext) 
        #print('Inversion: OK \n')
        
        #Sys_nondis = facto.solve(self.J)
        def dif_func(t,y):
            if self.dynamic_R :
                #Sys_loc = facto.solve(self.J-self.Rdyn(t))
                #return Sys_loc @ y + Sys_ctrl @ self.Ub(t)
                #return facto.solve((self.J-self.Rdyn(t))@y + self.Bext@self.Ub(t))
                #self.Rdyn(t) = csr_matrix(self.Rdyn(t))
                return Mass_solver.solve(self.my_mult(self.J-self.Rdyn(t),y) + self.my_mult(self.Bext,self.Ub(t)))
            else :
                #return Sys @ y + Sys_ctrl @ self.Ub(t)
                return Mass_solver.solve(self.my_mult(self.JR,y) + self.my_mult(self.Bext,self.Ub(t)))
        
        if self.verbose: print(40*'-')
        
        A      = np.zeros((self.Nsys, self.Nt+1))
        Ham    = np.zeros(self.Nt+1)
        
        A[:,0] = self.A0
        #Ham[0] = 1/2 * A[:,0] @ self.M @  A[:,0]
        Ham[0] = 1/2 * self.my_mult(A[:,0].T, self.my_mult(self.M_sparse, A[:,0])) 
        
        for k in range(self.Nt):
            k1 = dif_func(tspan[k], A[:,k])
            k2 = dif_func(tspan[k] + dt/2, A[:,k] + dt/2 * k1)
            k3 = dif_func(tspan[k] + dt/2, A[:,k] + dt/2 * k2)
            k4 = dif_func(tspan[k] + dt, A[:,k] + dt * k3)
            
            A[:,k+1] = A[:,k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            #Ham[k+1] = 1/2 * A[:,k+1] @ self.M @ A[:,k+1]
            Ham[k+1] = 1/2 * self.my_mult(A[:,k+1].T, self.my_mult(self.M_sparse, A[:,k+1]))
        
            # Progress bar
            perct = int(k/(self.Nt-1) * 100)  
            bar   = ('Time-stepping with RK4 :' + '|' + '#' * int(perct/2) + ' ' + str(perct) + '%' + '|')
            sys.stdout.write('\r' + bar)
        
        if self.verbose: print('\n')
        if self.verbose: print('Time integration completed !')
        
        return A, Ham

    ## Method.
    #  @param self The object pointer.    
    def Integration_ODE_Scipy(self, **kwargs):
        """
        Perform time integration for ODEs with the scipy.integrate.IVP package
        """
        assert self.set_time_setting == 1, 'Time discretization must be specified first'

        self.M_sparse  = csc_matrix(self.M)
        Mass_solver    = umfpack.UmfpackLU(self.M_sparse) 
        
        self.J         = csr_matrix(self.J)
        self.R         = csr_matrix(self.R)
        self.Bext      = csr_matrix(self.Bext)
        self.JR        = csr_matrix(self.J-self.R)
        
        # Definition of the rhs function required in scipy integration
        
        if not self.memory_constrained:
            Sys      = Mass_solver.solve(csc_matrix(self.J-self.R))
            Sys_ctrl = Mass_solver.solve(csc_matrix(self.Bext))
        
        def rhs(t,y):
            """
            Definition of the rhs function required in the ODE part of IVP
            """ 
            if self.memory_constrained:
                if not self.dynamic_R:  
                    #return Sys @ y + Sys_ctrl @ self.Ub(t)
                    return Mass_solver.solve(self.my_mult(self.JR,y) + self.my_mult(self.Bext,self.Ub(t)))
                else:
                    #return (self.J-self.Rdyn(t)) @ y + Sys_ctrl @ self.Ub(t)  
                    return Mass_solver.solve(self.my_mult(self.J-self.Rdyn(t),y) + self.my_mult(self.Bext,self.Ub(t)))
            else:
                if not self.dynamic_R:  
                    return Sys @ y + Sys_ctrl @ self.Ub(t)
                else:
                    return Mass_solver.solve((self.J-self.Rdyn(t)) @ y) + Sys_ctrl @ self.Ub(t)  
                
            
        print('ODE Integration using scipy.integrate built-in functions:')

        ivp_ode    = integrate.solve_ivp(rhs, (self.tinit,self.tfinal), self.A0,\
                                         **kwargs, t_eval=self.tspan, atol=1.e-3)   
        
        A          = ivp_ode.y
        self.Nt    = len(self.tspan) - 1
        
        print("Scipy: Number of evaluations of the right-hand side ",ivp_ode.nfev)
        print("Scipy: Number of evaluations of the Jacobian ",ivp_ode.njev)
        print("Scipy: Number of LU decompositions ",ivp_ode.nlu)
                        
        # Hamiltonian 
        Ham        = np.zeros(self.Nt+1)
        
        for k in range(self.Nt+1):
            #Ham[k] = 1/2 * A[:,k] @ self.M_sparse.dot(A[:,k])
            Ham[k] = 1/2 * self.my_mult(A[:,k].T, self.my_mult(self.M_sparse, A[:,k]))

        self.Ham = Ham
        
        if self.verbose: print('\n')
        if self.verbose: print('Time integration completed !')

    
        return A, Ham

 
   
        ## Method.
    #  @param self The object pointer.       
    def Integration_ODE_SV(self):
        """
        Stoermer-Verlet scheme for the numerical integration of the ODE system
        """
        assert self.set_time_setting == 1, 'Time discretization must be specified first'
        
        Nq = self.Nq
        
        Mq_sparse = csc_matrix(self.M[:Nq, :Nq])
        Mp_sparse = csc_matrix(self.M[Nq:, Nq:])
        
        M_sparse  = block_diag((Mq_sparse, Mp_sparse))
        
        D         = csr_matrix(self.J[:Nq, Nq:])
        Rq        = csr_matrix(self.R[:Nq, :Nq])
        Rp        = csr_matrix(self.R[Nq:, Nq:])
        Bext      = csr_matrix(self.Bext)
        
        Ms_q      = umfpack.UmfpackLU(Mq_sparse)
        Ms_p      = umfpack.UmfpackLU(Mp_sparse)
        
        if not self.memory_constrained:
            Sys_q, Sys_R_q, Sys_ctrl_q = Ms_q.solve(D), Ms_q.solve(Rq), Ms_q.solve(Bext[:Nq, :]) 
            Sys_p, Sys_R_p, Sys_ctrl_p = - Ms_p.solve(D.T), Ms_p.solve(Rp), Ms_p.solve(Bext[Nq:, :])
        
        # Solution and Hamiltonian versus time
        A   = np.zeros( (self.Nsys, self.Nt+1) )
        Ham = np.zeros(self.Nt+1)

        # Initialization
        A[:,0] = self.A0
        Ham[0] = 1/2 * self.my_mult(A[:,0].T, self.my_mult(M_sparse, A[:,0]))
        
        # Time loop
        dt = self.dt
        Ub = self.Ub
        t  = self.tspan
        
        for n in range(self.Nt):   
            Aq = A[:Nq, n]
            Ap = A[Nq:, n]
            if self.memory_constrained:
                # Aqq = Aq + dt/2 * (Sys_q @ Ap - Sys_R_q @ Aq + Sys_ctrl_q @ (Ub(t[n+1]) + Ub(t[n]))/2)
                Aqq  = Aq + dt/2 * Ms_q.solve(self.my_mult(D,Ap)-self.my_mult(Rq,Aq)+self.my_mult(Bext[:Nq, :],(Ub(t[n+1]) + Ub(t[n]))/2))
                # Apn = Ap + dt * (Sys_p @ Aqq - Sys_R_p @ Ap + Sys_ctrl_p @ (Ub(t[n+1]) + Ub(t[n]))/2)
                Apn  = Ap + dt * Ms_p.solve(-self.my_mult(D.T,Aqq)-self.my_mult(Rp,Ap)+self.my_mult(Bext[Nq:, :],(Ub(t[n+1]) + Ub(t[n]))/2))
                # Aqn = Aqq + dt/2 * (Sys_q @ Apn - Sys_R_q @ Aqq + Sys_ctrl_q @ (Ub(t[n+1]) + Ub(t[n]))/2)
                Aqn = Aqq + dt/2 * Ms_q.solve(self.my_mult(D,Apn)-self.my_mult(Rq,Aqq)+self.my_mult(Bext[:Nq, :],(Ub(t[n+1]) + Ub(t[n]))/2))
            else:
                Aqq = Aq + dt/2 * (Sys_q @ Ap - Sys_R_q @ Aq + Sys_ctrl_q @ (Ub(t[n+1]) + Ub(t[n]))/2)
                #Aqq  = Aq + dt/2 * Mass_solver_q.solve(self.my_mult(D,Ap)-self.my_mult(Rq,Aq)+self.my_mult(Bext[:Nq, :],(Ub(t[n+1]) + Ub(t[n]))/2))
                Apn = Ap + dt * (Sys_p @ Aqq - Sys_R_p @ Ap + Sys_ctrl_p @ (Ub(t[n+1]) + Ub(t[n]))/2)
                #Apn  = Ap + dt * Mass_solver_p.solve(-self.my_mult(D.T,Aqq)-self.my_mult(Rp,Ap)+self.my_mult(Bext[Nq:, :],(Ub(t[n+1]) + Ub(t[n]))/2))
                Aqn = Aqq + dt/2 * (Sys_q @ Apn - Sys_R_q @ Aqq + Sys_ctrl_q @ (Ub(t[n+1]) + Ub(t[n]))/2)
                #Aqn = Aqq + dt/2 * Mass_solver_q.solve(self.my_mult(D,Apn)-self.my_mult(Rq,Aqq)+self.my_mult(Bext[:Nq, :],(Ub(t[n+1]) + Ub(t[n]))/2))
            
            A[:Nq, n+1] = Aqn
            A[Nq:, n+1] = Apn
                    
            # Progress bar
            
            perct = int(n/(self.Nt-1) * 100)  
            bar   = ('Time-stepping : |' + '#' * int(perct/2) + ' ' + str(perct) + '%' + '|')
            sys.stdout.write('\r' + bar)
            
            Ham[n+1] = 1/2 * self.my_mult(A[:,n+1].T, self.my_mult(M_sparse, A[:,n+1]))
        
        if self.verbose: print('\n')
        if self.verbose: print('Time integration completed !')

        #Ham = np.array([ 1/2 * A[:,n] @ M_sparse @ A[:,n] for n in range(self.Nt+1) ])

        return A, Ham

    
    
    
    ## Method.
    #  @param self The object pointer.    
    def Integration_DAE_Assimulo(self, **kwargs):
        """
        Wrapper to Assimulo for the numerical integration of the DAE system
        """
        Nsys, Nb_D           = self.Nsys, self.Nb_D
        Ub_N, Ub_D, Ub_D_dir = self.Ub_N, self.Ub_D, self.Ub_D_dir
        
        Nq, Np               = self.Nq, self.Np
        
        self.M_sparse  = csc_matrix(self.M)
        Mass_solver    = umfpack.UmfpackLU(self.M_sparse)
        Mass_b_solver  = umfpack.UmfpackLU(csc_matrix(self.Mb_D))
        
        self.J         = csr_matrix(self.J)
        self.R         = csr_matrix(self.R)
        self.B_Dext    = csr_matrix(self.B_Dext)
        self.B_Next    = csr_matrix(self.B_Next)
        self.JR        = csr_matrix(self.J-self.R)
        
        if not self.memory_constrained:
            Sys        = Mass_solver.solve(csc_matrix(self.J - self.R))
            Sys_Ctrl_N = Mass_solver.solve(csc_matrix(self.B_Next))
            Sys_Ctrl_D = Mass_solver.solve(csc_matrix(self.B_Dext))
        
            try : 
                Sys_Ctrl_DT = umfpack.spsolve(csc_matrix(self.Mb_D), csc_matrix(self.B_Dext).T)
            except : 
                Sys_Ctrl_DT = np.empty((self.Nb_D,self.Nsys))

        
        def PHSDAE(t,y,yd):
            res_0 = np.zeros(Nsys)
            res_1 = np.zeros(Nb_D)
            
            if self.memory_constrained:
                z     = self.my_mult(self.JR,y[:Nsys]) + self.my_mult(self.B_Next,Ub_N(t)) + \
                        self.my_mult(self.B_Dext,y[Nsys:])
                res_0 = yd[:Nsys] - Mass_solver.solve(z)
                res_1 = Ub_D_dir(t) - Mass_b_solver.solve(self.my_mult(self.B_Dext.T,yd[:Nsys]))
            else:
                res_0 = yd[:Nsys] - Sys @ y[:Nsys] - Sys_Ctrl_N @ Ub_N(t) - Sys_Ctrl_D @ y[Nsys:]                  
                res_1 = Ub_D_dir(t) - Sys_Ctrl_DT @ yd[:Nsys] 
            
            return np.concatenate((res_0, res_1))
        
        # The initial conditions
        y0  = np.block([self.A0, linalg.solve(self.Mb_D, self.B_normal_D @ self.A0[:Nq] ) ])# Initial conditions
        yd0 = np.zeros(self.Nsys+Nb_D) # Initial conditions
            
        def handle_result(solver, t ,y, yd):
            global order
            order.append(solver.get_last_order())
            solver.t_sol.extend([t])
            solver.y_sol.extend([y])
            solver.yd_sol.extend([yd])   

        # Create an Assimulo implicit problem
        imp_mod        = Implicit_Problem(PHSDAE, y0, yd0, name='PHSDAE')
           
        # Set the algebraic components
        imp_mod.algvar = list( np.block([np.ones(Nsys), np.zeros(Nb_D)]) ) 
        
        # Create an Assimulo implicit solver (IDA)
        imp_sim        = IDA(imp_mod) #Create a IDA solver
         
        # Sets the paramters
        imp_sim.atol                = 1e-6 #Default 1e-6
        imp_sim.rtol                = 1e-6 #Default 1e-6
        imp_sim.suppress_alg        = True #Suppres the algebraic variables on the error test
        imp_sim.report_continuously = True
             
        # Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
        imp_sim.make_consistent('IDA_YA_YDP_INIT')
         
        # Simulate
        t, y, yd = imp_sim.simulate(self.tfinal, self.tinit, self.tspan) 
        A        = y[:,:self.Nsys].T
        
        # Hamiltonian
        #Ham = np.array([1/2 * A[:,i] @ self.M @ A[:,i] for i in range(self.Nt+1)])
        Ham      =  np.array([1/2 * self.my_mult(A[:,i].T, self.my_mult(self.M_sparse, A[:,i])) for i in range(self.Nt+1)])
        
        if self.verbose: print('\n')
        if self.verbose: print('Time integration completed !')

        
        return A, Ham

    ## Method.
    #  @param self The object pointer.  
    def Integration_DAE_RK4_Augmented(self):
        """
        4th order Runge-Kutta scheme for the numerical integration of the DAE system
        """
        self.M_sparse      = csc_matrix(self.M)
        self.JR            = csr_matrix(self.J - self.R)
        self.B_Dext_sparse = csr_matrix(self.B_Dext)
        self.B_Next_sparse = csr_matrix(self.B_Next)
        
        
        Mass_solver = umfpack.UmfpackLU(self.M_sparse)
        
        if not self.memory_constrained:
            Sys        = Mass_solver.solve(self.JR)
            Sys_Ctrl_N = Mass_solver.solve(self.B_Next_sparse)
        
        BDMinvBDT = self.B_Dext_sparse.T @ Mass_solver.solve(self.B_Dext_sparse) 
        
        if not self.memory_constrained:
            prefix         = Mass_solver.solve(self.B_Dext_sparse)
            Sys_Ctrl_D     = prefix @ linalg.solve(BDMinvBDT, self.Mb_D)
            Sys_Aug        = - prefix @ linalg.solve(BDMinvBDT, self.B_Dext.T) @ Sys
            Sys_Ctrl_N_Aug = - prefix @ linalg.solve(BDMinvBDT, self.B_Dext.T) @ Sys_Ctrl_N
            Sys_AUG        = Sys + Sys_Aug
            Sys_Ctrl_N_AUG = Sys_Ctrl_N + Sys_Ctrl_N_Aug
        
        def dif_aug_func(t,y):
            if self.memory_constrained:
                Sys_y_vec     =   Mass_solver.solve(self.my_mult(self.JR,y))
                Sys_Aug_y_vec = - Mass_solver.solve(self.my_mult(self.B_Dext_sparse,linalg.solve(BDMinvBDT, self.my_mult(self.B_Dext.T,Sys_y_vec))))
                Sys_D_vec     =   Mass_solver.solve(self.my_mult(self.B_Dext_sparse,linalg.solve(BDMinvBDT, self.my_mult(self.Mb_D,self.Ub_D_dir(t)))))
                Sys_N_vec     =   Mass_solver.solve(self.my_mult(self.B_Next_sparse,self.Ub_N(t)))
                Sys_N_Aug_vec = - Mass_solver.solve(self.my_mult(self.B_Dext_sparse,linalg.solve(BDMinvBDT,self.my_mult(self.B_Dext.T,Sys_N_vec))))
                return Sys_y_vec + Sys_Aug_y_vec + Sys_N_vec + Sys_N_Aug_vec + Sys_D_vec
            else:
                return Sys_AUG @ y + Sys_Ctrl_N_AUG @ self.Ub_N(t) + Sys_Ctrl_D @ self.Ub_D_dir(t)
    
        
        A      = np.zeros((self.Nsys, self.Nt+1))
        Ham    = np.zeros(self.Nt+1)
        
        A[:,0] = self.A0
        Ham[0] = 1/2 * self.my_mult(A[:,0].T, self.my_mult(self.M_sparse, A[:,0]))
        
        
        t  = self.tspan
        dt = self.dt
        
        for k in range(self.Nt):
            k1 = dif_aug_func(t[k]       , A[:,k])
            k2 = dif_aug_func(t[k] + dt/2, A[:,k] + dt/2 * k1)
            k3 = dif_aug_func(t[k] + dt/2, A[:,k] + dt/2 * k2)
            k4 = dif_aug_func(t[k] + dt  , A[:,k] + dt * k3)
            
            A[:,k+1] = A[:,k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            Ham[k+1] = 1/2 * self.my_mult(A[:,k+1].T, self.my_mult(self.M_sparse, A[:,k+1]))
        
            # Progress bar
            perct = int(k/(self.Nt-1) * 100)  
            bar   = ('Time-stepping: |' + '#' * int(perct/2) + ' ' + str(perct) + '%' + '|')
            sys.stdout.write('\r' + bar)
            
        #Ham = np.array([1/2 * A[:,k] @ self.M_sparse @ A[:,k] for k in range(self.Nt+1)])

        if self.verbose: print('\n')
        if self.verbose: print('Time integration completed !')

        return A, Ham

    ## Method.
    #  @param self The object pointer.  
    def Integration_DAE_SV2_Augmented(self):
        """
        2nd order Stoermer-Verlet scheme for the numerical integration of the DAE system
        """
        assert self.memory_constrained == False, "Memory might be an issue with this scheme"
        
        self.M_sparse      = csc_matrix(self.M)
        self.JR            = csc_matrix(self.J - self.R)
        self.B_Dext_sparse = csc_matrix(self.B_Dext)
        self.B_Next_sparse = csc_matrix(self.B_Next)
        
        Mass_solver        = umfpack.UmfpackLU(self.M_sparse)
        
        Sys                = Mass_solver.solve(self.JR)
        Sys_Ctrl_N         = Mass_solver.solve(self.B_Next_sparse)
        
        BDMinvBDT          = self.B_Dext_sparse.T @ Mass_solver.solve(self.B_Dext_sparse) 
        
        prefix         = Mass_solver.solve(self.B_Dext_sparse)
        Sys_Ctrl_D     = prefix @ linalg.solve(BDMinvBDT, self.Mb_D)
        Sys_Aug        = - prefix @ linalg.solve(BDMinvBDT, self.B_Dext.T) @ Sys
        Sys_Ctrl_N_Aug = - prefix @ linalg.solve(BDMinvBDT, self.B_Dext.T) @ Sys_Ctrl_N
        
        Sys_AUG        = Sys + Sys_Aug
        Sys_Ctrl_N_AUG = Sys_Ctrl_N + Sys_Ctrl_N_Aug
        
        def dif_aug_func(t,y):
            return Sys_AUG @ y + Sys_Ctrl_N_AUG @ self.Ub_N(t) + Sys_Ctrl_D @ self.Ub_D_dir(t)
        
        Nq = self.Nq
        Np = self.Np
        
        Sys_AUG_qq          = Sys_AUG[:Nq, :Nq]
        Sys_AUG_qp          = Sys_AUG[:Nq, Nq:]
        Sys_AUG_pq          = Sys_AUG[Nq:, :Nq]
        Sys_AUG_pp          = Sys_AUG[Nq:, Nq:]
        
        Sys_Ctrl_N_AUG_q    = Sys_Ctrl_N_AUG[:Nq]
        Sys_Ctrl_N_AUG_p    = Sys_Ctrl_N_AUG[Nq:]        
        
        Sys_Ctrl_D_q        = Sys_Ctrl_D[:Nq]
        Sys_Ctrl_D_p        = Sys_Ctrl_D[Nq:]
        
        
        A = np.zeros((self.Nsys, self.Nt+1))
        
        A[:,0] = self.A0
        
        t  = self.tspan
        dt = self.dt
        
        for k in range(self.Nt):

            App         = A[Nq:, k] + dt/2 * (Sys_AUG_pq @ A[:Nq, k] + Sys_AUG_pp @ A[Nq:, k] + Sys_Ctrl_N_AUG_p @ self.Ub_N(t[k]) + Sys_Ctrl_D_p @ self.Ub_D_dir(t[k]))   
            A[:Nq, k+1] = A[:Nq, k] + dt * (Sys_AUG_qq @ A[:Nq, k] + Sys_AUG_qp @ App + Sys_Ctrl_N_AUG_q @ self.Ub_N(t[k]) + Sys_Ctrl_D_q @ self.Ub_D_dir(t[k])) 
            A[Nq:, k+1] = App + dt/2 * (Sys_AUG_pq @ A[:Nq, k+1] + Sys_AUG_pp @ App + Sys_Ctrl_N_AUG_p @ self.Ub_N(t[k]) + Sys_Ctrl_D_p @ self.Ub_D_dir(t[k]))  
        
            # Progress bar
            perct = int(k/(self.Nt-1) * 100)  
            bar = ('Time-stepping : |' + '#' * int(perct/2) + ' ' + str(perct) + '%' + '|')
            sys.stdout.write('\r' + bar)

        Ham = np.array([1/2 * A[:,k] @ self.M_sparse @ A[:,k] for k in range(self.Nt+1)])

        if self.verbose: print('\n')
        if self.verbose: print('Time integration completed !')

        return A, Ham
    
    ## Method.
    #  @param self The object pointer.  
    def Integration_DAE_RK4_Extended(self):
        """
        4th order Runge-Kutta scheme for the numerical integration of the ODE system
        [To check]
        """
        assert self.set_time_setting == 1, 'Time discretization must be specified first'
        
        tspan = self.tspan
        dt = self.dt
        
        JR = csc_matrix(self.J - self.R)
        self.M_sparse = csc_matrix(self.M)
        
        print('Linear system resolution, this may take a while ...')
        Mfacto = umfpack.UmfpackLU(self.M_sparse)
        Sys = Mfacto.solve_sparse(JR)
        B_D_sparse = csc_matrix(self.B_Dext)
        B_N_sparse = csc_matrix(self.B_Next)
        BMBfacto = umfpack.UmfpackLU(B_D_sparse.T @ Mfacto.solve_sparse(B_D_sparse) )
        
        Sys = Mfacto.solve_sparse(JR)
        
        Sys_ext = JR - B_D_sparse @ BMBfacto.solve_sparse(B_D_sparse.T) @ JR
        Sys_ext = Mfacto.solve_sparse(Sys_ext)
        
        Sys_N = B_N_sparse - B_D_sparse @ \
                        BMBfacto.solve_sparse(B_D_sparse.T @ Mfacto.solve_sparse(B_N_sparse))
        Sys_N = Mfacto.solve_sparse(Sys_N)
        
        Sys_D = B_D_sparse @ BMBfacto.solve_sparse(csc_matrix(self.Mb_D))
        Sys_D = Mfacto.solve_sparse(Sys_D)
        print('Linear system resolution: OK \n')
        
        def dif_func(t,y):
            return Sys_ext @ y + Sys_N @ self.Ub_N(t) + Sys_D @ self.Ub_D_der(t)
        
        A = np.zeros((self.Nsys, self.Nt+1))
        Ham = np.zeros(self.Nt+1)
        
        A[:,0] = self.A0
        Ham[0] = 1/2 * A[:,0] @ self.M @  A[:,0] 
        
        for k in range(self.Nt):
            k1 = dif_func(tspan[k], A[:,k])
            k2 = dif_func(tspan[k] + dt/2, A[:,k] + dt/2 * k1)
            k3 = dif_func(tspan[k] + dt/2, A[:,k] + dt/2 * k2)
            k4 = dif_func(tspan[k] + dt, A[:,k] + dt * k3)
            
            A[:,k+1] = A[:,k] + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
            Ham[k+1] = 1/2 * A[:,k+1] @ self.M @ A[:,k+1]
        
            # Progress bar
            perct = int(k/(self.Nt-1) * 100)  
            bar = ('Time-stepping RK4 : |' + '#' * int(perct/2) + ' ' + str(perct) + '%' + '|')
            sys.stdout.write('\r' + bar)
        
        if self.verbose: print('\n')
        if self.verbose: print('Time integration completed !')

        
        return A, Ham

    ## Method.
    #  @param self The object pointer.    
    def Integration_DAE_KV_Extended(self, **kwargs):
        """
        [To check]
        """
        Nsys, Nq, Np = self.Nsys, self.Nq, self.Np

        
        def PHDAE(t,y,yd):
            res_0 = np.zeros(Nq)
            res_1 = np.zeros(Np)
            res_2 = np.zeros(Nq)
            res_3 = np.zeros(Nq)
            res_0 = self.Mq @ yd[:Nq] - ( self.D @ y[Nq:Nq+Np] + self.B @ self.Ub(t)  )
            res_1 = self.Mp @ yd[Nq:Nq+Np] - ( -self.D.T @ y[:Nq] + self.G @ y[Nq+Np+Nq:] + self.C @ y[Nq+Np+Nq:])
            res_2 = self.MR @ y[Nq+Np:Nq+Np+Nq] - (-self.G.T @ y[Nq:Nq+Np])
            res_3 = self.MR @ y[Nq+Np:Nq+Np+Nq] - (self.Mkappa @ y[Nq+Np+Nq:])
            
            return np.concatenate((res_0, res_1, res_2, res_3))
        
        # The initial conditons
        y0 =  np.block([self.A0, self.Aq0, self.Aq0 ])# Initial conditions
        yd0 = np.zeros(Nq + Np + Nq + Nq) # Initial conditions
            
        def handle_result(solver, t ,y, yd):
            global order
            order.append(solver.get_last_order())
            
            solver.t_sol.extend([t])
            solver.y_sol.extend([y])
            solver.yd_sol.extend([yd])   

        # Create an Assimulo implicit problem
        imp_mod = Implicit_Problem(PHDAE, y0, yd0, name='PHDAE')
           
        # Set the algebraic components
        imp_mod.algvar = list( np.block([np.ones(Nq+Np), np.zeros(Nq+Nq)]) ) 
        
        # Create an Assimulo implicit solver (IDA)
        imp_sim = IDA(imp_mod) #Create a IDA solver
         
        # Sets the paramters
        imp_sim.atol = 1e-6 #Default 1e-6
        imp_sim.rtol = 1e-6 #Default 1e-6
        imp_sim.suppress_alg = True #Suppres the algebraic variables on the error test
        imp_sim.report_continuously = True
             
        # Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
        imp_sim.make_consistent('IDA_YA_YDP_INIT')
         
        # Simulate
        t, y, yd = imp_sim.simulate(self.tfinal, self.tinit, self.tspan ) 
        A = y[:,:Nq+Np].T
        
        # Hamiltonian
        Ham = np.zeros(self.Nt+1)
        for i in range(self.Nt+1):
            Ham[i] = 1/2 * A[:,i] @ self.M @ A[:,i]
        
        if self.verbose: print('\n')
        if self.verbose: print('Time integration completed !')

        
        return A, Ham
   

    
    #%% POST-PROCESSING
    
    ## Method.
    #  @param self The object pointer.   
    def Get_Deflection(self, A):
        """
        Get the deflection variable
        """
        # Get p variables
        Ap  = A[self.Nq:,:]
        rho = interpolate(self.rho, self.Vp).vector()[:]
                
        w      = np.zeros((self.Np,self.Nt+1))
        w[:,0] = self.W0[:]
        
        #if self.verbose: print(40*'-', '\n')  
        
        for n in range(self.Nt):
            w[:,n+1]    = w[:,n] + self.dt * 1/rho[:] * .5 * (  Ap[:,n+1] +  Ap[:,n] ) 
            perct       = int(n/(self.Nt-1) * 100)  
            #bar         = ('Get deflection \t' + '|' + '#' * int(perct/2) + ' ' + str(perct) + '%' + '|')
            #sys.stdout.write('\r' + bar)
            
        #if self.verbose: print(40*'-', '\n')  
        
        return w
    
    ## Method.
    #  @param self The object pointer.   
    def Get_Linear_Momentum(self, A):
        """
        Get the linear momentum variable
        """  
        return A[self.Nq:,:]
    
    ## Method.
    #  @param self The object pointer.   
    def Get_Strain(self, A):
        """
        Get the strain variable
        """  
        return A[:self.Nq,:]   
  
    ## Method.
    #  @param self The object pointer.         
    def Moving_Contour(self, time_space_Vector, step=1, with_mesh=False, title='', save=False, figsize=(8,6.5), **kwargs):
        """
        Create a 2D animation with arrows on vector quantities
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('y coordinate')
        wframe = None
        tstart = time.time()
        temp_vec = Function(self.Vp)
        
        # Save
        if save :
            with self.writer.saving(fig, title+'.mp4', self.dpi):
                if with_mesh : plot(self.Msh, linewidth=.25)
                for i in range(0, self.Nt+1, step):
                    if wframe :
                        ax.collections.remove(wframe)
                    
                    temp_vec.vector()[range(self.Np)] = time_space_Vector[range(self.Np),i] 
                    wframe = plot(temp_vec, title=title+', t='+str(np.round(self.tspan[i],2)), **kwargs)
                    self.writer.grab_frame()
                    plt.pause(.001)
        # Do not Save
        else :
            if with_mesh : plot(self.Msh, linewidth=.25)
            for i in range(0, self.Nt+1, step):
                if wframe :
                    ax.collections.remove(wframe)
                temp_vec.vector()[range(self.Np)] = time_space_Vector[range(self.Np),i] 
                    
                wframe = plot(temp_vec, title=title+', t='+str(np.round(self.tspan[i],2), **kwargs)+'/'+str(self.tfinal))
                plt.pause(.001)
    
    ## Method.
    #  @param self The object pointer.   
    def Moving_Plot(self, y, x,  step=1, title='', save=False, figsize=(8,6.5), **kwargs):
        """
        Create a 2D animation with the plot command
        """
        fig = plt.figure(figsize=figsize)
        plt.xlim(x[0], x[-1])
        plt.ylim(np.min(y)*0.875, np.max(y)*1.125)
        plt.grid(True)
        
        # Save 
        if save :
            with self.writer.saving(fig, title+'.mp4', self.dpi):
                for i in range(0,self.Nt,step):
                    plt.plot(x[i:i+1], y[i:i+1], '.', color='#1f77b4')
                    plt.plot(x[0:i+1], y[0:i+1], '-', color='#1f77b4', **kwargs)
                    plt.title(title+ ' t='+np.array2string(round(x[i],2)) + '/' + np.array2string(round(x[-1],2)))
                    self.writer.grab_frame()
                    plt.pause(0.01)
        # Do not save
        else :
            for i in range(0,self.Nt,step):
                plt.plot(x[i:i+1], y[i:i+1], '.', color='#1f77b4')
                plt.plot(x[0:i+1], y[0:i+1], '-', color='#1f77b4', **kwargs)
                plt.title(title + ' t='+np.array2string(x[i]) + '/' + np.array2string(x[-1]))
                plt.pause(0.01)
        
    # Moving quiver
    def Moving_Quiver(self, time_space_Vector, step=1, with_mesh=False, title='', save=False, figsize=(8,6.5), **kwargs):
        """
        Create a 2D animation with arrows on vector quantities
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.gca()
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('y coordinate')
        wframe = None
        tstart = time.time()
        temp_vec = Function(self.Vq)
        plot(self.Msh, linewidth=0.5)
        
        # Save
        if save :
            with self.writer.saving(fig, title+'.mp4', self.dpi):
                if with_mesh : plot(self.Msh, linewidth=.25)
                for i in range(0, self.Nt+1, step):
                    if wframe :
                        ax.collections.remove(wframe)
                    
                    temp_vec.vector()[range(self.Nq)] = time_space_Vector[range(self.Nq),i] 
                    wframe = plot(temp_vec, title=title+', t='+str(np.round(self.tspan[i],2))+'/'+str(self.tfinal), **kwargs)
                    self.writer.grab_frame()
                    plt.pause(.001)
        # Do not Save
        else :
            if with_mesh : plot(self.Msh, linewidth=.25)
            for i in range(0, self.Nt+1, step):
                if wframe :
                    ax.collections.remove(wframe)
                temp_vec.vector()[range(self.Nq)] = time_space_Vector[range(self.Nq),i] 
                wframe = plot(temp_vec, title=title+', t='+str(np.round(self.tspan[i],2))+'/'+str(self.tfinal), **kwargs)
                plt.pause(.001)   
 
    ## Method.
    #  @param self The object pointer. 
    def Moving_Trisurf(self, SpaceTimeVector_Lag, step=1, title='', save=False, figsize=(8,6.5), cmap=plt.cm.CMRmap, **kwargs):
        """
        Create a 3D animation with the plot_trisurf Matplotlib command
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection='3d')
        #fig.colorbar(wfram, shrink=0.5, aspect=5)
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('y coordinate')
        plt.title(title)
        ax.set_xlim(self.x0, self.xL)
        ax.set_ylim(self.y0, self.yL)
        ax.set_zlim(np.min(SpaceTimeVector_Lag), np.max(SpaceTimeVector_Lag))
        
        wframe = None
        
        # Save
        if save :
            with self.writer.saving(fig, title+'.mp4', self.dpi):
                for i in range(0, self.Nt+1, step):
                    if wframe:
                        ax.collections.remove(wframe)
                    wframe = ax.plot_trisurf(self.xp, self.yp, SpaceTimeVector_Lag[:,i], linewidth=0.2, \
                                             antialiased=True, cmap=cmap, **kwargs)
                    ax.set_zlabel('time=' + np.array2string(self.tspan[i]) + '/' + np.array2string(self.tspan[-1]) ) 
                    self.writer.grab_frame()
                    plt.pause(.001)
        # Do not save
        else:
            for i in range(0, self.Nt+1, step):
                if wframe:
                    ax.collections.remove(wframe)
                wframe = ax.plot_trisurf(self.xp, self.yp, SpaceTimeVector_Lag[:,i], linewidth=0.2, \
                                         antialiased=True, cmap=cmap, **kwargs)
                ax.set_zlabel('time=' + np.array2string(self.tspan[i]) + '/' + np.array2string(self.tspan[-1]) ) 
                plt.pause(.001)
    
    
    ## Method.
    #  @param self The object pointer.            
    
    def Plot_Hamiltonian(self, tspan, Ham, figsize=(8,6.5), **kwargs):
        """
        Plot the Hamiltonian function versus time 
        """
        plt.figure(figsize=figsize)
        plt.plot(tspan, Ham, **kwargs)
        plt.xlabel('Time $(s)$')                                                                   
        plt.ylabel('Hamiltonian')
        plt.title('Hamiltonian')
        plt.grid(True)
        if not(self.notebook):
            plt.show()
        plt.savefig("Hamiltonian.png")    
        
        
    ## Method.
    #  @param self The object pointer.    
      
    def Plot_Hamiltonian_Relative_Error(self, tspan, Ham, figsize=(8,6.5), **kwargs):
        """
        Plot the relative error related to the Hamiltonian versus time 
        """
        
        plt.figure(figsize=figsize)
        plt.semilogy(tspan, 100 * np.abs(Ham-Ham[0])/Ham[0], **kwargs)
        plt.xlabel('Time $(s)$')
        plt.ylabel('Error (%)')
        plt.title('Hamiltonian Relative Error')
        plt.grid(True)
        
        if not(self.notebook):
            plt.show()
        plt.savefig("Hamiltonian_Relative_Error.png") 
        
    ## Method.
    #  @param self The object pointer.
    
    def Plot_Mesh(self, figure=True, figsize=(8,6.5), **kwargs):
        """
        Plot the two-dimensional mesh with the FEniCS plot method
        """
        
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"
       
        if figure : plt.figure(figsize=figsize)
        plot(self.Msh, **kwargs)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
        plt.title("Mesh with " + "Nv=" + str( self.Msh.num_vertices()) + ", hmax="+str(round(self.Msh.hmax(),3)) )
        plt.savefig("Mesh.png")

    ## Method.
    #  @param self The object pointer.

    def Plot_Mesh_with_DOFs(self, figure=True, figsize=(8,6.5), **kwargs):
        """
        Plot the two-dimensional mesh with the FEniCS plot method including DOFs
        """
        assert self.generate_mesh == 1, "The finite element mesh must be generated first"

        if figure : plt.figure(figsize=figsize)
        plot(self.Msh, **kwargs)
        plt.plot(self.xp, self.yp, 'o', label='Dof of $p$ variables')
        plt.plot(self.xq, self.yq, '^', label='Dof of $q$ variables')
        plt.plot(self.xb, self.yb, 'ko', label='Dof of $\partial$ variables')
        plt.title('Mesh with associated DOFs, $Nq=$'+ str(self.Nq)+ ', $Np=$'+ str(self.Np)+ ', $N_\partial=$'+ str(self.Nb) + ', $Nv=$' + str( self.Msh.num_vertices()) )
        plt.legend()
        if not(self.notebook):
            plt.show()
        plt.savefig("Mesh_with_DOFs.png")
    
    # Method.
    # @param self The object pointer.
    def Plot_3D(self, time_space_Vector, t, title):
        """
        Create a 3D plot at a specific time t
        """
        # Find the index of the nearest time with t
        index_list   = np.where(abs(self.tspan-t)==abs(self.tspan-t).min())[0]
        i            = index_list[0]
        
        fig = plt.figure()
        ax  = fig.gca(projection='3d')
        ax.plot_trisurf(self.xp, self.yp, time_space_Vector[:,i], linewidth=0.2, antialiased=True, cmap=plt.cm.CMRmap)
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('y coordinate')
        ax.set_zlabel('time=' + np.array2string(self.tspan[i]) + '/' + np.array2string(self.tspan[-1]))
        ax.set_title(title)
        if not(self.notebook):
         plt.show()
        plt.savefig("Space_Time_plot.png")
    
    
    ## Method.
    #  @param self The object pointer.
    def Set_Video_Writer(self, fps=128, dpi=256):
        """
        Set video writer options
        """
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata     = dict(title='Movie Test', artist='Matplotlib',
                            comment='Movie support!')
        self.writer  = FFMpegWriter(fps=fps, metadata=metadata)
        self.dpi     = dpi
        
    
    ## Method.
    # @param self The object pointer.
    def Spectrum(self, k=10, xl=None, figsize=(8,6.5), method='sparse', **kwargs):
        """
        Compute and plot the k leading eigenvalues of (J_R) x = \lambda M x
        """
        if method == 'sparse' :
            JR       = csr_matrix(self.J - self.R)
            M_sparse = csr_matrix(self.M)  
#            try :   
#                JR = bmat( [ [JR, csc_matrix(self.B_Dext)], [-csc_matrix(self.B_Dext).T, None] ] )
#        #            M_sparse = block_diag((M_sparse, csc_matrix((self.Nb_D,self.Nb_D))))
#                M_sparse = csc_matrix(block_diag((M_sparse, csc_matrix(self.Mb_D) )))
#            except :    pass
            
            eigen_values, self.eigenv_vectors = eigs(JR, M=M_sparse, k=k)
        
        if method == 'dense' :
            eigen_values = linalg.eigvals(self.J - self.R, self.M)
            k            = self.Nsys

        self.eigen_values_Real, self.eigen_values_Imag = eigen_values.real, eigen_values.imag

        #
        # Plot the partial spectrum
        #
        plt.figure(figsize=figsize)
        plt.plot(self.eigen_values_Real, self.eigen_values_Imag, '+', **kwargs)
        plt.xlabel('Real part')
        plt.ylabel('Imaginary part')
        plt.title('eigs($J-R$, $M$) with '+str(k)+' leading eigenvalues')
        plt.xlim(xl)
        plt.savefig('spectrum.png')
        
        return self.eigen_values_Real, self.eigen_values_Imag

    ## Method.
    # @param self The object pointer.
    def Trisurf(self, SpaceTimeVector_Lag, instance=0, title='', save=False, figsize=(8,6.5), **kwargs):
        """
        ?
        """
        instance = int(np.where(self.tspan == instance)[0])
        fig      = plt.figure(figsize=figsize)
        ax       = fig.gca(projection='3d')
        #fig.colorbar(wfram, shrink=0.5, aspect=5)
        ax.set_xlabel('x coordinate')
        ax.set_ylabel('y coordinate')
        plt.title(title)
        ax.set_xlim(self.x0, self.xL)
        ax.set_ylim(self.y0, self.yL)
        ax.set_zlim(np.min(SpaceTimeVector_Lag), np.max(SpaceTimeVector_Lag))
        ax.plot_trisurf(self.xp, self.yp, SpaceTimeVector_Lag[:,instance], linewidth=0.2, \
                                         antialiased=True, cmap=plt.cm.CMRmap)#, **kwargs)
        if save: fig.save(title+'png')  

          
    #%% UTILITY
    
    ## Method
    #  @param self The object pointer.
    def my_mult(self, A, B):
        """
        Matrix multiplication.

        Multiplies A and B together via the "dot" method.

        Parameters
        ----------
        A : array_like
            first matrix in the product A*B being calculated
        B : array_like
            second matrix in the product A*B being calculated

        Returns
        -------
        array_like
            product of the inputs A and B
        """
        
        if issparse(B) and not issparse(A):
            # dense.dot(sparse) is not available in scipy.
            return B.T.dot(A.T).T
        else:
            return A.dot(B) 