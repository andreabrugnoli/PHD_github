import numpy as np
from numpy.linalg import solve
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
from scikits import umfpack
from scipy.sparse import csc_matrix, csr_matrix

def theta_method(M, J, R, B_f, x_0, theta=0.5, t_f=0.1, dt=1e-6, n_ev=1000):
    
    assert theta>0
    t_ev = np.linspace(0, t_f, n_ev)
    
#    JR = J-R
#    A_J = M - dt*theta*J
#    A_JR = M - dt*theta*(JR)
#    
#    B_J = M + dt * (1 - theta) * J
#    B_JR = M + dt * (1 - theta) * JR
#    
#    Sys_J = solve(A_J, B_J)
#    Sys_JR = solve(A_JR, B_JR)
#    
#    Sys_BJ = solve(A_J, dt*B_f)
#    Sys_BJR = solve(A_JR, dt*B_f)
    
    n_sys = len(M)
    
    M_sparse = csc_matrix(M)
    J_sparse = csc_matrix(J)
        
    A_J_sparse = M_sparse - dt*theta*J_sparse
    B_J_sparse = M_sparse + dt * (1 - theta) * J_sparse
    Sys_J = spsolve(A_J_sparse, B_J_sparse)
    Sys_BJ = spsolve(A_J_sparse, dt*B_f)


    JR_sparse = csc_matrix(J - R)
    A_JR_sparse = M_sparse - dt*theta*(JR_sparse)
    B_JR_sparse = M_sparse + dt * (1 - theta) * JR_sparse
    
    Sys_JR = spsolve(A_JR_sparse, B_JR_sparse) 
    Sys_BJR = spsolve(A_JR_sparse, dt*B_f)
        
        
#    else:
#    
#        M_e = M[:-n_lmb, :-n_lmb]
#        
#        J_e = J[:-n_lmb, :-n_lmb]
#        G_D = J[:-n_lmb, -n_lmb:]
#        
#        R_D = R[-n_lmb:, -n_lmb:]
#        
#        B_e = B_f[:-n_lmb]
#        
#        n_sys = len(M_e)
#        
#        Me_sparse = csc_matrix(M_e)
#        Je_sparse = csc_matrix(J_e)
#        GD_sparse = csc_matrix(G_D)
#        RD_sparse = csc_matrix(R_D)
#                
#        Mass_solver = umfpack.UmfpackLU(Me_sparse)
##        RD_solver = umfpack.UmfpackLU(RD_sparse)
#
#        GDMinvGDT      = csc_matrix(GD_sparse.T @ Mass_solver.solve(GD_sparse))
#        GDMinvGDT_solver = umfpack.UmfpackLU(GDMinvGDT)
#
#        Jpr_sparse =  Je_sparse - GD_sparse @ \
#                                 GDMinvGDT_solver.solve(GD_sparse.T @ Mass_solver.solve(Je_sparse))
#        Bpr =  B_e - GD_sparse @ \
#                                GDMinvGDT_solver.solve(GD_sparse.T @ Mass_solver.solve(B_e))
#
#        A_J_sparse = Me_sparse - dt*theta*Jpr_sparse
#        B_J_sparse = Me_sparse + dt * (1 - theta) * Jpr_sparse
#        Sys_J = spsolve(A_J_sparse, B_J_sparse)
#        Sys_BJ = spsolve(A_J_sparse, dt*Bpr)
#        
##        JR_sparse = Je_sparse - GD_sparse @ RD_solver.solve(GD_sparse.T)
##        A_JR_sparse = Me_sparse - dt*theta*(JR_sparse)
##        B_JR_sparse = Me_sparse + dt * (1 - theta) * JR_sparse
##        
##        Sys_JR = spsolve(A_JR_sparse, B_JR_sparse) 
##        Sys_BJR = spsolve(A_JR_sparse, dt*B_pr)
                                
        
    Nt = int(t_f / dt) + 1
    if n_ev > Nt:
        raise ValueError("Choose less evaluation points")
        
    

    X_sol = np.zeros((n_sys, n_ev))
#    X_sol[:, 0] = x_0[:n_sys]
    X_sol[:, 0] = x_0

    X_old = X_sol[:, 0]

    X_new = np.zeros((n_sys,))
    k = 1
    

    for i in range(Nt):
        
        t = dt * (i + 1)
        
        if t<0.2*t_f:
            X_new = Sys_J @ X_old + Sys_BJ          
        else:
            X_new = Sys_JR @ X_old + Sys_BJR            
            
        X_old = X_new
        
        if k < n_ev and t >= t_ev[k]:
            X_sol[:, k] = X_new
            k = k + 1
        elif k == n_ev:
            break
        
    
    return t_ev, X_sol
