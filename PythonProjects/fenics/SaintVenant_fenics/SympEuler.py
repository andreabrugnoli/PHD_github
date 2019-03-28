import numpy as np
from math import ceil, floor

from scipy import linalg as la
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import inv as inv_sp


def SympEuler(funcdyn, p0, q0, t_f, t_0= 0.0, dt = 1e-6, n_ev = 100):
    n_Vp = len(p0)
    n_Vq = len(q0)
    p_sol = np.zeros((n_Vp, n_ev))
    q_sol = np.zeros((n_Vq, n_ev))

    p_sol[:, 0] = p0
    q_sol[:, 0] = q0

    n_t = floor((t_f - t_0) / dt)

    if n_ev > n_t:
        raise ValueError("Choose less evaluation points")

    t_ev = np.linspace(t_0, t_f, n_ev)


    p_old = p0
    q_old = q0

    k = 1
    
    for i in range(1, n_t + 1):
        t = t_0 + i * dt

        # Intergation for p (n+1)
        
        #al_q_.vector()[:] = q_old
        #e_q = assemble(e_q_).get_local()
        #e_q = M_q @ q_old * rho*g

        dpdt, dqdt, Ainv = funcdyn(t,p_old,q_old)
        p_new = p_old + dt * dpdt
        #al_p_.vector()[:] = p_new
        #e_p = assemble(e_p_).get_local()
        #e_p = M_p @ p_new / rho
        dpdt, dqdt, Ainv = funcdyn(t,p_new,q_old)
        
        q_new =  Ainv @ (q_old + dt * dqdt )
        
        p_old = p_new
        q_old = q_new            

        if t >= t_ev[k]:
            p_sol[:, k] = p_new
            q_sol[:, k] = q_new
            print(str(int(k / (n_ev-1) * 100)), '% of the solution computed ')
            k = k + 1
            # print('Solution number ' + str(k) + ' computed')



    return t_ev, p_sol, q_sol
