import classes_phdae
import numpy as np

n = 4
E = np.eye(n)
E[-1, -1] = 0

J = np.array([[0,   0, 0, 1],
              [0,   0, 1, 1],
              [0,  -1, 0, 0],
              [-1, -1, 0, 0]])
B = np.array([1, 1, 0, 0])

n_rig = 1
n_p = 1
n_q = 1
n_e = n_p + n_q
n_lmb = 1

sysDAErig = classes_phdae.SysPhdaePfemRig(n, n_lmb, n_rig, n_p, n_q, J=J, E=E, B=B)
n_ode = n-1
sysODErig = classes_phdae.SysPhdaePfemRig(n_ode, 0, n_rig, n_p, n_q, J=J[:n_ode, :n_ode], \
                                          E=E[:n_ode, :n_ode], B=B[:n_ode])

sysDAEfl = classes_phdae.SysPhdaePfemRig(n, 1, 0, 2, n_q, J=J, E=E, B=B)

