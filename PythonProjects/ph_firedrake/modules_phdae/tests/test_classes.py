import classes_phsystem
import numpy as np

n = 4
E = np.zeros((n, n))
E[[0, 1, 2], [0, 1, 2]] = 1

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
n_ode = n-1

sysDAErig = classes_phsystem.SysPhdaeRig(n, n_lmb, n_rig, n_p, n_q, J=J, E=E, B=B)
# print("M_r rigid DAE:", sysDAErig.M_r)
# print("M_fr rigid DAE:", sysDAErig.M_fr)
# print("B_r rigid DAE:", sysDAErig.B_r)
# print("G rigid DAE:", sysDAErig.G_e)
#


sysODErig = classes_phsystem.SysPhdaeRig(n_ode, 0, n_rig, n_p, n_q, J=J[:n_ode, :n_ode], \
                                         E=E[:n_ode, :n_ode], B=B[:n_ode])
# print("G rigid ODE:",sysODErig.G_e)

sysDAEfl = classes_phsystem.SysPhdaeRig(n, 1, 0, 2, n_q, J=J, E=E, B=B)
# print("M_r flex DAE:", sysDAEfl.M_r)
# print("M_fr flex DAE:", sysDAEfl.M_fr)
# print("B_r flex DAE:", sysDAEfl.B_r)
# print("G flex DAE:", sysDAEfl.G_e)

sysOde, T = sysDAEfl.dae_to_odeE()
print("J converted Ode:", sysOde.J)
print("Q converted Ode:", sysOde.Q)
print("B converted Ode:", sysOde.B)
print("T conversion:", T)
