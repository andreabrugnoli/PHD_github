from Elasticity2D_AFW import Elasticity2DConfig, construct_system
from firedrake import *
import matplotlib.pyplot as plt
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
from tools_plotting import setup

instance_El2D_case1 = Elasticity2DConfig(n_el=5, deg_FE=2)
E_1, J_1, B_1 = construct_system(instance_El2D_case1)


A_pH = csr_matrix(spsolve(E_1.tocsc(), J_1.tocsc())).todense()
B_pH = csr_matrix(spsolve(E_1.tocsc(), B_1)).todense()
m = np.shape(B_pH)[1]

C_pH = B_1.T
D_pH = np.zeros((m, m))
print(A_pH.shape, B_pH.shape, C_pH.shape, D_pH.shape)

# sys_pH = control.ss(A_pH, B_pH[:, 0], C_pH[0, :], D_pH[0,0])
# plt.figure()
# w, mag, phase = control.bode(sys_pH)
# plt.show()