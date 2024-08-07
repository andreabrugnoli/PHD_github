import os


os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from firedrake import *
import  matplotlib.pyplot as plt

n_el = 50
L = 1/2
mesh = RectangleMesh(n_el, n_el, 1, 0.5, quadrilateral=False)
n_ver = FacetNormal(mesh)


dx = Measure('dx')
ds = Measure('ds')

x, y = SpatialCoordinate(mesh)

t = Constant(0.0)
om_x = 1
om_y = 1

om_t = np.sqrt(om_x ** 2 + om_y ** 2)
phi_x = 0
phi_y = 0
phi_t = 0

# w_ex = sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_t * t + phi_t)
#
# p_ex = om_t * sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * cos(om_t * t + phi_t)
# u_ex = as_vector([om_x * cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_t * t + phi_t),
#                   om_y * sin(om_x * x + phi_x) * cos(om_y * y + phi_y) * sin(om_t * t + phi_t)])

ft = 2*sin(om_t * t + phi_t) + 3*cos(om_t * t + phi_t)
dft_t = om_t * (2*cos(om_t * t + phi_t) - 3*sin(om_t * t + phi_t)) #diff(dft_t, t)

gxy = cos(om_x * x + phi_x) * sin(om_y * y + phi_y)

grad_gxy = grad(gxy)
dgxy_x = - om_x * sin(om_x * x + phi_x) * sin(om_y * y + phi_y)
dgxy_y = om_y * cos(om_x * x + phi_x) * cos(om_y * y + phi_y)

w_ex = gxy * ft

p_ex = gxy * dft_t
u_ex = as_vector([dgxy_x * ft,
                  dgxy_y * ft]) # grad(gxy)

# bdflow_ex_n1 = dot(u_ex, n_ver) * ds(domain=mesh)
# bdflow_ex_n2 = p_ex * ds(domain=mesh)
bdflow_ex_n3 = p_ex * dot(u_ex, n_ver) * ds(domain=mesh)

H_ex_n1 = 0.5 * (p_ex**2 * dx(domain=mesh) + dot(u_ex, u_ex) * dx(domain=mesh))
H_ex_n2 = 0.5 * (diff(w_ex, t)**2 * dx(domain=mesh) + dot(grad(w_ex), grad(w_ex)) * dx(domain=mesh))


a1 = (0.5 - np.sin(0.5)*np.cos(0.5))*(1 + np.sin(1)*np.cos(1))
a2 = ((0.5 + np.sin(0.5) * np.cos(0.5)) * (1 + np.sin(1) * np.cos(1)) + \
                                  (0.5 - np.sin(0.5)*np.cos(0.5))*(1 - np.sin(1)*np.cos(1)))

ft_2 = lambda tau: 2*np.sin(om_t * tau + phi_t) + 3*np.cos(om_t * tau + phi_t)
dft_t_2 = lambda tau: om_t*(2*np.cos(om_t * tau + phi_t) - 3*np.sin(om_t * tau + phi_t))
ddft_tt_2 = lambda tau: -om_t**2*(2*np.sin(om_t * tau + phi_t) + 3*np.cos(om_t * tau + phi_t))

H_ex_an1 = 1/8*(dft_t)**2*a1 + 1/8 * (ft) ** 2 * a2
dH_ex_an1_t = diff(H_ex_an1, t)

H_ex_an2 = lambda tau: 1/8*(dft_t_2(tau))**2*a1 + 1/8 * (ft_2(tau)) ** 2 * a2
dH_ex_an2_t = lambda tau: 1/4*dft_t(tau)*ddft_tt_2(tau)*a1 + 1/4 * ft(tau) * dft_t(tau) * a2

Dt = 0.01
n_t = 100
bd_flow1_vec = np.zeros((n_t,))
bd_flow2_vec = np.zeros((n_t,))
bd_flow3_vec = np.zeros((n_t,))


H_n1_vec = np.zeros((n_t,))
H_n2_vec = np.zeros((n_t,))
H_ex_an1_vec = np.zeros((n_t,))
H_ex_an2_vec = np.zeros((n_t,))
dHex_t1_vec = np.zeros((n_t,))
dHex_t2_vec = np.zeros((n_t,))

t_vec = np.arange(0, Dt*n_t, Dt)

for ii, t_n in enumerate(t_vec):
    t.assign(float(t_n))
    # print(float(t), t_n)

    # bd_flow1_vec[ii] = assemble(bdflow_ex_n1)
    # bd_flow2_vec[ii] = assemble(bdflow_ex_n2)
    bd_flow3_vec[ii] = assemble(bdflow_ex_n3)
    dHex_t1_vec[ii] = dH_ex_an1_t
    dHex_t2_vec[ii] = dH_ex_an2_t(t_n)

    H_n1_vec[ii] = assemble(H_ex_n1)
    H_n2_vec[ii] = assemble(H_ex_n2)
    H_ex_an1_vec[ii] = H_ex_an1
    H_ex_an2_vec[ii] = H_ex_an2(t_n)



plt.figure()
# plt.plot(t_vec, bd_flow1_vec, 'r-', label=r'N flow')
# plt.plot(t_vec, bd_flow2_vec, 'b--', label=r'D flow')
plt.plot(t_vec, bd_flow3_vec, '*-', label=r'Bd flow')
plt.plot(t_vec, dHex_t1_vec, '+-.', label=r'dH_t1')
# plt.plot(t_vec, dHex_t2_vec, '+-', label=r'dH_t2')
plt.xlabel(r'Time [s]')
plt.title(r'Boundary flows')
plt.legend()


plt.figure()
plt.plot(t_vec, H_n1_vec, 'y--', label=r'Energy PH')
plt.plot(t_vec, H_n2_vec, 'g-', label=r'Energy classical')
plt.plot(t_vec, H_ex_an1_vec, 'r--', label=r'Energy exact1')
plt.plot(t_vec, dHex_t1_vec, '+-.', label=r'dH_t1')
plt.plot(t_vec, H_ex_an2_vec, 'b-.', label=r'Energy exact2')
plt.plot(t_vec, dHex_t2_vec, '--', label=r'dH_t2')

plt.xlabel(r'Time [s]')
plt.title(r'Hamiltonian')
plt.legend()

plt.show()
