import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True

import mpmath
mpmath.mp.dps = 15; mpmath.mp.pretty = True

delta = 1
xi = 1

gam1 = lambda s: (s/2*((1 + delta + s) + ((1+delta+s)**2-4*s)**(1/2)))**(1/2)
gam2 = lambda s: (s/2*((1 + delta + s) - ((1+delta+s)**2-4*s)**(1/2)))**(1/2)
phi = lambda s: 1/s

n_t = 100
t_vec = np.linspace(0.00001, 4, n_t)
theta_s = lambda s: phi(s)/(gam1(s)**2-gam2(s)**2) *\
                  ((gam1(s)**2-s**2)*mpmath.exp(-gam1(s)*xi) - (gam2(s)**2-s**2)*mpmath.exp(-gam2(s)*xi))

disp_s = lambda s: -phi(s)/(gam1(s)**2-gam2(s)**2)*(gam1(s)*mpmath.exp(-gam1(s)*xi) - gam2(s)*mpmath.exp(-gam2(s)*xi))
stress_s = lambda s: s**2*phi(s)/(gam1(s)**2-gam2(s)**2)*(mpmath.exp(-gam1(s)*xi) - mpmath.exp(-gam2(s)*xi))

theta_t = np.empty((n_t, ))
disp_t = np.empty((n_t, ))
stress_t = np.empty((n_t, ))
for i in range(n_t):
    theta_t[i] = mpmath.invertlaplace(theta_s, t_vec[i], method='dehoog')
    disp_t[i] = mpmath.invertlaplace(disp_s, t_vec[i], method='dehoog')
    stress_t[i] = mpmath.invertlaplace(stress_s, t_vec[i], method='dehoog')


plt.figure()
plt.plot(t_vec, theta_t, 'r', label='uncoupled')

plt.xlabel('Dimensionless Time')
plt.ylabel('Dimensionless Temperature')

plt.figure()
plt.plot(t_vec, disp_t, 'r', label='uncoupled')

plt.xlabel('Dimensionless Time')
plt.ylabel('Dimensionless Displacement')

plt.figure()
plt.plot(t_vec, stress_t, 'r', label='uncoupled')

plt.xlabel('Dimensionless Time')
plt.ylabel('Dimensionless Stress')

plt.show()