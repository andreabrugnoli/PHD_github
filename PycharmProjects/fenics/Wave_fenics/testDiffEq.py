from diffeqpy import de 

def f(u,p,t):
    x, y, z = u
    sigma, rho, beta = p
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]

u0 = [1.0,0.0,0.0]
tspan = (0., 100.)
p = [10.0,28.0,8/3]
prob = de.ODEProblem(f, u0, tspan, p)
sol = de.solve(prob,saveat=0.01)

import matplotlib.pyplot as plt

plt.plot(sol.t,sol.u)
plt.show()

import numpy

ut = numpy.transpose(sol.u)
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(ut[0,:],ut[1,:],ut[2,:])
plt.show()
