import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Make data
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)

x, y = np.meshgrid(x, y)

dx_thx_st = 2 * (y - 1) ** 3 * y ** 3 * (x - 1) * x * (5 * x ** 2 - 5 * x + 1)
dy_thx_st = 3 * (x - 1) ** 2 * x ** 2 * (2 * x - 1) * (y - 1) ** 2 * y ** 2 * (2 * y - 1)

dx_thy_st = 3 * (y - 1) ** 2 * y ** 2 * (2 * y - 1) * (x - 1) ** 2 * x ** 2 * (2 * x - 1)
dy_thy_st = 2 * (x - 1) ** 3 * x ** 3 * (y - 1) * y * (5 * y ** 2 - 5 * y + 1)

z1 = dx_thy_st
z2 = dy_thx_st
z = z1-z2
ax.plot_surface(x, y, z, linewidth=0.2, antialiased=True)
plt.show()