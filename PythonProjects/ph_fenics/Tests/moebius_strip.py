import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from tools_plotting import setup

fig = plt.figure()
ax = plt.axes(projection='3d')

# Number of times you'd like your strip to twist (only even numbers allowed!)
num_twists = 2
# The width of the strip
width = 0.5

# Mesh parameters
# Number of nodes along the length of the strip
nl = 300
# Number of nodes along the width of the strip (>= 2)
nw = 14

# Generate suitable ranges for parameters
u_range = np.arange(nl, dtype='d')/(nl)*2*np.pi
v_range = np.arange(nw, dtype='d')/(nw - 1.0)*width


# Create a list to store the vertices
x = np.zeros((nl*nw, ))
y = np.zeros((nl*nw, ))
z = np.zeros((nl*nw, ))
# Populate the list of vertices
j = 0
for u in u_range:
    for v in v_range:
        x[j] = np.cos(u) + v * np.cos(num_twists * u / 2.0) * np.cos(u)
        y[j] = np.sin(u) + v * np.cos(num_twists * u / 2.0) * np.sin(u)
        z[j] = v * np.sin(num_twists * u / 2.0)

        j = j + 1

# triangulate in the underlying parametrization
triangles = np.zeros((nl*(nw - 1)*2, 3))

# Populate the list of cells
k = 0
for i in range(nl - 1):
    for j in range(nw - 1):
        triangles[k] =   [i*nw + j, (i + 1)*nw + j + 1, i*nw + j + 1]
        triangles[k+1] = [i*nw + j, (i + 1)*nw + j    , (i + 1)*nw + j + 1]
        k = k + 2
# Close the geometry
for j in range(nw - 1):
    triangles[k] = [(nl - 1)*nw + j, j + 1, (nl - 1)*nw + j + 1]
    triangles[k+1] = [(nl - 1)*nw + j, j    , j + 1]
    k = k + 2

ax = plt.axes(projection='3d')
ax.plot_trisurf(x, y, z, triangles=triangles, edgecolor = 'grey', linewidths=0.2)

# ax.set_xlim(-1, 1); ax.set_ylim(-1, 1);

plt.show()