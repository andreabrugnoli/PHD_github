import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection

from matplotlib import patches

import utilities.plot_setup

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patheffects import Stroke, Normal
from scipy.ndimage import gaussian_filter

m=1
l=1
g=9.81
d=1
c=8

def f(x_1, x_2):
    return np.array([x_2, -g/l*np.sin(x_1) - d/(m*l**2)*x_2])


def V(x_1, x_2):
    return m*g*l*(1-np.cos(x_1)) + 0.5*m*l**2*x_2**2

def dotV(x_1, x_2):
    return -d*x_2**2


x, y = np.meshgrid(np.linspace(-3, 3, 25),
                   np.linspace(-6, 6, 25))

# Directional vectors

u = f(x, y)[0]
v = f(x, y)[1]

V_field = V(x, y)
dotV_field = dotV(x, y)


# Plotting Vector Field with QUIVER
fig, ax = plt.subplots()

qv = ax.quiver(x, y, u, v, color='black')
ct = ax.contour(x, y, V_field, vmin=c)

index_Vc = V_field<=c
index_dotV = dotV_field==0

x_setVc = x[index_Vc]
y_setVc = y[index_Vc]

x_setdotV = x[index_dotV & index_Vc]
y_setdotV = y[index_dotV & index_Vc]


width = max(x_setVc)
height = max(y_setVc)

theta = np.deg2rad(np.arange(0.0, 360.0, 1.0))
x_ell = width * np.cos(theta)
y_ell = height * np.sin(theta)

ax.fill(x_ell, y_ell, alpha=0.2, facecolor='grey',
        edgecolor='black', linewidth=1, label='$\mathcal{C}$')

# ax.plot(x_setVc, y_setVc, 'o', markersize=20, label='$\mathcal{C}$')
# ax.contour(xx_setVc, yy_setVc, Ones, label='$\mathcal{C}$')
ax.plot(x_setdotV, y_setdotV, 'r-', linewidth=4, label='$\mathcal{A}$')
ax.plot(0, 0, 'o', markersize=10, label='$\mathcal{I}$')
# plt.title('Vector Field')

# Setting x, y boundary limits
ax.set_xlim(-3, 3)
ax.set_ylim(-6, 6)

from matplotlib import rcParams
rcParams["legend.loc"] = 'right'

ax.legend()

# Show plot with grid
ax.grid()

path_fig = "/home/andrea/GitProjects/PHD_github/LaTeXProjects/CandidatureWuppertal/imagesDiss/"
plt.savefig(path_fig + "pendulum.svg", format="svg",transparent=True, bbox_inches='tight')
plt.show()
