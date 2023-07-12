import numpy as np
import matplotlib.pyplot as plt

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
rcParams['text.usetex'] = True
rcParams['text.latex.preamble']=r"\usepackage{amsmath}\usepackage{bm}"
rcParams["legend.loc"] = 'best'

# Define the vector field function
def vector_field(x, y):
    return np.array([y, -x])


lim_x, lim_y = 3, 3
# Define the range of x and y values
x_vf = np.linspace(-lim_x, lim_x, 20)
y_vf = np.linspace(-lim_y, lim_y, 20)

# Create a meshgrid of x and y values
X_vf, Y_vf = np.meshgrid(x_vf, y_vf)

# Calculate the vector field at each point in the meshgrid
U_vf, V_vf = vector_field(X_vf, Y_vf)

# Plot the vector field using quiver
plt.quiver(X_vf, Y_vf, U_vf, V_vf)

# Create an array of angles from 0 to 2*pi
theta = np.linspace(0, 2*np.pi, 100)

# Calculate the x and y coordinates of the circle
radius = 2
x_circle = radius * np.cos(theta)
y_circle = radius * np.sin(theta)

# Plot the circle
plt.plot(x_circle, y_circle, 'b', label="$\partial\Omega$")

# Set the aspect ratio of the plot to 'equal' for a better representation
plt.xlim((-lim_x, lim_x))
plt.ylim((-lim_x, lim_x))
plt.axis('equal')

# Add labels and title
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title(r'$\bm{u} = [-y, x], \quad \omega = 2$')
plt.text(0,0, "$\Omega$", fontsize=30, ha="center", va="center")
plt.legend()

plt.savefig("vorticity_vector_field.pdf", format="pdf")
# Show the plot
plt.show()
