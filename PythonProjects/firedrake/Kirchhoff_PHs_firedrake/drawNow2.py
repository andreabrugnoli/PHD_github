import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib, time
from firedrake.plot import _two_dimension_triangle_func_val


class plot3dClass( object ):

    def __init__( self, fun0,  minZ, maxZ, X2, Y2, xlabel = None, ylabel = None,  zlabel = None, title = None ):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot( 111, projection='3d' )

        self.X2, self.Y2 = X2, Y2
        tol = 1e-4
        fntsize = 20

        matplotlib.interactive(True)
        if minZ == maxZ:
            raise ValueError('Constant function for drawnow')

        self.ax.set_zlim(minZ - 1e-3 * abs(minZ), maxZ + 1e-3 * abs(maxZ))
        self.ax.w_zaxis.set_major_locator(LinearLocator(10))
        self.ax.w_zaxis.set_major_formatter(FormatStrFormatter('%1.2g'))

        self.ax.set_xlabel(xlabel, fontsize=fntsize)
        self.ax.set_ylabel(ylabel, fontsize=fntsize)
        self.ax.set_zlabel(zlabel, fontsize=fntsize)

        self.ax.set_title(title, fontsize=fntsize)

        triangulation, Z = _two_dimension_triangle_func_val(fun0, 10)
        self.tri_surf = self.ax.plot_trisurf( \
            triangulation, Z, cmap=cm.jet, linewidth=0, antialiased=False)

        Z2 = np.zeros( self.X2.shape )
        self.tri_line = self.ax.plot(self.X2, self.Y2, Z2, color='black')
        # plt.draw() maybe you want to see this frame?

    def drawNow2( self, fun, Z2, z2label=None ):
        self.tri_surf.remove()
        self.ax.lines.pop(0)
        triangulation, Z = _two_dimension_triangle_func_val(fun, 10)
        self.tri_surf = self.ax.plot_trisurf( \
            triangulation, Z, cmap=cm.jet, linewidth=0, antialiased=False)
        self.tri_line = self.ax.plot( self.X2, self.Y2, Z2, label=z2label, color='black')
        self.tri_surf._facecolors2d = self.tri_surf._facecolors3d
        self.tri_surf._edgecolors2d = self.tri_surf._edgecolors3d
        self.ax.legend(handles=[self.tri_line[0]])
        plt.draw()                    # redraw the canvas
        self.fig.canvas.flush_events()
        time.sleep(0.1)


