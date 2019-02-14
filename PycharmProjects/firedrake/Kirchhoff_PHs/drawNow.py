import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib, time
from firedrake.plot import _two_dimension_triangle_func_val


class plot3dClass( object ):

    def __init__( self, fun0,  minZ, maxZ, xlabel = None, ylabel = None,  zlabel = None, title = None ):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot( 111, projection='3d' )
        tol = 1e-4
        fntsize = 20
        
        matplotlib.interactive(True)
        self.ax.set_zlabel(zlabel, fontsize=fntsize)

        if minZ == maxZ:
            raise ValueError('Constant function for drawnow')

        self.ax.set_xlabel(xlabel, fontsize=fntsize)
        self.ax.set_ylabel(ylabel, fontsize=fntsize)

        self.ax.set_zlim(minZ-1e-3*abs(minZ) , maxZ+1e-3*abs(maxZ))
        self.ax.w_zaxis.set_major_locator( LinearLocator( 10 ) )
        self.ax.w_zaxis.set_major_formatter( FormatStrFormatter( '%1.2g' ) )

        self.ax.set_title(title, fontsize=fntsize)

        triangulation, Z = _two_dimension_triangle_func_val(fun0, 10)
        self.tri_surf = self.ax.plot_trisurf( \
            triangulation, Z, cmap=cm.jet, linewidth=0, antialiased=False )
        # plt.draw() maybe you want to see this frame?

    def drawNow( self, fun):
        self.tri_surf.remove()
        triangulation, Z = _two_dimension_triangle_func_val(fun, 10)
        self.tri_surf = self.ax.plot_trisurf( \
            triangulation, Z, cmap=cm.jet, linewidth=0, antialiased=False)
        self.tri_surf._facecolors2d = self.tri_surf._facecolors3d
        self.tri_surf._edgecolors2d = self.tri_surf._edgecolors3d
        plt.draw()                    # redraw the canvas
        self.fig.canvas.flush_events()
        time.sleep(0.01)


