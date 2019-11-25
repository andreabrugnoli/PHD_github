from firedrake import *
import numpy as np
import scipy as sp

import scipy.linalg as la
import scipy.sparse as spa
import scipy.sparse.linalg as sp_la

np.set_printoptions(threshold=np.inf)
from math import pi, floor

import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['text.usetex'] = True


def verify(n, r):

    E = Constant(1)
    nu = Constant(0.3)

    rho = Constant(1)
    k = Constant(5/6)
    h = Constant(1)

    D = E * h ** 3 / (1 - nu ** 2) / 12
    fl_rot = 12 / (E * h ** 3)

    G = E / 2 / (1 + nu)
    F = G * h * k

    # Operators and functions

    def bending_mom(kappa):
        momenta = D * ((1-nu)*kappa + nu * Identity(2) * tr(kappa))
        return momenta

    def bending_curv(momenta):
        kappa = fl_rot * ((1+nu)*momenta - nu * Identity(2) * tr(momenta))
        return kappa


    # The unit square mesh is divided in :math:`N\times N` quadrilaterals::

    Lx, Ly = 1, 1
    mesh = RectangleMesh(n, n, Lx, Lx, quadrilateral=False)

    # plot(mesh);
    # plt.show()


    # Finite element defition

    V_pw = FunctionSpace(mesh, "CG", r)
    V_pth = VectorFunctionSpace(mesh, "CG", r)


    dx = Measure('dx')
    ds = Measure('ds')

    x, y = SpatialCoordinate(mesh)

    w_st = 1/3*x**3*(x-1)**3*y**3*(y-1)**3 \
           - 2 *h**2/(5*(1-nu))*\
             (y**3*(y-1)**3*x*(x-1)*(5*x**2-5*x+1)
             +x**3*(x-1)**3*y*(y-1)*(5*y**2-5*y+1))

    thx_st = y ** 3 * (y - 1) ** 3 * x ** 2 * (x - 1) ** 2 * (2 * x - 1)
    thy_st = x ** 3 * (x - 1) ** 3 * y ** 2 * (y - 1) ** 2 * (2 * y - 1)

    th_st = as_vector([thx_st, thy_st])

    f_st = E/(12*(1-nu**2))*(12*y*(y-1)*(5*x**2-5*x+1)*(2*y**2*(y-1)**2+x*(x-1)*(5*y**2-5*y+1)) + \
                             12*x*(x-1)*(5*y**2-5*y+1)*(2*x**2*(x-1)**2+y*(y-1)*(5*x**2-5*x+1)))


    wst_ex = Function(V_pw)
    wst_ex.assign(interpolate(w_st, V_pw))

    thst_ex = Function(V_pth)
    thst_ex.assign(interpolate(th_st, V_pth))

    f_ex = Function(V_pw)
    f_ex.assign(interpolate(f_st, V_pw))

    # q_ex = Function(V_pth)
    # q_ex.assign(interpolate(F*(grad(wst_ex) - thst_ex), V_pth))

    q_ex = F*(grad(wst_ex) - thst_ex)

    eq_1 = div(q_ex) + f_ex

    kappa_ex = sym(grad(thst_ex))
    mom_ex = bending_mom(kappa_ex)


    eq_2 = div(mom_ex) + q_ex

    err_eq1 = assemble( eq_1**2*dx )
    err_eq2 = assemble( dot(eq_2, eq_2)*dx )

    return err_eq1, err_eq2

r = 2
n_test = 10
n_vec = np.array([2**(i) for i in range(n_test)])

err_eq1 = np.zeros((n_test))
err_eq2 = np.zeros((n_test))

for i in range(n_test):
    err_eq1[i], err_eq2[i] = verify(n_vec[i], r)


print('Error eq 1')
for i in range(n_test):
    print('n='+str(n_vec[i]) + ' '+ str(err_eq1[i]))

print('Error eq 2')
for i in range(n_test):
    print('n=' + str(n_vec[i]) + ' ' + str(err_eq2[i]))

