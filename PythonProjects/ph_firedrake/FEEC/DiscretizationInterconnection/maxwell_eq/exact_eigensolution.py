from firedrake import *
import numpy as np

def exact_sol_maxwell3D(mesh, t, t_1):
    x, y, z = SpatialCoordinate(mesh)

    om_x = 1
    om_y = 1
    om_z = 1

    om_t = np.sqrt(om_x ** 2 + om_y ** 2 + om_z ** 2)
    phi_x = 0
    phi_y = 0
    phi_z = 0


    ft = sin(om_t * t) / om_t
    dft = cos(om_t * t)

    ft_1 = sin(om_t * t_1) / om_t
    dft_1 = cos(om_t * t_1)

    g_x = -cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_z * z + phi_z)
    g_y = Constant(0.0)
    g_z = sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * cos(om_z * z + phi_z)

    g_fun = as_vector([g_x,
                       g_y,
                       g_z])

    curl_g = as_vector([om_y * sin(om_x * x + phi_x) * cos(om_y * y + phi_y) * cos(om_z * z + phi_z),
                        -(om_x + om_z) * cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * cos(om_z * z + phi_z),
                        om_y * cos(om_x * x + phi_x) * cos(om_y * y + phi_y) * sin(om_z * z + phi_z)])
    # curl_g = curl(g_fun)

    E_ex = g_fun * dft
    E_ex_1 = g_fun * dft_1

    H_ex = -curl_g * ft
    H_ex_1 = -curl_g * ft_1

    return E_ex, H_ex, E_ex_1, H_ex_1

