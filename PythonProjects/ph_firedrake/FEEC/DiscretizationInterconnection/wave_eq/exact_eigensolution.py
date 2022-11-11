from firedrake import *

def exact_sol_wave3D(x, y, z, t, t_1):

    om_x = 1
    om_y = 1
    om_z = 1

    om_t = np.sqrt(om_x ** 2 + om_y ** 2 + om_z ** 2)
    phi_x = 0
    phi_y = 0
    phi_z = 0
    phi_t = 0

    ft = 2 * sin(om_t * t + phi_t) + 3 * cos(om_t * t + phi_t)
    dft = om_t * (2 * cos(om_t * t + phi_t) - 3 * sin(om_t * t + phi_t))  # diff(dft_t, t)

    ft_1 = 2 * sin(om_t * t_1 + phi_t) + 3 * cos(om_t * t_1 + phi_t)
    dft_1 = om_t * (2 * cos(om_t * t_1 + phi_t) - 3 * sin(om_t * t_1 + phi_t))  # diff(dft_t, t)

    gxyz = cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_z * z + phi_z)

    dgxyz_x = - om_x * sin(om_x * x + phi_x) * sin(om_y * y + phi_y) * sin(om_z * z + phi_z)
    dgxyz_y = om_y * cos(om_x * x + phi_x) * cos(om_y * y + phi_y) * sin(om_z * z + phi_z)
    dgxyz_z = om_z * cos(om_x * x + phi_x) * sin(om_y * y + phi_y) * cos(om_z * z + phi_z)

    grad_gxyz = as_vector([dgxyz_x,
                           dgxyz_y,
                           dgxyz_z])  # grad(gxyz)

    p_ex = gxyz * dft
    u_ex = grad_gxyz * ft

    p_ex_1 = gxyz * dft_1
    u_ex_1 = grad_gxyz * ft_1

    return p_ex, u_ex, p_ex_1, u_ex_1


def exact_sol_wave2D(x, y, t, t_1):

    om_x = 1
    om_y = 1

    om_t = np.sqrt(om_x ** 2 + om_y ** 2)
    phi_x = 0
    phi_y = 0
    phi_t = 0

    ft = 2 * sin(om_t * t + phi_t) + 3 * cos(om_t * t + phi_t)
    dft = om_t * (2 * cos(om_t * t + phi_t) - 3 * sin(om_t * t + phi_t))  # diff(dft_t, t)

    ft_1 = 2 * sin(om_t * t_1 + phi_t) + 3 * cos(om_t * t_1 + phi_t)
    dft_1 = om_t * (2 * cos(om_t * t_1 + phi_t) - 3 * sin(om_t * t_1 + phi_t))  # diff(dft_t, t)

    gxyz = cos(om_x * x + phi_x) * sin(om_y * y + phi_y)

    dgxyz_x = - om_x * sin(om_x * x + phi_x) * sin(om_y * y + phi_y)
    dgxyz_y = om_y * cos(om_x * x + phi_x) * cos(om_y * y + phi_y)

    grad_gxyz = as_vector([dgxyz_x,
                           dgxyz_y])  # grad(gxyz)

    p_ex = gxyz * dft
    u_ex = grad_gxyz * ft

    p_ex_1 = gxyz * dft_1
    u_ex_1 = grad_gxyz * ft_1

    return p_ex, u_ex, p_ex_1, u_ex_1


def exact_homosol_wave2D(x, y, t, t_1):

    om_x = pi
    om_y = 2*pi

    om_t = np.sqrt(om_x ** 2 + om_y ** 2)
    phi_x = 0
    phi_y = 0
    phi_t = 0

    ft = 2 * sin(om_t * t + phi_t) + 3 * cos(om_t * t + phi_t)
    dft = om_t * (2 * cos(om_t * t + phi_t) - 3 * sin(om_t * t + phi_t))  # diff(dft_t, t)

    ft_1 = 2 * sin(om_t * t_1 + phi_t) + 3 * cos(om_t * t_1 + phi_t)
    dft_1 = om_t * (2 * cos(om_t * t_1 + phi_t) - 3 * sin(om_t * t_1 + phi_t))  # diff(dft_t, t)

    gxyz = sin(om_x * x + phi_x) * sin(om_y * y + phi_y)

    dgxyz_x = om_x * cos(om_x * x + phi_x) * sin(om_y * y + phi_y)
    dgxyz_y = om_y * sin(om_x * x + phi_x) * cos(om_y * y + phi_y)

    grad_gxyz = as_vector([dgxyz_x,
                           dgxyz_y])  # grad(gxyz)

    p_ex = gxyz * dft
    u_ex = grad_gxyz * ft

    p_ex_1 = gxyz * dft_1
    u_ex_1 = grad_gxyz * ft_1

    return p_ex, u_ex, p_ex_1, u_ex_1
