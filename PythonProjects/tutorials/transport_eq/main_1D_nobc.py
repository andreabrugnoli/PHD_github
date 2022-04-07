from FD_1D_nobc import mesh, u_0, exp_time_amont_space, ex_solution,animate_sol
import numpy as np

if __name__ == '__main__':

    t_end = 10
    Dt = t_end/100
    n_t = int(t_end / Dt + 1)
    t_vec = np.linspace(0, t_end, n_t)

    c = float(input("Enter physical velocity  : "))
    c_num = float(input("Enter numerical velocity  : "))
    L_0 = 10
    L_domain = abs(c)*t_end + L_0


    if c_num<=0:
        print("Error negative numerical velocity")
        exit()

    sigma = c / c_num
    print("sigma : " + str(sigma))
    Dx = c_num*Dt

    if c > 0:
        x_min = -L_0
        x_max = L_domain + x_min
    else:
        x_max = L_0
        x_min = x_max-L_domain

    n_x = int(np.floor(L_domain/Dx)+1)

    x_vec = np.linspace(x_min, x_max, n_x)

    # x_grid, t_grid = mesh(x_vec, t_vec)

    u_sol_num = exp_time_amont_space(u_0, sigma, x_vec, t_vec)

    anim = animate_sol(x_vec, t_vec, u_sol_num, c, sigma, save_anim=True)







