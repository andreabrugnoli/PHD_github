from problems.taylor_green2D import TaylorGreen2D
from problems.taylor_green3D import TaylorGreen3D
from solvers.dfNS_palha import compute_sol
import matplotlib.pyplot as plt

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
from math import pi
d = 2 # int(input("Spatial dimension ? "))
#
# param = {"ksp_type": "gmres", "ksp_gmres_restart":100, "pc_type":"ilu"}
# # param = {"ksp_type": "gmres", "pc_type": "ilu", 'pc_hypre_type': 'boomeramg'}

if __name__ == '__main__':
    # 1. Select Problem:
    # Taylor Green 2D

    deg = 2
    n_t = 100
    Delta_t = 1/100
    t_f = n_t * Delta_t
    options = {"n_el": 10, "t_fin": t_f, "n_t": n_t}

    if d == 2:
        problem = TaylorGreen2D(options)
    else:
        problem = TaylorGreen3D(options)
    results = compute_sol(problem, deg, n_t, t_f)

    tvec_int = results["tspan_int"]
    tvec_stag = results["tspan_stag"]

    H_ex = results["energy_ex"]
    H_pr = results["energy_pr"]
    H_dl = results["energy_dl"]

    E_ex = results["enstrophy_ex"]
    E_pr = results["enstrophy_pr"]
    E_dl = results["enstrophy_dl"]

    Hel_ex = results["helicity_ex"]
    Hel_pr = results["helicity_pr"]
    Hel_dl = results["helicity_dl"]

    uP_ex = results["uP_ex"]
    uP_pr = results["uP_pr"]
    uP_dl = results["uP_dl"]

    wP_ex = results["wP_ex"]
    wP_pr = results["wP_pr"]
    wP_dl = results["wP_dl"]

    pdynP_ex = results["pdynP_ex"]
    pdynP_pr = results["pdynP_pr"]
    pdynP_dl = results["pdynP_dl"]

    pP_ex = results["pP_ex"]
    pP_pr = results["pP_pr"]
    pP_dl = results["pP_dl"]

    divu_pr_L2 = results["divu_pr_L2"]
    divu_dl_L2 = results["divu_dl_L2"]

    plt.figure()
    plt.plot(tvec_int, H_pr, 'b', label="H primal")
    plt.plot(tvec_int, H_dl, 'r', label="H dual")
    if problem.exact:
        plt.plot(tvec_int, H_ex, 'g', label="H exact")
    plt.legend()

    plt.figure()
    plt.plot(tvec_int, E_pr, 'b', label="E primal")
    plt.plot(tvec_int, E_dl, 'r', label="E dual")
    if problem.exact:
        plt.plot(tvec_int, E_ex, 'g', label="E exact")
    plt.legend()

    plt.figure()
    plt.plot(tvec_int, uP_pr[:, 0], 'b', label="ux at P primal")
    plt.plot(tvec_int, uP_dl[:, 0], 'r', label="ux at P dual")
    if problem.exact:
        plt.plot(tvec_int, uP_ex[:, 0], 'g', label="ux at P exact")
    plt.legend()

    plt.figure()
    plt.plot(tvec_int, uP_pr[:, 1], 'b', label="uy at P primal")
    plt.plot(tvec_int, uP_dl[:, 1], 'r', label="uy at P dual")
    if problem.exact:
        plt.plot(tvec_int, uP_ex[:, 1], 'g', label="uy at P exact")
    plt.legend()

    plt.figure()
    plt.plot(tvec_int, wP_pr, 'b', label="w at P primal")
    plt.plot(tvec_int, wP_dl, 'r', label="w at P dual")
    if problem.exact:
        plt.plot(tvec_int, wP_ex, 'g', label="w at P exact")
    plt.legend()

    plt.figure()
    plt.plot(tvec_int, pdynP_pr, 'b', label="p dyn at P primal")
    plt.plot(tvec_stag, pdynP_dl, 'r', label="p dyn at P dual")
    if problem.exact:
        plt.plot(tvec_int, pdynP_ex, 'g', label="p dyn at P exact")
    plt.legend()

    plt.figure()
    plt.plot(tvec_int, pP_pr, 'b', label="p at P primal")
    plt.plot(tvec_stag, pP_dl, 'r', label="p at P dual")
    if problem.exact:
        plt.plot(tvec_int, pP_ex, 'g', label="p at P exact")
    plt.legend()

    plt.figure()
    plt.plot(tvec_int, divu_pr_L2, 'b', label="L2 norm div u primal")
    plt.plot(tvec_int, divu_dl_L2, 'r', label="L2 norm div u dual")
    plt.legend()
    plt.show()
