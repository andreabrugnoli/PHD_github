import numpy as np


def init_results(n_t):
    H_vec = np.zeros((1 + n_t,))
    errH_vec = np.zeros((1 + n_t,))
    bdflow_mid_vec = np.zeros((n_t,))

    errL2_var1_vec = np.zeros((1 + n_t,))
    errL2_var2_vec = np.zeros((1 + n_t,))
    errL2_varnor_vec = np.zeros((n_t,))
    errL2_vartan_vec = np.zeros((1 + n_t,))

    errSob_var1 = np.zeros((1 + n_t,))
    errSob_var2 = np.zeros((1 + n_t,))

    dict_res = {"H_vec": H_vec, "errH_vec": errH_vec, "bdflow_mid_vec": bdflow_mid_vec, \
                "err_var1_vec": [errL2_var1_vec, errSob_var1], "err_var2_vec": [errL2_var2_vec, errSob_var2], \
                "errL2_varnor_vec": errL2_varnor_vec, "errL2_vartan_vec": errL2_vartan_vec}

