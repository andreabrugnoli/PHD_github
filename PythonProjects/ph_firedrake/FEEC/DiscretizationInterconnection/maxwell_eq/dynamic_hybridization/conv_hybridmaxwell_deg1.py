from FEEC.DiscretizationInterconnection.maxwell_eq.dynamic_hybridization.hybrid_maxwell import compute_err

DEG=1

geo_case = "3D"
bc_input = "EH"

import numpy as np

save_res = True # input("Save results: ")

# path_project = "~/GitProjects/PHD_github/PythonProjects/ph_firedrake/FEEC/DiscretizationInterconnection/maxwell_eq/"
path_res = "results_hybrid/"
# path_res = os.path.join(path_project, folder_res)
# os.mkdir(path_res)

n_test_deg1 = 5

n_vec_deg1 = np.array([2 ** (i) for i in range(n_test_deg1)])
h_vec_deg1 = 1./n_vec_deg1

# E1H2 Results
E1_err_deg1 = np.zeros((n_test_deg1, 2))
H2_err_deg1 = np.zeros((n_test_deg1, 2))
E1tan_err_deg1 = np.zeros((n_test_deg1, ))
H1nor_err_deg1 = np.zeros((n_test_deg1, ))

H_E1H2_err_deg1 = np.zeros((n_test_deg1,))

order_E1_deg1 = np.zeros((n_test_deg1 - 1, 2))
order_H2_deg1 = np.zeros((n_test_deg1 - 1, 2))

order_E1tan_deg1 = np.zeros((n_test_deg1 - 1, ))
order_H1nor_deg1 = np.zeros((n_test_deg1 - 1, ))

order_H_E1H2_deg1 = np.zeros((n_test_deg1 - 1,))

# E2H1 Results
E2_err_deg1 = np.zeros((n_test_deg1, 2))
H1_err_deg1 = np.zeros((n_test_deg1, 2))
H1tan_err_deg1 = np.zeros((n_test_deg1, ))
E1nor_err_deg1 = np.zeros((n_test_deg1, ))

H_E2H1_err_deg1 = np.zeros((n_test_deg1,))

order_E2_deg1 = np.zeros((n_test_deg1 - 1, 2))
order_H1_deg1 = np.zeros((n_test_deg1 - 1, 2))

order_H1tan_deg1 = np.zeros((n_test_deg1 - 1, ))
order_E1nor_deg1 = np.zeros((n_test_deg1 - 1, ))

order_H_E2H1_deg1 = np.zeros((n_test_deg1 - 1,))

# Dual representation
E12_err_deg1 = np.zeros((n_test_deg1,))
H12_err_deg1 = np.zeros((n_test_deg1,))

order_E12_deg1 = np.zeros((n_test_deg1 - 1,))
order_H12_deg1 = np.zeros((n_test_deg1 - 1,))


for i in range(n_test_deg1):
    res_deg1 = compute_err(n_vec_deg1[i], 100, deg=DEG, bd_cond=bc_input)
    # E1H2 system
    E1_err_deg1[i, :] = res_deg1["err_E1"]
    H2_err_deg1[i, :] = res_deg1["err_H2"]

    E1tan_err_deg1[i] = res_deg1["err_E1tan"]
    H1nor_err_deg1[i] = res_deg1["err_H1nor"]

    # E2H1 system
    E2_err_deg1[i, :] = res_deg1["err_E2"]
    H1_err_deg1[i, :] = res_deg1["err_H1"]

    H1tan_err_deg1[i] = res_deg1["err_H1tan"]
    E1nor_err_deg1[i] = res_deg1["err_E1nor"]

    # Energies
    H_E1H2_err_deg1[i], H_E2H1_err_deg1[i] = res_deg1["err_H"]

    # Dual representation
    E12_err_deg1[i] = res_deg1["err_E12"]
    H12_err_deg1[i] = res_deg1["err_H12"]

    if i>0:
        # Order E1H2
        order_E1_deg1[i - 1, 0] = np.log(E1_err_deg1[i, 0] / E1_err_deg1[i - 1, 0]) / np.log(h_vec_deg1[i] / h_vec_deg1[i - 1])
        order_E1_deg1[i - 1, 1] = np.log(E1_err_deg1[i, 1] / E1_err_deg1[i - 1, 1]) / np.log(h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_H2_deg1[i - 1, 0] = np.log(H2_err_deg1[i, 0] / H2_err_deg1[i - 1, 0]) / np.log(h_vec_deg1[i] / h_vec_deg1[i - 1])
        order_H2_deg1[i - 1, 1] = np.log(H2_err_deg1[i, 1] / H2_err_deg1[i - 1, 1]) / np.log(h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_E1tan_deg1[i - 1] = np.log(E1tan_err_deg1[i] / E1tan_err_deg1[i - 1]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_H1nor_deg1[i - 1] = np.log(H1nor_err_deg1[i] / H1nor_err_deg1[i - 1]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_H_E1H2_deg1[i - 1] = np.log(H_E1H2_err_deg1[i] / H_E1H2_err_deg1[i - 1]) / np.log(h_vec_deg1[i] / h_vec_deg1[i - 1])

        # Order E2H1
        order_E2_deg1[i - 1, 0] = np.log(E2_err_deg1[i, 0] / E2_err_deg1[i - 1, 0]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])
        order_E2_deg1[i - 1, 1] = np.log(E2_err_deg1[i, 1] / E2_err_deg1[i - 1, 1]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_H1_deg1[i - 1, 0] = np.log(H1_err_deg1[i, 0] / H1_err_deg1[i - 1, 0]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])
        order_H1_deg1[i - 1, 1] = np.log(H1_err_deg1[i, 1] / H1_err_deg1[i - 1, 1]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_H1tan_deg1[i - 1] = np.log(H1tan_err_deg1[i] / H1tan_err_deg1[i - 1]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_E1nor_deg1[i - 1] = np.log(E1nor_err_deg1[i] / E1nor_err_deg1[i - 1]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])

        order_H_E2H1_deg1[i - 1] = np.log(H_E2H1_err_deg1[i] / H_E2H1_err_deg1[i - 1]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])

        # Dual representation
        order_E12_deg1[i - 1] = np.log(E12_err_deg1[i] / E12_err_deg1[i - 1]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])
        order_H12_deg1[i - 1] = np.log(H12_err_deg1[i] / H12_err_deg1[i - 1]) / np.log(
            h_vec_deg1[i] / h_vec_deg1[i - 1])


# E1H2 variables order
print("Estimated L2, Hcurl order of convergence for E_1: " + str(order_E1_deg1))
print("Estimated L2, Hdiv order of convergence for H_2: " + str(order_H2_deg1))

print("Estimated L2 order of convergence for H_1nor: " + str(order_H1nor_deg1))
print("Estimated L2 order of convergence for E_1tan: " + str(order_E1tan_deg1))

print("Estimated order of convergence for H_E1H2: " + str(order_H_E1H2_deg1))

# E2H1 variables order
print("Estimated L2, Hdiv order of convergence for E_2: " + str(order_E2_deg1))
print("Estimated L2, Hcurl order of convergence for H_1: " + str(order_H1_deg1))

print("Estimated L2 order of convergence for E_1nor: " + str(order_E1nor_deg1))
print("Estimated L2 order of convergence for H_1tan: " + str(order_H1tan_deg1))

print("Estimated order of convergence for H_E2H1: " + str(order_H_E2H1_deg1))

# Dual representation
print("Estimated L2 order of convergence for E_1 - E_2: " + str(order_E12_deg1))
print("Estimated L2 order of convergence for H_1 - H_2: " + str(order_H12_deg1))


if save_res:
    np.save(path_res + "h_deg1_" + bc_input + "_" + geo_case, h_vec_deg1)
    # Results to save E1H2
    np.save(path_res + "E1_err_deg1_" + bc_input + "_" + geo_case, E1_err_deg1)
    np.save(path_res + "H2_err_deg1_" + bc_input + "_" + geo_case, H2_err_deg1)
    np.save(path_res + "E1tan_err_deg1_" + bc_input + "_" + geo_case, E1tan_err_deg1)
    np.save(path_res + "H1nor_err_deg1_" + bc_input + "_" + geo_case, H1nor_err_deg1)

    np.save(path_res + "H_E1H2_err_deg1_" + bc_input + "_" + geo_case, H_E1H2_err_deg1)

    np.save(path_res + "order_E1_deg1_" + bc_input + "_" + geo_case, order_E1_deg1)
    np.save(path_res + "order_H2_deg1_" + bc_input + "_" + geo_case, order_H2_deg1)
    np.save(path_res + "order_E1tan_deg1_" + bc_input + "_" + geo_case, order_E1tan_deg1)
    np.save(path_res + "order_H1nor_deg1_" + bc_input + "_" + geo_case, order_H1nor_deg1)

    np.save(path_res + "order_H_E1E2_deg1_" + bc_input + "_" + geo_case, order_H_E1H2_deg1)

    # Results to save E2H1
    np.save(path_res + "E2_err_deg1_" + bc_input + "_" + geo_case, E2_err_deg1)
    np.save(path_res + "H1_err_deg1_" + bc_input + "_" + geo_case, H1_err_deg1)
    np.save(path_res + "H1tan_err_deg1_" + bc_input + "_" + geo_case, H1tan_err_deg1)
    np.save(path_res + "E1nor_err_deg1_" + bc_input + "_" + geo_case, E1nor_err_deg1)

    np.save(path_res + "H_E2H1_err_deg1_" + bc_input + "_" + geo_case, H_E2H1_err_deg1)

    np.save(path_res + "order_E2_deg1_" + bc_input + "_" + geo_case, order_E2_deg1)
    np.save(path_res + "order_H1_deg1_" + bc_input + "_" + geo_case, order_H1_deg1)
    np.save(path_res + "order_H1tan_deg1_" + bc_input + "_" + geo_case, order_H1tan_deg1)
    np.save(path_res + "order_E1nor_deg1_" + bc_input + "_" + geo_case, order_E1nor_deg1)

    np.save(path_res + "order_H_E2H1_deg1_" + bc_input + "_" + geo_case, order_H_E2H1_deg1)

    # Dual representation
    np.save(path_res + "E12_err_deg1_" + bc_input + "_" + geo_case, E12_err_deg1)
    np.save(path_res + "H12_err_deg1_" + bc_input + "_" + geo_case, H12_err_deg1)

    np.save(path_res + "order_E12_deg1_" + bc_input + "_" + geo_case, order_E12_deg1)
    np.save(path_res + "order_H12_deg1_" + bc_input + "_" + geo_case, order_H12_deg1)