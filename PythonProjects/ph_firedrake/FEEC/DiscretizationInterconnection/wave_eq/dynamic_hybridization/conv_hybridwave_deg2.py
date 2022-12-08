DEG = 2

geo_case = "3D"
bc_input = "DN"

if geo_case == "3D":
    from FEEC.DiscretizationInterconnection.wave_eq.dynamic_hybridization.hybrid_wave3D import compute_err
else:
    from FEEC.DiscretizationInterconnection.wave_eq.dynamic_hybridization.hybrid_wave2D import compute_err
import numpy as np

save_res = True  # input("Save results: ")

# path_project = "~/GitProjects/PHD_github/PythonProjects/ph_firedrake/FEEC/DiscretizationInterconnection/wave_eq/"
path_res = "results_hybrid/"
# path_res = os.path.join(path_project, folder_res)
# os.mkdir(path_res)

n_test_deg2 = 4

n_vec_deg2 = np.array([2 ** (i) for i in range(n_test_deg2)])
h_vec_deg2 = 1. / n_vec_deg2

# 01 Results
u1_err_deg2 = np.zeros((n_test_deg2, 2))
p0_err_deg2 = np.zeros((n_test_deg2, 2))
p0tan_err_deg2 = np.zeros((n_test_deg2,))
u0nor_err_deg2 = np.zeros((n_test_deg2,))

H01_err_deg2 = np.zeros((n_test_deg2,))

order_u1_deg2 = np.zeros((n_test_deg2 - 1, 2))
order_p0_deg2 = np.zeros((n_test_deg2 - 1, 2))

order_p0tan_deg2 = np.zeros((n_test_deg2 - 1,))
order_u0nor_deg2 = np.zeros((n_test_deg2 - 1,))

order_H01_deg2 = np.zeros((n_test_deg2 - 1,))

# 32 Results
u2_err_deg2 = np.zeros((n_test_deg2, 2))
p3_err_deg2 = np.zeros((n_test_deg2,))
u2tan_err_deg2 = np.zeros((n_test_deg2,))
p2nor_err_deg2 = np.zeros((n_test_deg2,))

H32_err_deg2 = np.zeros((n_test_deg2,))

order_u2_deg2 = np.zeros((n_test_deg2 - 1, 2))
order_p3_deg2 = np.zeros((n_test_deg2 - 1,))

order_u2tan_deg2 = np.zeros((n_test_deg2 - 1,))
order_p2nor_deg2 = np.zeros((n_test_deg2 - 1,))

order_H32_deg2 = np.zeros((n_test_deg2 - 1,))

# Dual representation
p30_err_deg2 = np.zeros((n_test_deg2,))
u12_err_deg2 = np.zeros((n_test_deg2,))

order_p30_deg2 = np.zeros((n_test_deg2 - 1,))
order_u12_deg2 = np.zeros((n_test_deg2 - 1,))

for i in range(n_test_deg2):
    res_deg2 = compute_err(n_vec_deg2[i], 100, deg=DEG, bd_cond=bc_input)

    u1_err_deg2[i, :] = res_deg2["err_u1"]
    p0_err_deg2[i, :] = res_deg2["err_p0"]

    p0tan_err_deg2[i] = res_deg2["err_p0tan"]
    u0nor_err_deg2[i] = res_deg2["err_u0nor"]

    u2_err_deg2[i, :] = res_deg2["err_u2"]
    p3_err_deg2[i] = res_deg2["err_p3"]

    u2tan_err_deg2[i] = res_deg2["err_u2tan"]
    p2nor_err_deg2[i] = res_deg2["err_p2nor"]

    H01_err_deg2[i], H32_err_deg2[i] = res_deg2["err_H"]

    p30_err_deg2[i] = res_deg2["err_p30"]
    u12_err_deg2[i] = res_deg2["err_u12"]

    if i > 0:
        # Order 01
        order_u1_deg2[i - 1, 0] = np.log(u1_err_deg2[i, 0] / u1_err_deg2[i - 1, 0]) / np.log(
            h_vec_deg2[i] / h_vec_deg2[i - 1])
        order_u1_deg2[i - 1, 1] = np.log(u1_err_deg2[i, 1] / u1_err_deg2[i - 1, 1]) / np.log(
            h_vec_deg2[i] / h_vec_deg2[i - 1])

        order_p0_deg2[i - 1, 0] = np.log(p0_err_deg2[i, 0] / p0_err_deg2[i - 1, 0]) / np.log(
            h_vec_deg2[i] / h_vec_deg2[i - 1])
        order_p0_deg2[i - 1, 1] = np.log(p0_err_deg2[i, 1] / p0_err_deg2[i - 1, 1]) / np.log(
            h_vec_deg2[i] / h_vec_deg2[i - 1])

        order_p0tan_deg2[i - 1] = np.log(p0tan_err_deg2[i] / p0tan_err_deg2[i - 1]) / np.log(
            h_vec_deg2[i] / h_vec_deg2[i - 1])

        order_u0nor_deg2[i - 1] = np.log(u0nor_err_deg2[i] / u0nor_err_deg2[i - 1]) / np.log(
            h_vec_deg2[i] / h_vec_deg2[i - 1])

        order_H01_deg2[i - 1] = np.log(H01_err_deg2[i] / H01_err_deg2[i - 1]) / np.log(
            h_vec_deg2[i] / h_vec_deg2[i - 1])

        # Order 32
        order_u2_deg2[i - 1, 0] = np.log(u2_err_deg2[i, 0] / u2_err_deg2[i - 1, 0]) / np.log(
            h_vec_deg2[i] / h_vec_deg2[i - 1])
        order_u2_deg2[i - 1, 1] = np.log(u2_err_deg2[i, 1] / u2_err_deg2[i - 1, 1]) / np.log(
            h_vec_deg2[i] / h_vec_deg2[i - 1])

        order_p3_deg2[i - 1] = np.log(p3_err_deg2[i] / p3_err_deg2[i - 1]) / np.log(
            h_vec_deg2[i] / h_vec_deg2[i - 1])

        order_u2tan_deg2[i - 1] = np.log(u2tan_err_deg2[i] / u2tan_err_deg2[i - 1]) / np.log(
            h_vec_deg2[i] / h_vec_deg2[i - 1])

        order_p2nor_deg2[i - 1] = np.log(p2nor_err_deg2[i] / p2nor_err_deg2[i - 1]) / np.log(
            h_vec_deg2[i] / h_vec_deg2[i - 1])

        order_H32_deg2[i - 1] = np.log(H32_err_deg2[i] / H32_err_deg2[i - 1]) / np.log(
            h_vec_deg2[i] / h_vec_deg2[i - 1])

        # Dual representation
        order_p30_deg2[i - 1] = np.log(p30_err_deg2[i] / p30_err_deg2[i - 1]) / np.log(
            h_vec_deg2[i] / h_vec_deg2[i - 1])
        order_u12_deg2[i - 1] = np.log(u12_err_deg2[i] / u12_err_deg2[i - 1]) / np.log(
            h_vec_deg2[i] / h_vec_deg2[i - 1])

# O1 variables order
print("Estimated L2, Hcurl order of convergence for u_1: " + str(order_u1_deg2))
print("Estimated L2, H1 order of convergence for p_0: " + str(order_p0_deg2))

print("Estimated L2 order of convergence for u_0nor: " + str(order_u0nor_deg2))
print("Estimated L2 order of convergence for p_0tan: " + str(order_p0tan_deg2))

print("Estimated order of convergence for H_10: " + str(order_H01_deg2))

# 32 variables order
print("Estimated L2, Hdiv order of convergence for u_2: " + str(order_u2_deg2))
print("Estimated L2 order of convergence for p_3: " + str(order_p3_deg2))

print("Estimated L2 order of convergence for p_2nor: " + str(order_p2nor_deg2))
print("Estimated L2 order of convergence for u_2tan: " + str(order_u2tan_deg2))

print("Estimated order of convergence for H_32: " + str(order_H32_deg2))

# Dual representation
print("Estimated L2 order of convergence for p_0 - p_3: " + str(order_p30_deg2))
print("Estimated L2 order of convergence for u_2 - u_1: " + str(order_u12_deg2))

if save_res:
    np.save(path_res + "h_deg2_" + bc_input + "_" + geo_case, h_vec_deg2)
    # Results to save 01
    np.save(path_res + "u1_err_deg2_" + bc_input + "_" + geo_case, u1_err_deg2)
    np.save(path_res + "p0_err_deg2_" + bc_input + "_" + geo_case, p0_err_deg2)
    np.save(path_res + "p0tan_err_deg2_" + bc_input + "_" + geo_case, p0tan_err_deg2)
    np.save(path_res + "u0nor_err_deg2_" + bc_input + "_" + geo_case, u0nor_err_deg2)

    np.save(path_res + "H01_err_deg2_" + bc_input + "_" + geo_case, H01_err_deg2)

    np.save(path_res + "order_u1_deg2_" + bc_input + "_" + geo_case, order_u1_deg2)
    np.save(path_res + "order_p0_deg2_" + bc_input + "_" + geo_case, order_p0_deg2)
    np.save(path_res + "order_p0tan_deg2_" + bc_input + "_" + geo_case, order_p0tan_deg2)
    np.save(path_res + "order_u0nor_deg2_" + bc_input + "_" + geo_case, order_u0nor_deg2)

    np.save(path_res + "order_H01_deg2_" + bc_input + "_" + geo_case, order_H01_deg2)

    # Results to save 32
    np.save(path_res + "u2_err_deg2_" + bc_input + "_" + geo_case, u2_err_deg2)
    np.save(path_res + "p3_err_deg2_" + bc_input + "_" + geo_case, p3_err_deg2)
    np.save(path_res + "u2tan_err_deg2_" + bc_input + "_" + geo_case, u2tan_err_deg2)
    np.save(path_res + "p2nor_err_deg2_" + bc_input + "_" + geo_case, p2nor_err_deg2)

    np.save(path_res + "H32_err_deg2_" + bc_input + "_" + geo_case, H32_err_deg2)

    np.save(path_res + "order_u2_deg2_" + bc_input + "_" + geo_case, order_u2_deg2)
    np.save(path_res + "order_p3_deg2_" + bc_input + "_" + geo_case, order_p3_deg2)
    np.save(path_res + "order_u2tan_deg2_" + bc_input + "_" + geo_case, order_u2tan_deg2)
    np.save(path_res + "order_p2nor_deg2_" + bc_input + "_" + geo_case, order_p2nor_deg2)

    np.save(path_res + "order_H32_deg2_" + bc_input + "_" + geo_case, order_H32_deg2)

    # Dual representation
    np.save(path_res + "p30_err_deg2_" + bc_input + "_" + geo_case, p30_err_deg2)
    np.save(path_res + "u12_err_deg2_" + bc_input + "_" + geo_case, u12_err_deg2)

    np.save(path_res + "order_p30_deg2_" + bc_input + "_" + geo_case, order_p30_deg2)
    np.save(path_res + "order_u12_deg2_" + bc_input + "_" + geo_case, order_u12_deg2)
