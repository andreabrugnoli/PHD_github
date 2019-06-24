# =============================================================================
# CLOSED PHS SOLUTION with LAGRANGE MULTIPLIER
# =============================================================================
print('Computing closed-lagrange pHs solution: in progress')
# Globlal matrix
Zr = np.zeros((N_lag, N_lag))
Asys = np.vstack((np.hstack((A, C)),
                  np.hstack((C.T, Zr))))
lag_mult = 1
if lag_mult:
    # Defines the residual
    def dae_closed_phs(t, y, yd):
        res_0 = np.zeros(N_T)
        res_1 = np.zeros(N_lag)

        res_0 = M_T @ yd[0:N_T] - A @ y[:N_T] - Bext @ U * time_fun(t) - C @ y[N_T:]
        res_1 = C.T @ y[:N_T] + B_lag @ U * time_fun(t)

        return np.concatenate((res_0, res_1))


    def handle_result(solver, t, y, yd):
        global order
        order.append(solver.get_last_order())

        solver.t_sol.extend([t])
        solver.y_sol.extend([y])
        solver.yd_sol.extend([yd])

        # The initial conditons


    y0 = np.concatenate((Tclosed0, np.zeros(N_lag)))  # Initial conditions
    yd0 = np.zeros(N_T + N_lag)  # Initial conditions

    # Create an Assimulo implicit problem
    imp_mod = Implicit_Problem(dae_closed_phs, y0, yd0, name='dae_closed_pHs')
    imp_mod.handle_result = handle_result

    # Set the algebraic components
    imp_mod.algvar = list(np.concatenate((np.ones(N_T), np.zeros(N_lag))))

    # Create an Assimulo implicit solver (IDA)
    imp_sim = IDA(imp_mod)  # Create a IDA solver

    # Sets the paramters
    imp_sim.atol = 1e-6  # Default 1e-6
    imp_sim.rtol = 1e-6  # Default 1e-6
    imp_sim.suppress_alg = True  # Suppres the algebraic variables on the error test
    imp_sim.report_continuously = True

    # Let Sundials find consistent initial conditions by use of 'IDA_YA_YDP_INIT'
    imp_sim.make_consistent('IDA_YA_YDP_INIT')

    # Simulate
    t, y, yd = imp_sim.simulate(tfinal, 0, tspan)
    Tlag = y[:, :N_T].T
    Tlag_mult = y[:, N_T:].T

    # Moving plot
    enable_moving_plot = 0
    if enable_moving_plot:
        moving_plot(Tlag, coord_T[:, 0], coord_T[:, 1], Lx, Ly, Nt, tspan, step=15, disk=True, radius=radius)