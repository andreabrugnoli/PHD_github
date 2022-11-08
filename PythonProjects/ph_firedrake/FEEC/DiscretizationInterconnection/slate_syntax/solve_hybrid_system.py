from firedrake import *


def solve_hybrid(a_form, b_form, bcs, V_gl, V_loc):

    n_block_loc = V_loc.num_sub_spaces()
    _A = Tensor(a_form)
    _F = Tensor(b_form)
    # Extracting blocks for Slate expression of the reduced system
    A = _A.blocks
    F = _F.blocks
    S = A[n_block_loc, n_block_loc] - A[n_block_loc, :n_block_loc] * A[:n_block_loc, :n_block_loc].inv * A[:n_block_loc, n_block_loc]
    E = F[n_block_loc] - A[n_block_loc, :n_block_loc] * A[:n_block_loc, :n_block_loc].inv * F[:n_block_loc]

    # Assemble and solve: SΛ = E
    Smat = assemble(S, bcs=bcs)
    Evec = assemble(E)
    lambda_h = Function(V_gl)
    solve(Smat, lambda_h, Evec, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})

    x_h = Function(V_loc)  # Function to store the result: x_loc

    # Intermediate expressions
    Lambda = AssembledVector(lambda_h)  # Local coefficient vector for Λ
    # Local solve expressions
    x_sys = A[:n_block_loc, :n_block_loc].solve(F[:n_block_loc] - A[:n_block_loc, n_block_loc] * Lambda, decomposition="PartialPivLU")
    assemble(x_sys, x_h)

    sol = Function(V_loc * V_gl)
    for ii in range(n_block_loc):
        sol.sub(ii).assign(x_h.sub(ii))
    sol.sub(n_block_loc).assign(lambda_h)

    return sol



def solve_hybrid_2constr(a_form, b_form, bcs, W0, W1, W0_nor, V0_tan):
    """Specific function for the hybridization
           """
    n_dyn = 2
    _A = Tensor(a_form)
    _F = Tensor(b_form)
    # Extracting blocks for Slate expression of the reduced system
    A = _A.blocks
    F = _F.blocks

    Adyn = A[:n_dyn, :n_dyn]
    Bpar = - A[:n_dyn, n_dyn]
    Rgam = A[n_dyn, 3]

    Aloc_mul = Bpar.T * Adyn.inv * Bpar
    Agl_mul = Rgam.T * Aloc_mul.inv * Rgam
    Fgl_mul = F[3] - Rgam.T * Aloc_mul.inv * Bpar.T * Adyn.inv * F[:n_dyn]   \
              + Rgam.T * Aloc_mul.inv * F[2] # Should not be present

    # Assemble and solve for the global multiplier
    Amat_gl = assemble(Agl_mul, bcs=bcs)
    Fvec_gl = assemble(Fgl_mul)
    lam_gl_h = Function(V0_tan)
    solve(Amat_gl, lam_gl_h, Fvec_gl, solver_parameters={"ksp_type": "preonly"})


    # Local lagrange multiplier
    lam_loc_h = Function(W0_nor)
    # Intermediate expressions
    Lam_gl = AssembledVector(lam_gl_h)  # Local coefficient vector for Λ
    # Local solve expressions
    lam_loc_sys = Aloc_mul.solve(F[2] -Bpar.T * Adyn.inv * F[:n_dyn] - Rgam * Lam_gl)
    assemble(lam_loc_sys, lam_loc_h)

    # Local state variable
    x_h = Function(W0*W1)
    # Intermediate expressions
    Lam_loc = AssembledVector(lam_loc_h)  # Local coefficient vector for Λ
    # Local solve expressions
    x_loc_sys = Adyn.solve(F[:n_dyn] + Bpar * Lam_loc)

    assemble(x_loc_sys, x_h)

    p_h, u_h = x_h.split()

    return p_h, u_h, lam_loc_h, lam_gl_h