from firedrake import *
import numpy as np
import pandas as pd

def feech_solution(f, mesh, r):

    n = FacetNormal(mesh)

    S_element = FiniteElement('N1curl', mesh.ufl_cell(), r + 1)
    V_element = FiniteElement('RT', mesh.ufl_cell(), r + 1)

    S = FunctionSpace(mesh, BrokenElement(S_element))
    V = FunctionSpace(mesh, BrokenElement(V_element))
    Vnor = FunctionSpace(mesh, BrokenElement(S_element[facet]))
    Rnor = FunctionSpace(mesh, BrokenElement(V_element[facet]))
    Stan = FunctionSpace(mesh, S_element[facet])
    Vtan = FunctionSpace(mesh, V_element[facet])

    W_local = S * V * Vnor * Rnor
    W_global = Stan * Vtan
    W = W_local * W_global

    sigma, u, unor, rhonor, sigmatan, utan = TrialFunctions(W)
    tau, v, vnor, etanor, tautan, vtan = TestFunctions(W)

    a_interior = (-inner(sigma, tau) + inner(u, curl(tau))
                  + inner(curl(sigma), v) + div(u)*div(v))
    a_boundary = (inner(cross(sigmatan - sigma, n), cross(vnor, n))
                  + inner(utan - u, n)*inner(etanor, n)
                  + inner(cross(unor, n), cross(tautan - tau, n))
                  + inner(rhonor, n)*inner(vtan - v, n))
    a = a_interior*dx + (a_boundary('+') + a_boundary('-'))*dS + a_boundary*ds

    L = inner(f, v)*dx        

    # static condensation with Slate
    A = Tensor(a).blocks
    b = Tensor(L).blocks
    
    A_sc = A[4:6, 4:6] - A[4:6, :4] * A[:4, :4].inv * A[:4, 4:6]
    b_sc = b[4:6] - A[4:6, :4] * A[:4, :4].inv * b[:4]
    
    A_sc_mat = assemble(A_sc, mat_type='aij')
    b_sc_vec = assemble(b_sc)

    # solve condensed system using LU decomposition
    params = {
        'ksp_type': 'preonly',
        'pc_type': 'lu',
        'pc_factor_mat_solver_type': 'mumps'
    }

    w_global = Function(W_global)
    solve(A_sc_mat, w_global, b_sc_vec, solver_parameters=params)

    # recover local variables
    w_local = assemble(A[:4, :4].inv *
                       (b[:4] - A[:4, 4:6] * AssembledVector(w_global)))

    return w_local.split() + w_global.split()

def pp_solution(f, sigmatan, utan, r):

    mesh = utan.function_space().mesh()
    n = FacetNormal(mesh)

    r_pp = r

    R_element = FiniteElement('CG', mesh.ufl_cell(), r_pp + 1)
    V_element = FiniteElement('N1curl', mesh.ufl_cell(), r_pp + 1)

    R = FunctionSpace(mesh, BrokenElement(R_element))
    V = FunctionSpace(mesh, BrokenElement(V_element))
    W = R * V

    rho, u = TrialFunctions(W)
    eta, v = TestFunctions(W)

    a = (-rho*eta - inner(u, grad(eta))
         - inner(grad(rho), v) + inner(curl(u), curl(v)))*dx
    L_interior = inner(f, v)
    L_boundary = -inner(utan, eta*n) - inner(sigmatan, cross(v, n))
    L = L_interior*dx + (L_boundary('+') + L_boundary('-'))*dS + L_boundary*ds

    # solve this problem using a Schur complement, since even the
    # local postprocessing matrices can be quite large for high degree
    A = Tensor(a).blocks
    b = Tensor(L).blocks

    A_sc = A[1, 1] - A[1, 0] * A[0, 0].inv * A[0, 1]
    b_sc = b[1] - A[1, 0] * A[0, 0].inv * b[0]

    u = assemble(A_sc.inv * b_sc)
    rho = assemble(A[0, 0].inv * (b[0] - A[0, 1] * AssembledVector(u)))

    return rho, u
    
def sigmatan_error(sigma_e, sigmatan):

    mesh = sigmatan.function_space().mesh()
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)
    err = h*cross(sigma_e - sigmatan, n)**2

    return sqrt(assemble((err('+') + err('-'))*dS + err*ds))    

def utan_error(u_e, utan):

    mesh = utan.function_space().mesh()
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)
    err = h*inner(u_e - utan, n)**2

    return sqrt(assemble((err('+') + err('-'))*dS + err*ds))    

def unor_error(u_e, unor):

    Vnor = unor.function_space()
    
    mesh = Vnor.mesh()
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)
    
    # project normal trace of u_e onto Vnor
    unor_p = TrialFunction(Vnor)
    vnor = TestFunction(Vnor)

    a_form = inner(cross(unor_p, n), cross(vnor, n))
    a = (a_form('+') + a_form('-'))*dS + a_form*ds

    L_form = inner(cross(cross(u_e, n), n), cross(vnor, n))
    L = (L_form('+') + L_form('-'))*dS + L_form*ds

    A = Tensor(a)
    b = Tensor(L)
    unor_p = assemble(A.inv * b)

    err = h*cross(unor_p - unor, n)**2
    return sqrt(assemble((err('+') + err('-'))*dS + err*ds))

def rhonor_error(rho_e, rhonor):

    Rnor = rhonor.function_space()
    
    mesh = Rnor.mesh()
    h = CellDiameter(mesh)
    n = FacetNormal(mesh)

    # project normal trace of rho_e onto Rnor
    rhonor_p = TrialFunction(Rnor)
    etanor = TestFunction(Rnor)

    a_form = inner(rhonor_p, n)*inner(etanor, n)
    a = (a_form('+') + a_form('-'))*dS + a_form*ds

    L_form = rho_e*inner(etanor, n)
    L = (L_form('+') + L_form('-'))*dS + L_form*ds

    A = Tensor(a)
    b = Tensor(L)
    rhonor_p = assemble(A.inv * b)

    err = h*inner(rhonor_p - rhonor, n)**2
    return sqrt(assemble((err('+') + err('-'))*dS + err*ds))

def feech_error(N, r):

    mesh = UnitCubeMesh(N, N, N)
    x, y, z = SpatialCoordinate(mesh)

    # exact solution
    u_e_divfree = as_vector([sin(pi*y)*sin(pi*z),
                             sin(pi*x)*sin(pi*z),
                             sin(pi*x)*sin(pi*y)])
    u_e_curlfree = as_vector([cos(pi*x)*sin(pi*y)*sin(pi*z),
                              sin(pi*x)*cos(pi*y)*sin(pi*z),
                              sin(pi*x)*sin(pi*y)*cos(pi*z)])
    u_e = u_e_divfree + u_e_curlfree

    sigma_e = curl(u_e)
    rho_e = div(u_e)
    f = curl(sigma_e) - grad(rho_e)

    sigma, u, unor, rhonor, sigmatan, utan = feech_solution(f, mesh, r)
    rho_pp, u_pp = pp_solution(f, sigmatan, utan, r)

    error_dict = {
        'sigma_error': norm(sigma_e - sigma),
        'sigmatan_error': sigmatan_error(sigma_e, sigmatan),
        'sigma_pp_error': norm(sigma_e - curl(u_pp)),
        'u_error': norm(u_e - u),
        'utan_error': utan_error(u_e, utan),
        'unor_error': unor_error(u_e, unor),
        'u_pp_error': norm(u_e - u_pp),
        'rho_error': norm(rho_e - div(u)),
        'rhonor_error': rhonor_error(rho_e, rhonor),
        'rho_pp_error': norm(rho_e - rho_pp),
        'delrho_error': norm(grad(rho_e - div(u))),
        'delrho_pp_error': norm(grad(rho_e - rho_pp)),
    }

    return error_dict

def convergence_df(log2N_min, log2N_max, r):

    # get list of error dictionaries and store in pandas DataFrame
    Ns = [2**k for k in range(log2N_min, log2N_max + 1)]
    df = pd.DataFrame([feech_error(N, r) for N in Ns], index=Ns)
    df.index.name = 'N'

    # compute convergence rates
    df[df.columns.str.replace('error', 'rate')] = -df.apply(np.log2).diff()

    return df

def make_convergence_csv(log2N_min, log2N_max, r):
    df = convergence_df(log2N_min, log2N_max, r)
    df.to_csv('n=3_k=2_r=' + str(r) + '.csv', na_rep='---')
