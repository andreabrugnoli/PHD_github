from firedrake import *


def spaces01(mesh, deg):
    P0 = FiniteElement("CG", mesh.ufl_cell(), deg)
    P0f = FacetElement(P0)
    P0_b = BrokenElement(P0)
    P0f_b = BrokenElement(P0f)

    # Careful with freezing of simulation for variant integral
    # P1 = FiniteElement("N1curl", mesh.ufl_cell(), deg, variant="integral")
    P1 = FiniteElement("N1curl", mesh.ufl_cell(), deg)
    P1_b = BrokenElement(P1)

    V0 = FunctionSpace(mesh, P0)
    V1 = FunctionSpace(mesh, P1)

    W0 = FunctionSpace(mesh, P0_b)
    W1 = FunctionSpace(mesh, P1_b)

    W0_nor = FunctionSpace(mesh, P0f_b)

    W01_loc = W0 * W1 * W0_nor
    V0_tan = FunctionSpace(mesh, P0f)
    V01 = V0 * V1

    return W01_loc, V0_tan, V01


def spaces32(mesh, deg):
    P3 = FiniteElement("DG", mesh.ufl_cell(), deg-1)
    # Careful with freezing of simulation for variant integral
    P2 = FiniteElement("RT", mesh.ufl_cell(), deg)
    P2f = FacetElement(P2)
    P2_b = BrokenElement(P2)
    P2f_b = BrokenElement(P2f)


    V3 = FunctionSpace(mesh, P3)
    V2 = FunctionSpace(mesh, P2)

    W3 = FunctionSpace(mesh, P3)
    W2 = FunctionSpace(mesh, P2_b)

    W2_nor = FunctionSpace(mesh, P2f_b)

    W32_loc = W3 * W2 * W2_nor

    V2_tan = FunctionSpace(mesh, P2f)
    V32 = V3 * V2

    return W32_loc, V2_tan, V32

def assign_exact01(p_ex, u_ex, state01, W01_loc, V01_gl, V01):
    pex_0 = project(interpolate(p_ex, V01.sub(0)), W01_loc.sub(0))
    uex_1 = project(interpolate(u_ex, V01.sub(1)), W01_loc.sub(1))
    u_ex_Pnor = project_uex_W0nor(u_ex, W01_loc.sub(2))

    # The Lagrange multiplier is computed at half steps
    state01.sub(0).assign(pex_0)
    state01.sub(1).assign(uex_1)
    # For lambda nor the resolution of a linear system is required
    state01.sub(2).assign(u_ex_Pnor)
    state01.sub(3).assign(interpolate(p_ex, V01_gl))


def assign_exact32(p_ex, u_ex, state32, W32_loc, V32_gl, V32):
    pex_3 = project(interpolate(p_ex, V32.sub(0)), W32_loc.sub(0))
    uex_2 = project(interpolate(u_ex, V32.sub(1)), W32_loc.sub(1))
    p_ex_Pnor = project_pex_W2nor(p_ex, W32_loc.sub(2))

    # The Lagrange multiplier is computed at half steps
    state32.sub(0).assign(pex_3)
    state32.sub(1).assign(uex_2)
    # For lambda nor the resolution of a linear system is required
    state32.sub(2).assign(p_ex_Pnor)
    state32.sub(3).assign(interpolate(u_ex, V32_gl))


def m_form01(v_1, u_1, v_0, p_0):
    m_form = inner(v_1, u_1) * dx + inner(v_0, p_0) * dx

    return m_form


def j_form01(v_1, u_1, v_0, p_0):
    j_form = dot(v_1, grad(p_0)) * dx - dot(grad(v_0), u_1) * dx

    return j_form


def constr_loc01(v_0, p_0, v_0_nor, u_0_nor):
    form_W0_Wnor = v_0 * u_0_nor
    form_Wnor_W0 = v_0_nor * p_0

    form = (form_W0_Wnor('+') + form_W0_Wnor('-')) * dS + form_W0_Wnor * ds \
           - ((form_Wnor_W0('+') + form_Wnor_W0('-')) * dS + form_Wnor_W0 * ds)
    return form


def constr_global01(v_0_nor, u_0_nor, v_0_tan, p_0_tan):
    form_Wnor_Vtan = v_0_nor * p_0_tan
    form_Vtan_Wnor = v_0_tan * u_0_nor

    form = (form_Wnor_Vtan('+') + form_Wnor_Vtan('-')) * dS + form_Wnor_Vtan * ds \
           - ((form_Vtan_Wnor('+') + form_Vtan_Wnor('-')) * dS + form_Vtan_Wnor * ds)
    return form


def neumann_flow0(v_0_tan, neumann_bc):
    return v_0_tan * neumann_bc * ds


def m_form32(v_3, p_3, v_2, u_2):
    m_form = inner(v_3, p_3) * dx + inner(v_2, u_2) * dx

    return m_form


def j_form32(v_3, p_3, v_2, u_2):
    j_form = dot(v_3, div(u_2)) * dx - dot(div(v_2), p_3) * dx

    return j_form


def constr_loc32(v_2, u_2, v_2_nor, p_2_nor, n_ver):

    form_W2_Wnor = inner(v_2, n_ver) * inner(p_2_nor, n_ver)
    form_Wnor_W2 = inner(v_2_nor, n_ver) * inner(u_2, n_ver)


    form = (form_W2_Wnor('+') + form_W2_Wnor('-')) * dS + form_W2_Wnor * ds \
           - ((form_Wnor_W2('+') + form_Wnor_W2('-')) * dS + form_Wnor_W2 * ds)
    return form


def constr_global32(v_2_nor, p_2_nor, v_2_tan, u_2_tan, n_ver):

    form_Wnor_Vtan = inner(v_2_nor, n_ver) * inner(u_2_tan, n_ver)
    form_Vtan_Wnor = inner(v_2_tan, n_ver) * inner(p_2_nor, n_ver)

    form = (form_Wnor_Vtan('+') + form_Wnor_Vtan('-')) * dS + form_Wnor_Vtan * ds \
           - ((form_Vtan_Wnor('+') + form_Vtan_Wnor('-')) * dS + form_Vtan_Wnor * ds)
    return form

def dirichlet_flow2(v_2_tan, dirichlet_bc, n_ver):

    return dot(v_2_tan, n_ver) * dirichlet_bc * ds

def project_uex_W0nor(u_ex, W0_nor):
    # project normal trace of u_e onto Vnor
    unor = TrialFunction(W0_nor)
    wtan = TestFunction(W0_nor)

    mesh = W0_nor.mesh()
    n_ver = FacetNormal(mesh)


    a_form = inner(wtan, unor)
    a = (a_form('+') + a_form('-')) * dS + a_form * ds

    L_form = inner(wtan, dot(u_ex, n_ver))
    L = (L_form('+') + L_form('-')) * dS + L_form * ds

    A = Tensor(a)
    b = Tensor(L)
    exsol_Pnor = assemble(A.inv * b)

    return exsol_Pnor


def project_pex_W2nor(p_ex, W2_nor):
    # project normal trace of u_e onto Vnor
    pnor = TrialFunction(W2_nor)
    wtan = TestFunction(W2_nor)

    mesh = W2_nor.mesh()
    n_ver = FacetNormal(mesh)


    a_form = inner(wtan, n_ver)*inner(pnor, n_ver)
    a = (a_form('+') + a_form('-')) * dS + a_form * ds

    L_form = inner(wtan, n_ver)*p_ex
    L = (L_form('+') + L_form('-')) * dS + L_form * ds

    A = Tensor(a)
    b = Tensor(L)
    exsol_Pnor = assemble(A.inv * b)

    return exsol_Pnor