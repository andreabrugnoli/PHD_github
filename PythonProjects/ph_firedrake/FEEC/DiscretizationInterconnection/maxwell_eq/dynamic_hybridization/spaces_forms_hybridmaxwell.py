from firedrake import *


def spacesE1H2(mesh, deg):
    P1 = FiniteElement("N1curl", mesh.ufl_cell(), deg)
    P1f = P1[facet]
    P1_b = BrokenElement(P1)
    P1f_b = BrokenElement(P1f)

    # Careful with freezing of simulation for variant integral
    # P1 = FiniteElement("N1curl", mesh.ufl_cell(), deg, variant="integral")
    P2 = FiniteElement("RT", mesh.ufl_cell(), deg)
    P2_b = BrokenElement(P2)

    V1 = FunctionSpace(mesh, P1)
    V2 = FunctionSpace(mesh, P2)

    W1 = FunctionSpace(mesh, P1_b)
    W2 = FunctionSpace(mesh, P2_b)

    W1_nor = FunctionSpace(mesh, P1f_b)

    W12_loc = W1 * W2 * W1_nor
    V1_tan = FunctionSpace(mesh, P1f)
    V12 = V1 * V2

    V1W2 = V1 * W2

    return W12_loc, V1_tan, V12, V1W2

def spacesE2H1(mesh, deg):
    P2 = FiniteElement("RT", mesh.ufl_cell(), deg)
    P2_b = BrokenElement(P2)

    # Careful with freezing of simulation for variant integral
    P1 = FiniteElement("N1curl", mesh.ufl_cell(), deg)
    P1f = P1[facet]
    P1_b = BrokenElement(P1)
    P1f_b = BrokenElement(P1f)

    V2 = FunctionSpace(mesh, P2)
    V1 = FunctionSpace(mesh, P1)

    W2 = FunctionSpace(mesh, P2_b)
    W1 = FunctionSpace(mesh, P1_b)

    W1_nor = FunctionSpace(mesh, P1f_b)

    W21_loc = W2 * W1 * W1_nor
    V1_tan = FunctionSpace(mesh, P1f)
    V21 = V2 * V1

    W2V1 = W2 * V1


    return W21_loc, V1_tan, V21, W2V1

def assign_exactE1H2(E_ex, H_ex, state12, W12_loc, V12_gl, V12):
    Eex_1 = project(interpolate(E_ex, V12.sub(0)), W12_loc.sub(0))
    Hex_2 = project(interpolate(H_ex, V12.sub(1)), W12_loc.sub(1))
    H_ex_Pnor = project_ex_W1nor(H_ex, W12_loc.sub(2))

    # The Lagrange multiplier is computed at half steps
    state12.sub(0).assign(Eex_1)
    state12.sub(1).assign(Hex_2)
    # For lambda nor the resolution of a linear system is required
    state12.sub(2).assign(H_ex_Pnor)
    state12.sub(3).assign(interpolate(E_ex, V12_gl))


def assign_exactE2H1(E_ex, H_ex, state21, W21_loc, V21_gl, V21):
    Eex_2 = project(interpolate(E_ex, V21.sub(0)), W21_loc.sub(0))
    Hex_1 = project(interpolate(H_ex, V21.sub(1)), W21_loc.sub(1))
    E_ex_Pnor = project_ex_W1nor(E_ex, W21_loc.sub(2))

    # The Lagrange multiplier is computed at half steps
    state21.sub(0).assign(Eex_2)
    state21.sub(1).assign(Hex_1)
    # For lambda nor the resolution of a linear system is required
    state21.sub(2).assign(E_ex_Pnor)
    state21.sub(3).assign(interpolate(H_ex, V21_gl))

def m_formE1H2(v_1, E_1, v_2, H_2):
    m_form = inner(v_1, E_1) * dx + inner(v_2, H_2) * dx

    return m_form


def j_formE1H2(v_1, E_1, v_2, H_2):
    j_form = dot(curl(v_1), H_2) * dx - dot(v_2, curl(E_1)) * dx

    return j_form


def constr_locE1H2(v_1, E_1, v_1_nor, H_1_nor, n_ver):
    form_W1_Wnor = -inner(cross(v_1, n_ver), cross(H_1_nor, n_ver))
    form_Wnor_W1 = -inner(cross(v_1_nor, n_ver), cross(E_1, n_ver))

    form = (form_W1_Wnor('+') + form_W1_Wnor('-')) * dS + form_W1_Wnor * ds \
           - ((form_Wnor_W1('+') + form_Wnor_W1('-')) * dS + form_Wnor_W1 * ds)
    return form


def constr_globalE1H2(v_1_nor, H_1_nor, v_1_tan, E_1_tan, n_ver):
    form_Wnor_Vtan = -inner(cross(v_1_nor, n_ver), cross(E_1_tan, n_ver))
    form_Vtan_Wnor = -inner(cross(v_1_tan, n_ver), cross(H_1_nor, n_ver))

    form = (form_Wnor_Vtan('+') + form_Wnor_Vtan('-')) * dS + form_Wnor_Vtan * ds \
           - ((form_Vtan_Wnor('+') + form_Vtan_Wnor('-')) * dS + form_Vtan_Wnor * ds)
    return form


def bdflowE1H2(v_1_tan, H_1, n_ver):
    return -dot(cross(v_1_tan, H_1), n_ver) * ds


def m_formE2H1(v_2, E_2, v_1, H_1):
    m_form = inner(v_2, E_2) * dx + inner(v_1, H_1) * dx

    return m_form


def j_formE2H1(v_2, E_2, v_1, H_1):
    j_form = dot(v_2, curl(H_1)) * dx - dot(curl(v_1), E_2) * dx

    return j_form


def constr_locE2H1(v_1, H_1, v_1_nor, E_1_nor, n_ver):
    form_W1_Wnor = inner(cross(v_1, n_ver), cross(E_1_nor, n_ver))
    form_Wnor_W1 = inner(cross(v_1_nor, n_ver), cross(H_1, n_ver))

    form = (form_W1_Wnor('+') + form_W1_Wnor('-')) * dS + form_W1_Wnor * ds \
           - ((form_Wnor_W1('+') + form_Wnor_W1('-')) * dS + form_Wnor_W1 * ds)
    return form


def constr_globalE2H1(v_1_nor, E_1_nor, v_1_tan, H_1_tan, n_ver):
    form_Wnor_Vtan = inner(cross(v_1_nor, n_ver), cross(H_1_tan, n_ver))
    form_Vtan_Wnor = inner(cross(v_1_tan, n_ver), cross(E_1_nor, n_ver))

    form = (form_Wnor_Vtan('+') + form_Wnor_Vtan('-')) * dS + form_Wnor_Vtan * ds \
           - ((form_Vtan_Wnor('+') + form_Vtan_Wnor('-')) * dS + form_Vtan_Wnor * ds)
    return form


def bdflowE2H1(v_1, E_1, n_ver):
    b_form = dot(cross(v_1, E_1), n_ver) * ds

    return b_form


def project_ex_W1nor(var_ex, W1_nor):
    # project normal trace of u_e onto Vnor
    var_nor = TrialFunction(W1_nor)
    wtan = TestFunction(W1_nor)

    mesh = W1_nor.mesh()
    n_ver = FacetNormal(mesh)

    a_form = inner(cross(wtan, n_ver), cross(var_nor, n_ver))
    a = (a_form('+') + a_form('-')) * dS + a_form * ds

    L_form = inner(cross(wtan, n_ver), cross(cross(var_ex, n_ver), n_ver))
    L = (L_form('+') + L_form('-')) * dS + L_form * ds

    A = Tensor(a)
    b = Tensor(L)
    exsol_Pnor = assemble(A.inv * b)

    return exsol_Pnor