from firedrake import *


def spaces_postprocessing01(mesh, deg):
    P0 = FiniteElement("CG", mesh.ufl_cell(), deg)
    P0_b = BrokenElement(P0)

    # Careful with freezing of simulation for variant integral
    # P1 = FiniteElement("N1curl", mesh.ufl_cell(), deg, variant="integral")
    P1 = FiniteElement("N1curl", mesh.ufl_cell(), deg)
    P1_b = BrokenElement(P1)

    W0 = FunctionSpace(mesh, P0_b)
    W1 = FunctionSpace(mesh, P1_b)

    W01 = W0 * W1

    return W01

def spaces_postprocessing32(mesh, deg):

    P3 = FiniteElement("DG", mesh.ufl_cell(), deg - 1)
    # Careful with freezing of simulation for variant integral
    P2 = FiniteElement("RT", mesh.ufl_cell(), deg)
    P2_b = BrokenElement(P2)

    W3 = FunctionSpace(mesh, P3)
    W2 = FunctionSpace(mesh, P2_b)

    W32 = W3 * W2

    return W32

def assign_exact01_pp(p_ex, u_ex, state01_pp, W01_pp, V01):
    pex_0 = project(interpolate(p_ex, V01.sub(0)), W01_pp.sub(0))
    uex_1 = project(interpolate(u_ex, V01.sub(1)), W01_pp.sub(1))

    # The Lagrange multiplier is computed at half steps
    state01_pp.sub(0).assign(pex_0)
    state01_pp.sub(1).assign(uex_1)


def assign_exact32_pp(p_ex, u_ex, state32_pp, W32_pp, V32):
    pex_3 = project(interpolate(p_ex, V32.sub(0)), W32_pp.sub(0))
    uex_2 = project(interpolate(u_ex, V32.sub(1)), W32_pp.sub(1))

    # The Lagrange multiplier is computed at half steps
    state32_pp.sub(0).assign(pex_3)
    state32_pp.sub(1).assign(uex_2)



def neumann_flow0_pp(v_0_pp, neumann_bc, n_ver):
    form_bd = v_0_pp * dot(neumann_bc, n_ver)
    form = (form_bd('+') + form_bd('-')) * dS + form_bd * ds
    return form

def dirichlet_flow2_pp(v_2_pp, dirichlet_bc, n_ver):
    form_bd = dot(v_2_pp, n_ver) * dirichlet_bc
    form = (form_bd('+') + form_bd('-')) * dS + form_bd * ds
    return form
