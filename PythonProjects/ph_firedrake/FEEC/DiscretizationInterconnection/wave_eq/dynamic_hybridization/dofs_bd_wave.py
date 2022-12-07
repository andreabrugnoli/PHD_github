from firedrake import *

def dofs_ess_nat(bcs_ess, W_loc, V_gl):
    dofs_ess = []

    for ii in range(len(bcs_ess)):
        nodes_ess = bcs_ess[ii].nodes

        dofs_ess = dofs_ess + list(nodes_ess)

    dofs_ess = list(set(dofs_ess))
    dofs_nat = list(set(V_gl.boundary_nodes("on_boundary")).difference(set(dofs_ess)))

    dofsV_gl_ess = W_loc.dim() + np.array(dofs_ess)
    dofsV_gl_nat = W_loc.dim() + np.array(dofs_nat)

    # dofsV10_gl_NoD = W_loc.dim() + np.array(list(set(np.arange(V_gl.dim())).difference(set(dofs_ess))))

    return dofsV_gl_ess, dofsV_gl_nat


