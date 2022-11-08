
from firedrake import *

# Element tensors defining the local 3-by-3 block system
_A = Tensor(a)
_F = Tensor(L)

# Extracting blocks for Slate expression of the reduced system
A = _A.blocks
F = _F.blocks
S = A[2, 2] - A[2, :2] * A[:2, :2].inv * A[:2, 2]
E = F[2] - A[2, :2] * A[:2, :2].inv * F[:2]

# Assemble and solve: SΛ = E
Smat = assemble(S, bcs=[...])
Evec = assemble(E)
lambda_h = Function(M)
solve(Smat, lambda_h, Evec, solver_parameters={"ksp_type": "preonly", "pc_type": "lu"})
p_h = Function(V) # Function to store the result: P
u_h = Function(U) # Function to store the result: U

# Intermediate expressions
Sd = A[1, 1] - A[1, 0] * A[0, 0].inv * A[0, 1]
Sl = A[1, 2] - A[1, 0] * A[0, 0].inv * A[0, 2]
Lambda = AssembledVector(lambda_h) # Local coefficient vector for Λ
P = AssembledVector(p_h) # Local coefficient vector for P
# Local solve expressions for P and U
p_sys = Sd.solve(F[1] - A[1, 0] * A[0, 0].inv * F[0] - Sl * Lambda, decomposition="PartialPivLu")
u_sys = A[0, 0].solve(F[0] - A[0, 1] * P - A[0, 2] * Lambda, decomposition="PartialPivLu")
assemble(p_sys, p_h)
assemble(u_sys, u_h)