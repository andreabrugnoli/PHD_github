using NPZ
cd("/home/a.brugnoli/JuliaProjects/TestDAE/")
M = npzread("M.npy")
J = npzread("J.npy")
R = npzread("R.npy")
y0 = npzread("X0.npy")

n = length(y0)
f(u,p,t) = (J) * u

function fDAE(res,du,u,p,t)
  res = M * du - (J - R) * u
end

function massOde(res,du,u,p,t)
  res = M * du - (J - R) * u
end

t_0 = 0.0
t_f = 10.0
tspan = (t_0, t_f)
n_ev = 100
dt = (t_f - t_0)/n_ev

using DifferentialEquations
m_ode_prob = ODEProblem(ODEFunction(f; mass_matrix = M), y0, tspan)

sol = solve(m_ode_prob, RadauIIA5(), saveat=dt)
#
# y_sol = sol.u
# t_ev = sol.t
# n_ev = length(t_ev)
#
# cd("/home/a.brugnoli/PycharmProjects/SolveDAEs/")
#
# x_sol = zeros(n, n_ev)
#
# for i = 1:n_ev
#   x_sol[:,i] = y_sol[i]
# end
#
# npzwrite("sol.npy", x_sol)
# npzwrite("t_ev.npy", t_ev)
