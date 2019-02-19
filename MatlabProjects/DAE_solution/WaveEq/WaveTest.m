clear all; close all; clc;

load J.mat; load M.mat; load R.mat; load X0.mat; load partitioning;
load dofsX.mat; load dofsVp.mat; load dofsVq.mat;
Mint = sparse(M);
Jint = sparse(J);
Rint = sparse(R);
% Convert form Python the indexes
dofsVp = dofsVp + 1;
dofsVq = dofsVq + 1;
dofs2x = dofsX';

x0est = X0';
t_0 = 0;
t_f = 10;
n_ev = 300;
x_ev = dofs2x(dofsVp);
t_ev = linspace(t_0, t_f, n_ev);

end1 = partitioning(1); end2 = partitioning(2); end3 = partitioning(3);
n_p = length(dofsVp);
n_q = length(dofsVq);

n = length(X0); 

y_vec = sym([]);

syms t
for i = 1:n
    syms(sprintf('y_vec%d(t)',i))
    y_vecb = symfun(eval(sprintf('y_vec%d(t)', i)), t);
    y_vec = [y_vec; y_vecb];    
end

eqns = Mint * diff(y_vec(t), t, 1) == (Jint - Rint)*y_vec(t);
vars = y_vec;

isLowIndexDAE(eqns,vars)

[DAEeq, DAEvar] = reduceDAEIndex(eqns,vars);
% [DAEeq,DAEvar] = reduceRedundancies(DAEeq,DAEvar);

n_eq = length(DAEeq);

isLowIndexDAE(DAEeq, DAEvar)

f = daeFunction(DAEeq, DAEvar);
F = @(t,y,yp) f(t,y,yp);

y0est = zeros(n_eq,1);
y0est(1:n) = x0est;
yp0est = zeros(n_eq,1);

fix_y0  = zeros(n_eq,1); 
fix_y0(dofsVp) = 1;
[y0, yp0] = decic(F, 0, y0est, [], yp0est, []);

[t_sol,pHs] = ode15i(F, t_ev, y0, yp0);

sol = pHs';
n_t = length(t_sol)
e_st =  sol(1:end1, :);
e_os = sol(end2+1:end3, : );

ep_st = e_st(dofsVp, :);
ep_os = e_os(1, :);


w_st0 = zeros(n_p,1);
w_st = zeros(length(dofsVp), n_t);
w_st(:,1) = w_st0;
w_st_old = w_st(:,1);

w_os0 = 0;
w_os = zeros(n_ev,1);
w_os(1) = w_os0;
w_os_old = w_os(1);
dt_vec = diff(t_sol);

for i = 2:n_t
    w_st(:,i) = w_st_old + 0.5*(ep_st(:,i-1) + ep_st(:,i))*dt_vec(i-1);
    w_st_old  = w_st(:,i);

    w_os(i) = w_os_old + 0.5*(ep_os(i-1) + ep_os(i))*dt_vec(i-1);
    w_os_old = w_os(i);
end
% minZ = min(min(w_st));
% maxZ = max(max(w_st));

Hst_vec = zeros(n_ev,1);
Hos_vec = zeros(n_ev,1);

Mst=M(1:end1, 1:end1);
Mos=M(end2+1:end3, end2+1:end3);

for i= 1:n_ev
    Hst_vec(i) = 0.5 * (e_st(:, i)' * Mst * e_st(:, i));
    Hos_vec(i) = 0.5 * (e_os(:, i)' * Mos * e_os(:, i));
end
H_tot = Hst_vec + Hos_vec;
figure(1);
plot(t_sol, Hst_vec, 'b-', t_sol, Hos_vec, 'r-', t_sol, H_tot, 'g-')
legend('H string','H oscillator','H total')
title('Hamiltonians')
[X_ev, T_ev] = meshgrid(x_ev, t_ev);

figure(2);
surf(X_ev,T_ev,w_st')
hold on
line(ones(n_t,1),t_sol,w_os)
title('Vertical disiplacement')
xlabel('x')
ylabel('t')
legend('w string', 'w oscillator')
