clc
close all
clear all
addpath('./Matrices_ClampedEB/')
addpath('/home/a.brugnoli/GitProjects/MatlabProjects/tools_sanfe/Beam/')

load E_dae; load J_dae; load B_dae;
load J_ode; load Q_ode; load B_ode;

parameters

sys_phdae = dss(J_dae, B_dae, B_dae', 0, E_dae);
sys_phode = ss(J_ode * Q_ode, B_ode, B_ode' * Q_ode, 0);

M = TwoPort_NElementsBeamTyRz(2, rho1, 1, L1, EI1, 1, 0);

Aug=[0 0 0 0;1 0 0 0;0 1 0 0;0 0 0 0;0 0 1 0;0 0 0 1];
Dau=zeros(6,6); Dau(1,4)=1;Dau(4,1)=1;Dau(4,4)=-rho1*L1;
Maug=Aug*M*Aug'+Dau;
% On rajoute inertie et masse punctuelle
Maug(4:6, 4:6) = Maug(4:6, 4:6) - diag([m_joint1, m_joint1, J_joint1]);

der = tf([1, 0], 1);
int = tf(1, [1, 0]);
n_u = 3;
acc2vec = append(int, int, int, 1, 1, 1);
vel2acc = append(1, 1, 1, der, der, der);

P = [zeros(n_u), eye(n_u);
     eye(n_u), zeros(n_u)];
sys_TITOPvel = P * acc2vec * Maug * vel2acc * P;

% figure(); sigma(sys_phdae, 'b', sys_phode, 'r', {w0, wf})
% legend('pHDAE', 'pHODE')

figure(); sigma(sys_phdae, 'b', sys_TITOPvel, 'r', {w0, wf})
legend('pHDAE', 'TITOP')
% ind = 1;
% figure(); sigma(sys_phdae(ind, ind), 'b', {w0, wf})
% figure(); sigma(sys_TITOPvel(ind, ind), 'r', {w0, wf})



