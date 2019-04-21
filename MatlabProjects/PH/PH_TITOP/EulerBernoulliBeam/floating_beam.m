clc
close all
clear all
addpath('/home/a.brugnoli/GitProjects/MatlabProjects/tools_sanfe/Beam/')
addpath('./Matrices_FreeEB/')

load J_pH; load Q_pH; load B_pH

parameters
sys_pH = ss(J_pH*Q_pH, B_pH, B_pH'*Q_pH, 0);

M = TwoPort_NElementsBeamTyRz(2, rho1, 1, L1, EI1, 1, 0);

Aug=[0 0 0 0;1 0 0 0;0 1 0 0;0 0 0 0;0 0 1 0;0 0 0 1];
Dau=zeros(6,6); Dau(1,4)=1;Dau(4,1)=1;Dau(4,4)=-rho1*L1;
Maug=Aug*M*Aug'+Dau;

% % On rajoute l'inertie
Maug(4:6, 4:6) = Maug(4:6, 4:6) - diag([m_joint1, m_joint1, J_joint1]);

P = [zeros(3), eye(3);
     eye(3), zeros(3)];
sys_TITOP = P * invio(Maug, [4, 5, 6]) * tf(1, [1, 0]) * P;

sys_pH = minreal(sys_pH);
sys_TITOP = minreal(sys_TITOP);
figure(); sigma(sys_pH, 'b',{1e-3, 1e6})
figure(); sigma(sys_TITOP, 'r',{1e-3, 1e6})
figure(); sigma(sys_pH, 'b', sys_TITOP, 'r', {w0, wf})
legend('pHDAE', 'TITOP')

% for ind = [1:6]
%     figure(); sigma(sys_pH(ind,ind), 'b', sys_TITOP(ind,ind), 'r', {1e-3, 1e6}); legend('PH', 'TITOP')
% end



