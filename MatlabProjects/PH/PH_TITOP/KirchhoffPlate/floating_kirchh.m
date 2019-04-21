close all; clear all; clc;
addpath('./TITOP_Kirchh/')
addpath('/home/a.brugnoli/GitProjects/MatlabProjects/PH/Settings/')
addpath('./Matrices_Free/')
parameters;
load M_pH; load J_pH; load Q_pH; load B_pH

sys_phode = dss(J_pH, B_pH, B_pH', 0, M_pH);
n_C = (nx+1);
plate_titop = NPort_KirchhoffTzRyRx(Lx,Ly,h,rho,E,nu,nx,ny,1,n_C,0);

% plate_titop = balred(plate_titop, 30);

% % On rajoute l'inertie

P = [zeros(3), eye(3);
     eye(3), zeros(3)];
sys_titop = P * invio(plate_titop, [4,5,6]) * P;


fntsize = 13;
% figure(); sigma(sys_phode(1:2,1:2), 'r', sys_titop(1:2,1:2), 'b', {w0, wf});
% set_graphics_sigma(gca, fntsize);
% legend('PHODE', 'TITOP');
for i=1:6
%     if i ~=1 && i ~=2 && i ~=6 && i ~=7 && i ~=8 && i ~=12
    figure(i); sigma( sys_phode(i,i)* tf([1, 0], 1), 'r', sys_titop(i,i), 'b', {w0, wf});
%     set_graphics_sigma(gca, fntsize);
    legend('PHODE', 'TITOP');
%     end
end
% 
% figure(); sigma(sys_phode(1:6, 1:6), 'r', {w0, wf});
% set_graphics_sigma(gca, fntsize);
% legend('PHODE');
% figure(); sigma(sys_titop(1:6,1:6), 'b', {w0, wf});
% set_graphics_sigma(gca, fntsize);
% legend('TITOP');
% 
% figure(); sigma(sys_phode(6:12,6:12), 'r', {w0, wf});
% set_graphics_sigma(gca, fntsize);
% legend('PHODE');
% figure(); sigma(sys_titop(6:12,6:12), 'b', {w0, wf});
% set_graphics_sigma(gca, fntsize);
% legend('TITOP');




