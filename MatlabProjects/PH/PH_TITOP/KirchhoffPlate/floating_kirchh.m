% close all; clear all; clc;
addpath('./TITOP_Kirchh/')
addpath('/home/a.brugnoli/GitProjects/MatlabProjects/PH/Settings/')
addpath('./Matrices_Free/')
parameters;
load J_pH; load Q_pH; load B_pH

sys_phode = ss(J_pH*Q_pH, B_pH, B_pH'*Q_pH, 0);
n_P = (nx+1)*ny/2 + 1;
n_C = (nx+1)*(ny/2 + 1);
plate_titop = NPort_KirchhoffTzRyRx(Lx,Ly,h,rho,E,nu,nx,ny,n_P,n_C,0.00001);
plate_titop = balred(plate_titop, length(Q_pH));
% % On rajoute l'inertie

P = [zeros(3), eye(3);
     eye(3), zeros(3)];
sys_titop = P * invio(plate_titop, [4,5,6]) * tf(1, [1, 0]) * P;

fntsize = 13;

for i=1:6
%     if i ~=1 && i ~=2 && i ~=6 && i ~=7 && i ~=8 && i ~=12
    figure(); sigma( sys_phode(i,i), 'r', sys_titop(i,i), 'b', {w0, wf});
%     set_graphics_sigma(gca, fntsize);
    legend('PHODE', 'TITOP');
%     end
end

% sys_ph = minreal(sys_phode);
% sys_tt = minreal(sys_titop);
% 
% ind_in = 1; ind_fin = 6;
% figure(); sigma(sys_ph(ind_in:ind_fin, ind_in:ind_fin), 'r',sys_tt(ind_in:ind_fin, ind_in:ind_fin), 'b', {w0, wf});
% set_graphics_sigma(gca, fntsize);
% legend('PHODE', 'TITOP');





