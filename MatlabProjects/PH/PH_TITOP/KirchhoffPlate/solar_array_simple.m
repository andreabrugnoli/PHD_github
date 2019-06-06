clear all; close all; clc;
three_panels_simple;

addpath('./MatricesSASimple/')
load J_pH; load Q_pH; load B_pH

sys_phode = ss(J_pH*Q_pH, B_pH, B_pH'*Q_pH, 0);

sys_phode = balred(sys_phode, 60);
for i=1:3
%     if i ~=1 && i ~=2 && i ~=6 && i ~=7 && i ~=8 && i ~=12
    figure(); sigma( sys_phode(i,i), 'r', sys_titop(i,i), 'b', {w0, wf});
%     set_graphics_sigma(gca, fntsize);
    legend('PHODE', 'TITOP');
%     end
end