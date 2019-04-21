clc
close all
clear all
addpath('./Matrices_Clamped/')
addpath('./TITOP_Kirchh/')

load E_dae; load J_dae; load B_dae;
% load J_ode; load Q_ode; load B_ode;
parameters

sys_phdae = dss(J_dae, B_dae, B_dae', 0, E_dae);
% sys_phode = ss(J_ode * Q_ode, B_ode, B_ode' * Q_ode, 0);
n_C = (nx+1);
plate_titop = NPort_KirchhoffTzRyRx(Lx,Ly,h,rho,E,nu,nx,ny,1,n_C,0);

der = tf([1, 0], 1);
int = tf(1, [1, 0]);
sysder = [der 0 0;
          0 der 0;
          0 0 der];

sysint = [int 0 0;
          0 int 0;
          0 0 int];
      
acc2vel = append(sysint, sysder);

P = [zeros(3), eye(3);
     eye(3), zeros(3)];
sys_titop = P * (acc2vel * plate_titop)  * P;

for i=1:6
%     if i ~=1 && i ~=2 && i ~=6 && i ~=7 && i ~=8 && i ~=12
    figure(i); bode( sys_phdae(i,i), 'r', sys_titop(i,i), 'b', {w0, wf});
%     set_graphics_sigma(gca, fntsize);
    legend('PHODE', 'TITOP');
% end
end


