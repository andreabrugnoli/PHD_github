% Equivalent Matlab script to the Simulink model
clear all
close all
clc
m_panel = 43.2;
Jzz_SA = 212;

lx=4.143;
ly=2.200;
t=0.04;
rho=m_panel/lx/ly/t;
E=70*10^9;
xi=0.003; 
ni = 0.35;

lx_1=lx; %m
ly_1=ly; %m
lx_2=lx; %m
ly_2=ly; %m
lx_3=lx; %m
ly_3=ly; %m
xi_1 = xi; xi_2 = xi; xi_3 = xi;


P_xy_1 = [0, ly/4];
P_xy_2 = [0, ly/4];
P_xy_3 = [0, ly/4];

C_xy_1 = [lx, ly/4;
          lx, 3*ly/4];
      
C_xy_2 = [0,  3*ly/4;
          lx, ly/4;
          lx, 3*ly/4];
      
C_xy_3 = [0, 3*ly/4];

ind_1 = 0;
ind_2 = [1:3];
ind_3 = [1:3];


[A_sim,B_sim,C_sim,D_sim] = linmod('Sentinel_model_1elem_modal');
sys_Simu = ss(A_sim,B_sim,C_sim,D_sim);
sys_Simu_red = minreal(sys_Simu, 10^-8);
Masses = dcgain(sys_Simu_red);
lambda_S = eig(sys_Simu_red.a);

fontsize = 30;
figure();
plot(real(lambda_S),imag(lambda_S),'r*')
hold on 
plot([0 0],[min([imag(lambda_S)]) max([imag(lambda_S)]) ],'b')

set(gca,'FontSize',fontsize)
lgd = legend('Eigenvalues', 'Imaginary Axis');
set(lgd, 'Interpreter','latex')
set(lgd,'Location','northeast')
xlabel('Re$(\lambda)$', 'Interpreter','latex')
ylabel('Im$(\lambda)$', 'Interpreter','latex')
title(['Eigenvalues for three interconneted panels'], 'Interpreter', 'latex')
grid on

print(gcf,['Eigen_3panels'],'-depsc2');