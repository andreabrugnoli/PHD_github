% Equivalent Matlab script to the Simulink model
clear all
close all
clc

m_panel = 43.2; 
lx=4.143; l1 = lx;
ly=2.200; l2 = ly;
t=0.04;
rho=m_panel/lx/ly/t;
E=70*10^9;
ni = 0.35;
xi=0.003;

nx=1; ny=4;

lx_1=lx; %m
ly_1=ly; %m
lx_2=lx; %m
ly_2=ly; %m
lx_3=lx; %m
ly_3=ly; %m
xi_1 = xi; 
xi_2 = xi;
xi_3 = xi;
 
M_SA = rho*t*(lx_1*ly_1+lx_2*ly_2+lx_3*ly_3);
Jzz_SA = 212;

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


P_id_1 = 3;
P_id_2 = 3;
P_id_3 = 3;
C_ids_1 = [4 8];
C_ids_2 = [7 4 8];
C_ids_3 = 7;

[A_sim1,B_sim1,C_sim1,D_sim1] = linmod('Sentinel_model');
sys_Simu1 = ss(A_sim1,B_sim1,C_sim1,D_sim1);
sys_Simu_red1 = minreal(sys_Simu1);
% Masses1 = dcgain(sys_Simu_red1);
lambda_1 = sort(eig(sys_Simu_red1.a));
%lambda_1 = lambda_1(real(lambda_1)<0);


[A_sim2,B_sim2,C_sim2,D_sim2] = linmod('Sentinel_model_1elem_modal');
sys_Simu2 = ss(A_sim2,B_sim2,C_sim2,D_sim2);
sys_Simu_red2 = minreal(sys_Simu2, 10^-8);
% Masses2 = dcgain(sys_Simu_red2);
lambda_2 = sort(eig(sys_Simu_red2.a));

real_lam_1 = real(lambda_1);
real_lam_1 = real_lam_1(1:length(lambda_2)); 
imag_lam_1 = imag(lambda_1);
imag_lam_1 = imag_lam_1(1:length(lambda_2));

fontsize = 25
figure(1)
plot(real_lam_1, imag_lam_1,'bo',real(lambda_2),imag(lambda_2),'r*')
hold on 
plot([0 0],[min([imag_lam_1;imag(lambda_2)]) max([imag_lam_1;imag(lambda_2)]) ],'g')
set(gca,'FontSize',fontsize)
lgd = legend('Four Elements','Reduced Element', 'Imaginary Axis');
set(lgd, 'Interpreter','latex')
set(lgd,'Location','northeast')
xlabel('Re$(\lambda)$', 'Interpreter','latex')
ylabel('Im$(\lambda)$', 'Interpreter','latex')
title(['Eigenvalues for three interconneted panels'], 'Interpreter', 'latex')
grid on

%print(gcf,['ReducedVs4Elem'],'-depsc2');


% figure(2)
% plot(real(lambda_Mine),imag(lambda_Mine),'r*')
% hold on 
% plot([0 0],[min([imag(lambda_Mine)]) max([imag(lambda_Mine)]) ],'g')
% legend('Matlab','Imag Axis')
% 
% figure(3)
% plot(real(lambda_S),imag(lambda_S),'bo')
% hold on 
% plot([0 0],[min([imag(lambda_S)]) max([imag(lambda_S)]) ],'g')
% legend('Simulink','Imag Axis')
