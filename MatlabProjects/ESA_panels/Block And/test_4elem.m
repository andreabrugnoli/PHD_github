% Equivalent Matlab script to the Simulink model
clear all
close all
clc

% Equivalent Matlab script to the Simulink model
clear all
close all
clc

l1=1; %m
l2=1; %m
t=0.003; %m
rho = 2000; %M_SA/(l1*l2*3*t); % kg/m^3
E = 70*10^9; % Pa
ni = 0.35; 
nx = 1;
ny = 1;
xi = 0.003;
xi_1 = xi; xi_2 = xi; xi_3 = xi;

lx=1; ly=1;

lx_1=lx; %m
ly_1=ly; %m
lx_2=lx; %m
ly_2=ly; %m
lx_3=lx; %m
ly_3=ly; %m
 
M_SA = rho*t*(lx_1*ly_1+lx_2*ly_2+lx_3*ly_3);
Jzz_SA = 212;

P_xy_1 = [0, 0];
P_xy_2 = [0, 0];
P_xy_3 = [0, 0];

C_xy_1 = [lx, 0;
          lx, ly];
      
C_xy_2 = [0,  ly;
          lx, 0;
          lx, ly];
      
C_xy_3 = [0, ly];

ind_1 = 0;
ind_2 = [1:3];
ind_3 = [1:3];


P_id_1 = 1;3;%
P_id_2 = 1;
P_id_3 = 1;
C_ids_1 = [2 4]; %[4 8];
C_ids_2 = [3 2 4]; %[7 4 8]; %
C_ids_3 = 3; %7; %

[A_sim1,B_sim1,C_sim1,D_sim1] = linmod('Sentinel_model');
sys_Simu1 = ss(A_sim1,B_sim1,C_sim1,D_sim1);
sys_Simu_red1 = minreal(sys_Simu1, 10^-8);

lambda_1 = eig(sys_Simu_red1.a);

[A_sim2,B_sim2,C_sim2,D_sim2] = linmod('Sentinel_model_1elem_modal');
sys_Simu2 = ss(A_sim2,B_sim2,C_sim2,D_sim2);
sys_Simu_red2 = minreal(sys_Simu2, 10^-8);
Masses = dcgain(sys_Simu_red2);
lambda_2 = eig(sys_Simu_red2.a);


figure(1)
plot(real(lambda_1),imag(lambda_1),'bo',real(lambda_2),imag(lambda_2),'r*')
hold on 
plot([0 0],[min([imag(lambda_1);imag(lambda_2)]) max([imag(lambda_1);imag(lambda_1)]) ],'g')
legend('Fra','Mio','Imag Axis')

