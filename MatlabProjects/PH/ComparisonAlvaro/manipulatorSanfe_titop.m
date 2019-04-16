% Model Alvaro 
addpath('/home/a.brugnoli/GitProjects/MatlabProjects/tool_sanfe/Beam')
addpath('/home/a.brugnoli/GitProjects/MatlabProjects/tool_alvaro')
% model_constants
rho1 = 0.2; % Kg/m
rho2 = 0.2; % Kg/m

l1 = 0.5; % m
l2 = 0.5; % m
d2 = 0.25; % m

m1 = 0.1; % Kg
m2 = 0.1; % Kg
mh2 = 1.0; % Kg
mp = 0.1; % Kg

Jo1 = m1*l1^2/3; %0.0083; % Kgm2
Jo2 = m1*l1^2/3; % Kgm2
Jh1 = 0.1; % Kgm2
Jh2 = 0.1; % Kgm2
Jp = 0.0005; % Kgm2

EI1 = 1; % Nm2;
EI2 = 1; % Nm2; 
n_e = 2;
M = TwoPort_NElementsBeamTyRz(n_e, rho1, 1, l1, EI1, 1, 0);
% Autre méthode
% Aug=[0 0 0 0;1 0 0 0;0 1 0 0;0 0 0 0;0 0 1 0;0 0 0 1];
% Dau=zeros(6,6); Dau(1,4)=1;Dau(4,1)=1;Dau(4,4)=-rho1*l1;
% Maug=Aug*M*Aug'+Dau;
% Mb=[eye(6);0 0 0 0 0 -1]*Maug*[eye(6) [0;0;0;0;0;1]];
% Mbm1_7=invio(Mb,7);

% On rajoute l'inertie locale:
% Brin 1
M1=M+[0;0;0;-1]*Jh1*[0 0 0 1];
% Brin 2
M2=M+[0;0;0;-1]*Jh2*[0 0 0 1];


% Rajoute les pivots
% Brin 1
M1a=[eye(4);0 0 0 -1]*M1*[eye(4) [0;0;0;1]];
M1am1_5=invio(M1a,5);
% Brin 2
M2a=[eye(4);0 0 0 -1]*M2*[eye(4) [0;0;0;1]];
M2am1_5=invio(M2a,5);

% Ma=[eye(4);0 0 0 -1]*M*[eye(4) [0;0;0;1]];
% Mam1_5=invio(Ma,5);

% Angular configuration:
theta2 = 0;
T21=[cos(theta2) -sin(theta2) 0;sin(theta2) cos(theta2) 0;0 0 1];


[a,b,c,d]=linmod('modelSuperLUCAc');
directDynamicsSuper = ss(a,b,c,d);
sys_titop=directDynamicsSuper*tf(1, [1, 0]);

