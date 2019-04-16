% Model Alvaro 
addpath('/home/a.brugnoli/GitProjects/MatlabProjects/PH/PH_TITOP/TITOP_tools/')
addpath('/home/a.brugnoli/GitProjects/MatlabProjects/PH/PH_TITOP/Settings/')
% model_constants
parameters

M1 = TwoPort_NElementsBeamTyRz(2, rho1,1,L1,EI1,1,0);
M2 = TwoPort_NElementsBeamTyRz(2, rho2,1,L2,EI2,1,0);
% M1 = TwoPortBeamTyRz(rho1,1,L1,EI1,1,0);
% M2 = TwoPortBeamTyRz(rho2,1,L2,EI2,1,0);
% Autre méthode
% Aug=[0 0 0 0;1 0 0 0;0 1 0 0;0 0 0 0;0 0 1 0;0 0 0 1];
% Dau=zeros(6,6); Dau(1,4)=1;Dau(4,1)=1;Dau(4,4)=-rho1*l1;
% Maug=Aug*M*Aug'+Dau;*tf(1, [1, 0])
% Mb=[eye(6);0 0 0 0 0 -1]*Maug*[eye(6) [0;0;0;0;0;1]];
% Mbm1_7=invio(Mb,7);

% On rajoute inertie 
M1(4, 4) = M1(4, 4)-J_joint1;
% On rajoute inertie
M2(4, 4) = M2(4, 4)-J_joint2;

% Rajoute les pivots
M1a=[eye(4);0 0 0 -1]*M1*[eye(4) [0;0;0;1]];
M1am1_5=invio(M1a,5);
% Brin 2
M2a=[eye(4);0 0 0 -1]*M2*[eye(4) [0;0;0;1]];
M2am1_5=invio(M2a,5);

% Ma=[eye(4);0 0 0 -1]*M*[eye(4) [0;0;0;1]];
% Mam1_5=invio(Ma,5);


% Angular configuration:
theta2 = 0;
T21=[cos(theta2) -sin(theta2) 0; sin(theta2) cos(theta2) 0;0 0 1];


[a,b,c,d]=linmod('titop_ol');
directDynamicsSuper = ss(a,b,c,d);
sys_titop=directDynamicsSuper*tf(1, [1 0]);


