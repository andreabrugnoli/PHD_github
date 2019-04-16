% Model Alvaro 
addpath('../TITOP_tools/')
addpath('../Parameters/')
% model_constants
parameters

M1 = TwoPort_NElementsBeamTyRz(2, rho1,1,L1,EI1,1,0);
M2 = TwoPort_NElementsBeamTyRz(2, rho1,1,L1,EI1,1,0);
% M1 = TwoPortBeamTyRz(rho1,1,L1,EI1,1,0);
% M2 = TwoPortBeamTyRz(rho2,1,L2,EI2,1,0);
% On rajoute J_joint
% On rajoute inertie 
M1(4, 4) = M1(4, 4)-J_joint1;
% On rajoute inertie
M2(4, 4) = M2(4, 4)-J_joint2;

% Autre méthode
Aug=[0 0 0 0;1 0 0 0;0 1 0 0;0 0 0 0;0 0 1 0;0 0 0 1];
Dau=zeros(6,6); Dau(1,4)=1;Dau(4,1)=1;
Dau(4,4)=-rho1*L1;
Maug1=Aug*M1*Aug'+Dau;
Mbm1=[eye(6);0 0 0 0 0 -1]*Maug1*[eye(6) [0;0;0;0;0;1]];
Mbm1_7=invio(Mbm1,7);

Dau(4,4)=-rho2*L2;
Maug2=Aug*M2*Aug'+Dau;
Mbm2=[eye(6);0 0 0 0 0 -1]*Maug2*[eye(6) [0;0;0;0;0;1]];
Mbm2_7=invio(Mbm2,7);

% Angular configuration:
theta2 = 0;
T21=[cos(theta2) -sin(theta2) 0; sin(theta2) cos(theta2) 0;0 0 1];


[a,b,c,d]=linmod('titopSimple_ol_pivot');
directDynamicsSuper = ss(a,b,c,d);
sys_titop=directDynamicsSuper*tf(1, [1 0]);

