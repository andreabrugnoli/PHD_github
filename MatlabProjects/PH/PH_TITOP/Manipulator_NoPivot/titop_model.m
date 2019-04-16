% Model Alvaro 
addpath('../TITOP_tools/')
% model_constants
parameters

M1 = TwoPort_NElementsBeamTyRz(2, rho1,1,L1,EI1,1,0);
M2 = TwoPort_NElementsBeamTyRz(2, rho1,1,L1,EI1,1,0);
% M1 = TwoPortBeamTyRz(rho1,1,L1,EI1,1,0);
% M2 = TwoPortBeamTyRz(rho2,1,L2,EI2,1,0);
% Autre méthode
% Aug=[0 0 0 0;1 0 0 0;0 1 0 0;0 0 0 0;0 0 1 0;0 0 0 1];
% Dau=zeros(6,6); Dau(1,4)=1;Dau(4,1)=1;Dau(4,4)=-rho1*l1;
% Maug=Aug*M*Aug'+Dau;*tf(1, [1, 0])
% Mb=[eye(6);0 0 0 0 0 -1]*Maug*[eye(6) [0;0;0;0;0;1]];
% Mbm1_7=invio(Mb,7);

Aug=[0 0 0 0;1 0 0 0;0 1 0 0;0 0 0 0;0 0 1 0;0 0 0 1];
Dau=zeros(6,6); Dau(1,4)=1;Dau(4,1)=1;Dau(4,4)=-rho1*L1;
Maug1=Aug*M1*Aug'+Dau;
% On rajoute inertie et masse punctuelle
Maug1(4:6, 4:6) = Maug1(4:6, 4:6) - diag([0, 0, J_joint1]);

Dau(4,4)=-rho2*L2;
Maug2=Aug*M2*Aug'+Dau;
% On rajoute inertie et masse punctuelle
Maug2(4:6, 4:6) = Maug2(4:6, 4:6) - diag([m_joint2, m_joint2, J_joint2]);

Maug1 = invio(Maug1,6);
Maug2 = invio(Maug2,6);

% Angular configuration:
theta2 = 0;
T21=[cos(theta2) -sin(theta2) 0; sin(theta2) cos(theta2) 0;0 0 1];


[a,b,c,d]=linmod('titop_ol');
directDynamicsSuper = ss(a,b,c,d);
sys_titop_nopivot=directDynamicsSuper*tf(1, [1 0]);

