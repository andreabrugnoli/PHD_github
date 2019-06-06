model_constants

M=TwoPortBeamTyRz(rho1,1,l1,EI1,1,0);
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
Gs=directDynamicsSuper;
Ms=dcgain(inv(Gs))

% Marice de masse rigide théorique:
Mrig=[8/3*rho1*l1^3+Jh1+Jh2+Jp+mh2*l1^2+4*mp*l1^2 5/6*rho1*l1^3+Jh2+Jp+2*mp*l1^2;
      5/6*rho1*l1^3+Jh2+Jp+2*mp*l1^2 rho1*l1^3/3+Jh2+Jp+mp*l1^2]
  
[a,b,c,d]=linmod('COMPARISON');
G0= ss(a,b,c,d);
dcgain(inv(G0))

figure
bodemag(Gs,G0)

% EN BF:
w1=10;
kp1=w1^2*Mrig(1,1); kv1=sqrt(2)*w1*Mrig(1,1);
w2=10;
kp2=w2^2*Mrig(2,2); kv2=sqrt(2)*w2*Mrig(2,2);

modelSuperLUCbf
sim('modelSuperLUCbf');

COMPARISON_bf
sim('COMPARISON_bf');

return
% Cela colle !!
% 
% [a,b,c,d]=linmod('modelSuperLUCAb');
% Gsb= ss(a,b,c,d);

inverseDynamicsSuper = inv(directDynamicsSuper);

[WnIs,zeta] = damp(inverseDynamicsSuper);
[WnDs,zeta] = damp(directDynamicsSuper);
% De Luca and Siciliano have 0.48 Hz, 1.80 Hz, 2.18 Hz and 15.91 Hz (with
% the articulations fixed, I suppose...that is, for our inverse dynamics
% (because it is in that model where we can fix the articulations)
f_d_s =  WnDs(1:2:8)/(2*pi)
f_i_s =  WnIs(1:2:8)/(2*pi)
disp('error %')
error = abs(f_i_s - [0.48; 1.8; 2.18; 15.91])./[0.48; 1.8; 2.18; 15.91]*100 


