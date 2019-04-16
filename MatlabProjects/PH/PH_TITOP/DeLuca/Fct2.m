function dx = Fct2(input)
% FOR SIMULINK ONLY
%  Evaluates the right side of the differential equations of the two-link 
% flexible manipulator proposed by De Luca and Siciliano:"Closed-Form Dynamic Model Planar Multilink
% Lightweight Robots", IEEE Transactions of systems Vol 21, N°4, 1991
%
% F(t,x,u) = F(q,dq,u) = -h(q,dq) -Kq + Qu
%
% Inputs: 
%
%       u: input vector 
%       x: state vector, x = (q,dq)' with q = (theta1,theta2,delta11,delta12
%          ,delta21,delta22)'
%
% Outputs: dx (state vector derivate)
%
% Auxiliar functions: u = U(time) (actuators at the joints)
%
% Last revision:
% J. Alvaro Perez 23rd May 2016
%
% % Verification examples: Watch the graphics of the original paper
%--------------------------------------------------------------------------

%% NOT CLEAN:initialize constants manually and not by argument

global t11 t12 t21 t22 t31 t32 
global b111 b112 b113 b114 b121 b122 b123 b124 b131 b132 b133 b134 b141 b142 
global b143 b144 b151 b152 b153 b161 b162 b163 b221 b231 b232 b233 b234 b241 
global b242 b243 b244 b251 b261 b331 b332 b333 b341 b342 b343 b351 b352 b353
global b361 b362 b363 b441 b442 b443 b451 b452 b453 b461 b462 b463 b551 b561 
global b661 
global rho1 rho2 l1 l2 d2 m1 m2 mh2 mp Jo1 Jo2 Jh1 Jh2 Jp EI1 EI2
global phi11e phi12e phi_t_11e phi_t_12e phi21e phi22e phi_t_21e phi_t_22e 
global v11 v12 v21 v22 w11 w12 w21 w22 f11 f12 f21 f22 
x=input(1:12);Mdx=input(13:24);


c2 = cos(x(2)); % cos(theta2)
s2 = sin(x(2)); % sin(theta2)
t1 = t11*x(3) + t12*x(4); % t1 = t11*delta11 + t12*delta12
t2 = t21*x(5) + t22*x(6); % t2 = t21*delta21 + t22*delta22
t3 = t31*x(3) + t32*x(4); % t3 = t31*delta11 + t32*delta12

% Mass Matrix terms
B11 = b111 + b112*c2 + (b113*t1 + b114*t2)*s2;
B12 = b121 + b122*c2 + (b123*t1 + b124*t2)*s2;
B13 = b131 + b132*c2 + (b133*t2 + b134*x(4))*s2;
B14 = b141 + b142*c2 + (b143*t2 + b144*x(3))*s2;
B15 = b151 + b152*c2 + b153*t1*s2;
B16 = b161 + b162*c2 + b163*t1*s2;

B22 = b221;
B23 = b231 + b232*c2 + (b233*t2 + b234*t3)*s2;
B24 = b241 + b242*c2 + (b243*t2 + b244*t3)*s2;
B25 = b251;
B26 = b261;

B33 = b331 + b332*c2 + b333*t2*s2;
B34 = b341 + b342*c2 + b343*t2*s2;
B35 = b351 + b352*c2 + b353*t3*s2;
B36 = b361 + b362*c2 + b363*t3*s2;

B44 = b441 + b442*c2 + b443*t2*s2;
B45 = b451 + b452*c2 + b453*t3*s2;
B46 = b461 + b462*c2 + b463*t3*s2;

B55 = b551;
B56 = b561;

B66 = b661;

B = [B11 B12 B13 B14 B15 B16;
     B12 B22 B23 B24 B25 B26;
     B13 B23 B33 B34 B35 B36;
     B14 B24 B34 B44 B45 B46;
     B15 B25 B35 B45 B55 B56;
     B16 B26 B36 B46 B56 B66];

% Mass Matrix assembly (second order system) : 
M = [eye(6) zeros(6); zeros(6) B];

% Since M*dx/dt = F(x,t)
% The derivate of states is therefore
dx = inv(M)*Mdx;


end