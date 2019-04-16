% Constants of the  two-link flexible manipulator proposed
% by De Luca and Siciliano:"Closed-Form Dynamic Model Planar Multilink
% Lightweight Robots", IEEE Transactions of systems Vol 21, N�4, 1991
%
%
% Last revision:
% J. Alvaro Perez 19th May 2016
%
% % Verification examples: Watch the graphics of the original paper
%--------------------------------------------------------------------------

global h101 h102 h103 h104 h105 h106 h107 h108 h109 h110 h111 h112 h113 h114 
global h115 h116 h117 h118 h119 h120 h121 h122 h123 h124 h201 h202 h203 h204 
global h205 h206 h207 h208 h209 h210 h211 h212 h213 h214 h215 h216 h217 h218 
global h301 h302 h303 h304 h305 h306 h307 h308 h309 h310 h311 h312 h313 h314 
global h315 h316 h317 h318 h319 h320 h321 h322 h401 h402 h403 h404 h405 h406 
global h407 h408 h409 h410 h411 h412 h413 h414 h415 h416 h417 h418 h419 h420 
global h421 h422 h501 h502 h503 h504 h505 h506 h601 h602 h603 h604 h605 h606 
global t11 t12 t21 t22 t31 t32 
global K Q
global b111 b112 b113 b114 b121 b122 b123 b124 b131 b132 b133 b134 b141 b142 
global b143 b144 b151 b152 b153 b161 b162 b163 b221 b231 b232 b233 b234 b241 
global b242 b243 b244 b251 b261 b331 b332 b333 b341 b342 b343 b351 b352 b353
global b361 b362 b363 b441 b442 b443 b451 b452 b453 b461 b462 b463 b551 b561 
global b661 
global rho1 rho2 l1 l2 d2 m1 m2 mh2 mp Jo1 Jo2 Jh1 Jh2 Jp EI1 EI2
global phi11e phi12e phi_t_11e phi_t_12e phi21e phi22e phi_t_21e phi_t_22e 
global v11 v12 v21 v22 w11 w12 w21 w22 f11 f12 f21 f22 
%=================

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

% Parameters coming for the AMM for the nominal configuration:
phi11e = 0.186;
phi12e = 0.215;
phi_t_11e = 0.657;
phi_t_12e = -0.560;
phi21e = 0.883;
phi22e = -0.069;
phi_t_21e = 2.641;
phi_t_22e = -10.853;

v11 = 0.007;
v12 = 0.013;
v21 = 0.033;
v22 = 0.054;

w11 = 0.002;
w12 = 0.004;
w21 = 0.012;
w22 = 0.016;

f11 = 0.48; % Hz
f12 = 1.80; % Hz
f21 = 2.18; % Hz
f22 = 15.91; % Hz

% K matrix computation for the nominal configuration
K = diag([0,0, (2*pi*f11)^2*m1, (2*pi*f12)^2*m1, (2*pi*f21)^2*m2, (2*pi*f22)^2*m2]);

% Q matrix computation for the nominal configuration
Q = [eye(2); zeros(4,2)];

% b coefficients: First Round! 
%
b111 = Jh1 + Jo1 + Jh2 + mh2*l1^2 + Jo2 + m2*l1^2 + Jp + mp*(l1^2+l2^2);
b112 = 2*(m2*d2 + mp*l2)*l1;
b113 = 2*(m2*d2 + mp*l2);
b114 = -2*l1;
b121 = Jh2 + Jo2 + Jp + mp*l2^2;
b122 = (m2*d2 + mp*l2)*l1;
b123 = m2*d2 + mp*l2;
b124 = -l1;
b131 = w11 + (Jh2 + Jo2 + Jp + mp*l2^2)*phi_t_11e + (mh2 + m2 + mp)*l1*phi11e;
b132 = (m2*d2 + mp*l2)*(phi11e + l1*phi_t_11e);
b133 = -(phi11e + l1*phi_t_11e);
b134 = -(m2*d2 + mp*l2)*(phi11e*phi_t_12e - phi12e*phi_t_11e);
b141 = w12 + (Jh2 + Jo2 + Jp + mp*l2^2)*phi_t_12e + (mh2 + m2 + mp)*l1*phi12e;
b142 = (m2*d2 + mp*l2)*(phi12e + l1*phi_t_12e);
b143 = -(phi12e + l1*phi_t_12e);
b144 = -(m2*d2 + mp*l2)*(phi12e*phi_t_11e - phi11e*phi_t_12e);
b151 = w21 + Jp*phi_t_21e + mp*l2*phi21e;
b152 = (v21 + mp*phi21e)*l1;
b153 = v21 + mp*phi21e;
b161 = w22 + Jp*phi_t_22e + mp*l2*phi22e;
b162 = (v22 + mp*phi22e)*l1;
b163 = v22 + mp*phi22e;

% b coefficients: Second Round! 
%
b221 = Jh2 + Jo2 + Jp + mp*l2^2;
b231 = (Jh2 + Jo2 + Jp + mp*l2^2)*phi_t_11e;
b232 = (m2*d2 + mp*l2)*phi11e;
b233 = -phi11e;
b234 = -(m2*d2 + mp*l2)*phi11e;
b241 = (Jh2 + Jo2 + Jp + mp*l2^2)*phi_t_12e;
b242 = (m2*d2 + mp*l2)*phi12e;
b243 = -phi12e;
b244 = -(m2*d2 + mp*l2)*phi12e;
b251 = w21 + Jp*phi_t_21e + mp*l2*phi21e;
b261 = w22 + Jp*phi_t_22e + mp*l2*phi22e;

% b coefficients: Third Round! 
%

b331 = m1;
b332 = 2*(m2*d2 + mp*l2)*phi11e*phi_t_11e;
b333 = -2*phi11e*phi_t_11e;
b341 = 0;
b342 = (m2*d2 + mp*l2)*(phi11e*phi_t_12e + phi12e*phi_t_11e);
b343 = -(phi11e*phi_t_12e + phi12e*phi_t_11e);
b351 = (w21 + Jp*phi_t_21e + mp*l2*phi21e)*phi_t_11e;
b352 = (v21 + mp*phi21e)*phi11e;
b353 = -(v21 + mp*phi21e)*phi11e;
b361 = (w22 + Jp*phi_t_22e + mp*l2*phi22e)*phi_t_11e;
b362 = (v22 + mp*phi22e)*phi11e;
b363 = -(v22 + mp*phi22e)*phi11e;

% b coefficients: Fourth Round!
%

b441 = m1;
b442 = 2*(m2*d2 + mp*l2)*phi12e*phi_t_12e;
b443 = -2*phi12e*phi_t_12e;
b451 = (w21 + Jp*phi_t_21e + mp*l2*phi21e)*phi_t_12e;
b452 = (v21 + mp*phi21e)*phi12e;
b453 = -(v21 + mp*phi21e)*phi12e;
b461 = (w22 + Jp*phi_t_22e + mp*l2*phi22e)*phi_t_12e;
b462 = (v22 + mp*phi22e)*phi12e;
b463 = -(v22 + mp*phi22e)*phi12e;

b551 = m2;
b561 = 0;
b661 = m2;

% t coefficients

t11 = phi11e - l1*phi_t_11e;
t12 = phi12e - l1*phi_t_12e;
t21 = v21 + mp*phi21e;
t22 = v22 + mp*phi22e;
t31 = phi_t_11e;
t32 = phi_t_12e;

% h coefficients: First Round!
%

h101 = -2*(m2*d2 + mp*l2)*l1;
h102 = 2*(m2*d2 + mp*l2)*(phi11e - l1*phi_t_11e);
h103 = 2*(m2*d2 + mp*l2)*(phi12e - l1*phi_t_12e);
h104 = -2*(v21 + mp*phi21e)*l1;
h105 = -2*(v22 + mp*phi22e)*l1;
h106 = -(m2*d2 + mp*l2)*l1;
h107 = -(m2*d2 + mp*l2)*l1*phi_t_11e;
h108 = -2*(m2*d2 + mp*l2)*l1*phi_t_12e;
h109 = -2*(v21 + mp*phi21e)*l1;
h110 = -2*(v22 + mp*phi22e)*l1;
h111 = -2*(v21 + mp*phi21e)*l1*phi_t_11e;
h112 = -2*(v22 + mp*phi22e)*l1*phi_t_11e;
h113 = -2*(v21 + mp*phi21e)*l1*phi_t_12e;
h114 = -2*(v22 + mp*phi22e)*l1*phi_t_12e;
h115 = 2*(m2*d2 + mp*l2);
h116 = m2*d2 + mp*l2;
h117 = -(v21 + mp*phi21e);
h118 = -(v22 + mp*phi22e);
h119 = -2*l1;
h120 = -l1;
h121 = -(phi11e + l1*phi_t_11e);
h122 = -(phi12e + l1*phi_t_12e);
h123 = -(m2*d2 + mp*l2)*(phi11e*phi_t_12e - phi12e*phi_t_11e);
h124 = -(m2*d2 + mp*l2)*(phi12e*phi_t_11e - phi11e*phi_t_12e);

% h coefficients: Second Round!
%

h201 = (m2*d2 + mp*l2)*l1;
h202 = 2*(m2*d2 + mp*l2)*phi11e;
h203 = 2*(m2*d2 + mp*l2)*phi12e;
h204 = -(m2*d2 + mp*l2);
h205 = -(v21 + mp*phi21e);
h206 = -(v22 + mp*phi22e);
h207 = l1;
h208 = phi11e + l1*phi_t_11e;
h209 = phi12e + l1*phi_t_12e;
h210 = (m2*d2 + mp*l2)*(phi11e*phi_t_12e - phi12e*phi_t_11e);
h211 = (m2*d2 + mp*l2)*(phi12e*phi_t_11e - phi11e*phi_t_12e);
h212 = phi11e*phi_t_11e;
h213 = phi11e*phi_t_12e + phi12e*phi_t_11e;
h214 = (v21 + mp*phi21e)*phi11e;
h215 = (v22 + mp*phi22e)*phi11e;
h216 = phi12e*phi_t_12e;
h217 = (v21 + mp*phi21e)*phi12e;
h218 = (v22 + mp*phi22e)*phi12e;

% h coefficients: Third Round!
%

h301 = -(m2*d2 + mp*l2)*(phi11e - l1*phi_t_11e);
h302 = -2*(m2*d2 + mp*l2)*phi11e;
h303 = 2*(m2*d2 + mp*l2)*(phi12e*phi_t_11e - phi11e*phi_t_12e);
h304 = -2*(v21 + mp*phi21e)*phi11e;
h305 = -2*(v22 + mp*phi22e)*phi11e;
h306 = -(m2*d2 + mp*l2)*phi11e;
h307 = -2*(m2*d2 + mp*l2)*phi11e*phi_t_11e;
h308 = -2*(m2*d2 + mp*l2)*phi11e*phi_t_12e;
h309 = -2*(v21 + mp*phi21e)*phi11e;
h310 = -2*(v22 + mp*phi22e)*phi11e;
h311 = -2*(v21 + mp*phi21e)*phi11e*phi_t_11e;
h312 = -2*(v22 + mp*phi22e)*phi11e*phi_t_11e;
h313 = -2*(v21 + mp*phi21e)*phi11e*phi_t_12e;
h314 = -2*(v22 + mp*phi22e)*phi11e*phi_t_12e;
h315 = -(phi11e + l1*phi_t_11e);
h316 = -phi11e;
h317 = -2*phi11e*phi_t_11e;
h318 = -(phi11e*phi_t_12e + phi12e*phi_t_11e);
h319 = -(m2*d2 + mp*l2)*phi11e;
h320 = -(v21 + mp*phi21e)*phi11e;
h321 = -(v22 + mp*phi22e)*phi11e;
h322 = -(m2*d2 +mp*l2)*(phi11e*phi_t_12e - phi12e*phi_t_11e);

% h coefficients: Fourth Round!
%

h401 = -(m2*d2 + mp*l2)*(phi12e - l1*phi_t_12e);
h402 = -2*(m2*d2 + mp*l2)*phi12e;
h403 = 2*(m2*d2 + mp*l2)*(phi11e*phi_t_12e - phi12e*phi_t_11e);
h404 = -2*(v21 + mp*phi21e)*phi12e;
h405 = -2*(v22 + mp*phi22e)*phi12e;
h406 = -(m2*d2 + mp*l2)*phi12e;
h407 = -2*(m2*d2 + mp*l2)*phi12e*phi_t_11e;
h408 = -2*(m2*d2 + mp*l2)*phi12e*phi_t_12e;
h409 = -2*(v21 + mp*phi21e)*phi12e;
h410 = -2*(v22 + mp*phi22e)*phi12e;
h411 = -2*(v21 + mp*phi21e)*phi12e*phi_t_11e;
h412 = -2*(v22 + mp*phi22e)*phi12e*phi_t_12e;
h413 = -2*(v21 + mp*phi21e)*phi12e*phi_t_11e;
h414 = -2*(v22 + mp*phi22e)*phi12e*phi_t_12e;
h415 = -(phi12e + l1*phi_t_12e);
h416 = -phi12e;
h417 = -(phi11e*phi_t_12e + phi12e*phi_t_11e);
h418 = -2*phi12e*phi_t_12e;
h419 = -(m2*d2 + mp*l2)*phi12e;
h420 = -(v21 + mp*phi21e)*phi12e;
h421 = -(v22 + mp*phi22e)*phi12e;
h422 = -(m2*d2 + mp*l2)*(phi12e*phi_t_11e - phi11e*phi_t_12e);


% h coefficients: Fifth Round!
%

h501 = (v21 + mp*phi21e)*l1;
h502 = 2*(v21 + mp*phi21e)*phi11e;
h503 = 2*(v21 + mp*phi21e)*phi12e;
h504 = v21 + mp*phi21e;
h505 = -(v21 + mp*phi21e)*phi11e;
h506 = -(v21 + mp*phi21e)*phi12e;

% h coefficients: Sixth Round!
%

h601 = (v22 + mp*phi22e)*l1;
h602 = 2*(v22 + mp*phi22e)*phi11e;
h603 = 2*(v22 + mp*phi22e)*phi12e;
h604 = v22 + mp*phi22e;
h605 = -(v22 + mp*phi22e)*phi11e;
h606 = -(v22 + mp*phi22e)*phi12e;



