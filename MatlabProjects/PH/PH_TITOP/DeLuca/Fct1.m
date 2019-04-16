function Mdx = Fct1(input)
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
global h101 h102 h103 h104 h105 h106 h107 h108 h109 h110 h111 h112 h113 h114 
global h115 h116 h117 h118 h119 h120 h121 h122 h123 h124 h201 h202 h203 h204 
global h205 h206 h207 h208 h209 h210 h211 h212 h213 h214 h215 h216 h217 h218 
global h301 h302 h303 h304 h305 h306 h307 h308 h309 h310 h311 h312 h313 h314 
global h315 h316 h317 h318 h319 h320 h321 h322 h401 h402 h403 h404 h405 h406 
global h407 h408 h409 h410 h411 h412 h413 h414 h415 h416 h417 h418 h419 h420 
global h421 h422 h501 h502 h503 h504 h505 h506 h601 h602 h603 h604 h605 h606 
global t11 t12 t21 t22 t31 t32 
global K Q

u=input(1:2);x=input(3:14);
%% State variables definition x  = (q,dq) = (theta1,theta2,delta11,delta12
%  ,delta21,delta22,dtheta1, dtheta2,ddelta11,ddelta12,ddelta21,ddelta22)'



D11 = x(3); % delta11
D12 = x(4); % delta12
D21 = x(5); % delta21
D22 = x(6); % delta22

dt1 = x(7); % dtheta1
dt2 = x(8); % dtheta2
dD11 = x(9); % ddelta11
dD12 = x(10); % ddelta12
dD21 = x(11); % ddelta21
dD22 = x(12); % ddelta22

c2 = cos(x(2)); % cos(theta2)
s2 = sin(x(2)); % sin(theta2)
t1 = t11*D11 + t12*D12; % t1 = t11*delta11 + t12*delta12
t2 = t21*D21 + t22*D22; % t2 = t21*delta21 + t22*delta22
t3 = t31*D11 + t32*D12; % t3 = t31*delta11 + t32*delta12


% Computing h(q,dq);

h1 = ((h101*dt2 + h102*dD11 + h103*dD12 + h104*dD21 + h105*dD22)*dt1 +...
     (h106*dt2 + h107*dD11 + h108*dD12 + h109*dD21 + h110*dD22)*dt2 +...
     (h111*dD21 + h112*dD22)*dD11 + (h113*dD21 + h114*dD22)*dD12)*s2 + ...
     ((h115*dt1 + h116*dt2 + h117*dD21 +h118*dD22)*t1 +...
     (h119*dt1 + h120*dt2 + h121*dD11 +h122*dD12)*t2 +...
     h123*D12*dD11 + h124*D11*dD12)*dt2*c2;

h2 = (h201*dt1 + h202*dD11 + h203*dD12)*dt1*s2 + ...
     (((h204*dt1 + h205*dD21 + h206*dD22)*t1 +...
     (h207*dt1 + h208*dD11 + h209*dD12)*t2 + ...
      h210*D12*dD11 + h211*D11*dD12)*dt1 + ...
      ((h212*dD11 + h213*dD12)*t2 + (h214*dD21 + h215*dD22)*t3)*dD11 + ...
      (h216*dD12*t2 + (h217*dD21 + h218*dD22)*t3)*dD12)*c2;

h3 = ((h301*dt1 + h302*dt2 + h303*dD12 + h304*dD21 + h305*dD22)*dt1 + ...
      (h306*dt2 + h307*dD11 + h308*dD12 + h309*dD21 + h310*dD22)*dt2 + ...
       (h311*dD21 + h312*dD22)*dD11 + (h313*dD21 + h314*dD22)*dD12)*s2 + ...
       ((h315*dt1 + h316*dt2 + h317*dD11 + h318*dD12)*t2 + ...
       (h319*dt2 + h320*dD21 + h321*dD22)*t3 + h322*D12*dt1)*dt2*c2;

h4 = ((h401*dt1 + h402*dt2 + h403*dD11 + h404*dD21 + h405*dD22)*dt1 + ...
      (h406*dt2 + h407*dD11 + h408*dD12 + h409*dD21 + h410*dD22)*dt2 + ...
      (h411*dD21 + h412*dD22)*dD11 + (h413*dD21 + h414*dD22)*dD12)*s2 +...
      ((h415*dt1 + h416*dt2 + h417*dD11 + h418*dD12)*t2 + ...
       (h419*dt2 + h420*dD21 + h421*dD22)*t3 + h422*D11*dt1)*dt2*c2;
   
h5 = (h501*dt1 + h502*dD11 + h503*dD12)*dt1*s2 + ...
     (h504*t1*dt1 + (h505*dD11 + h506*dD12)*t3)*dt2*c2;

h6 = (h601*dt1 + h602*dD11 + h603*dD12)*dt1*s2 + ...
     (h604*t1*dt1 +(h605*dD11 + h606*dD12)*t3)*dt2*c2;

% Right hand of the M(t,y)y' = f(t,y) equation
H  = [h1; h2; h3; h4; h5; h6];
KX = K*x(1:6);
QU = Q*u;
Bddq = -H - KX + QU;

% Right hand of the dq/dt = dq/dt
dq = x(7:12);

% Final result
Mdx = [dq ;Bddq];


end

