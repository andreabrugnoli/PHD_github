function [MtzRyRx,M2,K2,D,Lp,tauCGP] = TwoPort_KirchhoffTzRyRx(lx,ly,t,rho,E,ni,xi)
% [MtyRz,M2,K2,D] = TwoPort_KirchhoffTzRyRx(lx,ly,t,rho,E,ni,xi)
% model M (6x6) of a uniform Kirchhoff Plate characterized by:
%   * 
%   * elements number: ne
%   * mass density: rho  (Kg/m^3),
%   * section: S (m^2),
%   * lenght: ll (m),
%   * Young modulus: E (Pascal or N/m^2),
%   * xi: arbitrary damping ratio for all flexible modes.
%          ^ z    y
%          |     /  
%          |    /C (x=0, y=ly) 
%          |   x-----------/
%          |  /           / 
%          | /           / 
%          |/           / 
%          x-----------/----------> x
%          P (x=0, y=0)                
%       
%       
%   Only pure flexion in the plane (P,x,y) is considered.
%
%   The 4 inputs of M are:
%      * the external force F_Cy (along y) and torque T_Cz (around z) 
%        applied to the beam at point C,
%      * the linear ddot(y)_P (along y) and angular ddot(theta)_P
%        (around z) accelerations at point P.
%   The 4 outputs of M are:
%      * the linear ddot(y)_C (along y) and angular ddot(theta)_C
%        (around z) accelerations at point C,
%      * the external force F_Py (along y) and torque T_Pz (around z) 
%        applied to the beam at point P.

% Element Matrices
[Kel,Mel] = makeKirchhoffElement(rho,E,ni,lx,ly,t);

% Constraints
P = [eye(3);[1 0 -lx;0 1 0;0 0 1];[1 ly -lx;0 1 0;0 0 1];[1 ly 0;0 1 0;0 0 1]];
P = [P [zeros(3,9);eye(9,9)]];
 

M2 = P'*Mel*P;
K2 = P'*Kel*P;

[V,D,W] = eig(K2(4:12,4:12),M2(4:12,4:12));

EV_C = V(7:9,:);
 
cdP = [0;0;0];
cdC = [0;ly;0];
 
tauCP =[1 ly 0;0 1 0;0 0 1];
% 
Lp = V'*M2(1:3,4:12)';
% 
m = rho*lx*ly*t;
xcg = [lx/2;ly/2;0];
CGPvec=cdP-xcg;

tauCGP=[eye(3) -[0 CGPvec(3) -CGPvec(2);
    -CGPvec(3) 0 CGPvec(1);
    CGPvec(2) -CGPvec(1) 0] ;
    zeros(3) eye(3)];

Dpa = M2(1:3,1:3);
Dpa0 = Dpa - Lp'*Lp;


aa = [zeros(9,9) eye(9,9);-D zeros(9,9)];
bb = [zeros(9,6);EV_C' -Lp];
cc = [-EV_C*D zeros(3,9);Lp'*D zeros(3,9)];
dd = [EV_C*EV_C' tauCP-EV_C*Lp;(tauCP-EV_C*Lp)' -Dpa0];

MtzRyRx = ss(aa,bb,cc,dd);


end 
