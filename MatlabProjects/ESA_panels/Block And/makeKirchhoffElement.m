function[Kel,Mel] = makeKirchhoffElement(rho,E,ni,lx,ly,t)
% =========================================================================
% [Kel,Mel] = makeKirchhoffElement(E,ni,l1,l2,d)
%
%  ^
% y|
%  |
%  |
%  4 ------- 3
%  |         |
%  |         |
%  1 --------2 -----> x
%
% Input: 
% rho: density
% E :  Young Modulus [i.e.: Pa]
% ni:  Poisson Ratio [/]
% lx:  x-element length [i.e.: m]
% ly:  y-element length [i.e.: m]
% t :  plate thickness  [i.e.: m]
% =========================================================================
% 

% Coordinates

px = [0;lx;lx;0];
py = [0;0;ly;ly];


D = E*t^3/12/(1-ni^2);          % Flexural Rigidity


DD = [D ni*D 0; ...
      ni*D D 0; ...
      0 0 (1-ni)*D/2];
% Dm = - [DD zeros(3,2); ...
%         zeros(2,3) k*G*t.*eye(2,2)];
%     
syms x y;
P = sym([]);
P = [1 x y x^2 x*y y^2 x^3 x^2*y x*y^2 y^3 x^3*y x*y^3; ...
     0 0 1 0 x 2*y 0 x^2 2*x*y 3*y^2 x^3 3*x*y^2; ...
     0 -1 0 -2*x -y 0 -3*x^2 -2*x*y -y^2 0 -3*x^2*y -y^3];

C = zeros(12,12);
for i = 1:4
   C((3*i-2):3*i,:) = [1 px(i) py(i) px(i)^2 px(i)*py(i) py(i)^2 px(i)^3 px(i)^2*py(i) px(i)*py(i)^2 py(i)^3 px(i)^3*py(i) px(i)*py(i)^3; ...
     0 0 1 0 px(i) 2*py(i) 0 px(i)^2 2*px(i)*py(i) 3*py(i)^2 px(i)^3 3*px(i)*py(i)^2; ...
     0 -1 0 -2*px(i) -py(i) 0 -3*px(i)^2 -2*px(i)*py(i) -py(i)^2 0 -3*px(i)^2*py(i) -py(i)^3];
end

% Shape function matrix
N = sym([]);
N = P*inv(C);

Q = sym([]);
Q = [0 0 0 -2 0 0 -6*x -2*y 0 0 -6*x*y 0; ...
     0 0 0 0 0 -2 0 0 -2*x -6*y 0 -6*x*y; ...
     0 0 0 0 -2 0 0 -4*x  -4*y 0 -6*x^2 -6*y^2];
 
B = sym([]);
B = Q*inv(C);


% Stiffness Matrix
Kel = double(int((int(B.'*DD*B,x,0,lx)),y,0,ly));

% Mass Matrix
I = rho.*diag([t,t^3/12,t^3/12]);
Mel = double(int(int(N.'*I*N,x,0,lx),y,0,ly));

end

