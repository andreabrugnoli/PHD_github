function MFzTxTy = Kirchhoff_1elem(lx, ly, t, rho, E, ni, P_xy, C_xy)

xP = P_xy(1); yP = P_xy(2);
xC = C_xy(:,1); yC = C_xy(:,2);

if xP>lx+eps | yP>ly+eps | xP<-eps | yP<-eps
    error('Not valid location for the Parent')
end

if xC>lx+eps | yC>ly+eps | xC<-eps | yC<-eps
    error('Not valid location for the Children')
end

xA1 = lx; xA2 = 0; xA3 = lx;
yA1 = 0; yA2 = ly; yA3 = ly;

px = [xA1-xP; xA2-xP; xA3-xP];
py = [yA1-yP; yA2-yP; yA3-yP];

D = E*t^3/12/(1-ni^2);          % Flexural Rigidity
DD = D*[1 ni 0;
        ni 1 0;
        0 0 (1-ni)/2];
    
syms x y;
xtilde = x - xP;
ytilde = y - yP;

% Rigid Part
phi_rT = [1, ytilde, -xtilde];
phi_r_thxT = [0, 1, 0];
phi_r_thyT = [0, 0, 1];

Phi_rT = [phi_rT; phi_r_thxT; phi_r_thyT]; 

% Flexible Part
P = [xtilde^2 xtilde*ytilde ytilde^2 xtilde^3 xtilde^2*ytilde xtilde*ytilde^2 ytilde^3 xtilde^3*ytilde xtilde*ytilde^3; ...
     0 xtilde 2*ytilde 0 xtilde^2 2*xtilde*ytilde 3*ytilde^2 xtilde^3 3*xtilde*ytilde^2; ...
     -2*xtilde -ytilde 0 -3*xtilde^2 -2*xtilde*ytilde -ytilde^2 0 -3*xtilde^2*ytilde -ytilde^3];

n_modes = 9;
C = zeros(n_modes,n_modes);
for i = 1:3
   C((3*i-2):3*i,:) = [px(i)^2 px(i)*py(i) py(i)^2 px(i)^3 px(i)^2*py(i) px(i)*py(i)^2 py(i)^3 px(i)^3*py(i) px(i)*py(i)^3;
                       0 px(i) 2*py(i) 0 px(i)^2 2*px(i)*py(i) 3*py(i)^2 px(i)^3 3*px(i)*py(i)^2;
                      -2*px(i) -py(i) 0 -3*px(i)^2 -2*px(i)*py(i) -py(i)^2 0 -3*px(i)^2*py(i) -py(i)^3];
end

% Shape function matrix
invC = inv(C);
phi_wT = P(1,:)*invC;
phi_thxT = P(2,:)*invC;
phi_thyT = P(3,:)*invC;

Phi_fT = [phi_wT; phi_thxT; phi_thyT];

% Mass Matrix
N = [Phi_rT Phi_fT];
I = rho*diag([t,t^3/12,t^3/12]);
M = double(int(int(N'*I*N,x,0,lx),y,0,ly));
% Subcomponents
Mrr = M(1:3, 1:3);
Mrf = M(1:3, 4:12);
MrfT = Mrf';
Mff = M(4:12, 4:12);

Q = [-2  0  0 -6*xtilde -2*ytilde 0 0 -6*xtilde*ytilde 0;
      0  0 -2 0 0 -2*xtilde -6*ytilde 0 -6*xtilde*ytilde;
      0 -2  0 0 -4*xtilde  -4*ytilde 0 -6*xtilde^2 -6*ytilde^2];
 
B = Q*inv(C);
% Stiffness Matrix
K = double(int((int(B'*DD*B,x,0,lx)),y,0,ly));

[V,omega2] = eig(K,Mff);
L_P = V'*MrfT;

% Kinematic model 
n_C = length(C_xy(:,1));
tauCP = zeros(3,3,n_C);
for i = 1:n_C
    dx = xC(i) - xP; % x-distance of node C(i) from node P
    dy = yC(i) - yP; % y-distance of node C(i) from node P
    tauCP(:,:,i) =[1 dy -dx;0 1 0;0 0 1];
end

% Phi matrices 
Phi_C = zeros(n_modes,3,n_C);
for i = 1:n_C
    x = xC(i);
    y = yC(i);
    Phi_C(:,:,i) = subs(Phi_fT');
end

invMff = inv(Mff);

a = [zeros(n_modes) eye(n_modes); -invMff*K  zeros(n_modes)]; 

n_input = 3*(n_C+1);
b = zeros(2*n_modes, n_input);
for i = 1:n_C
    b(:,(3*i-2):3*i) = [zeros(n_modes,3); invMff*Phi_C(:,:,i)];
end
b(:, (n_input-2):n_input) = [zeros(n_modes,3); -invMff*MrfT];

c = zeros(n_input, 2*n_modes);
for i = 1:n_C
    c((3*i-2):3*i, :) = [-Phi_C(:,:,i)'*invMff*K,  zeros(3,n_modes)];
end
c((n_input-2):n_input,:) = [Mrf*invMff*K, zeros(3,n_modes)];

d = zeros(n_input,n_input);
for i = 1:n_C
    d((3*i-2):3*i, (n_input-2):n_input) =  tauCP(:,:,i)-Phi_C(:,:,i)'*invMff*MrfT;
    d((n_input-2):n_input, (3*i-2):3*i) = [tauCP(:,:,i)-Phi_C(:,:,i)'*invMff*MrfT]';
    for j = 1:n_C
      d((3*i-2):3*i, (3*j-2):3*j) = Phi_C(:,:,i)'*invMff*Phi_C(:,:,j);
    end
end

d( (n_input-2):n_input,(n_input-2):n_input ) = Mrf*invMff*MrfT-Mrr;

MFzTxTy = ss(a,b,c,d);

