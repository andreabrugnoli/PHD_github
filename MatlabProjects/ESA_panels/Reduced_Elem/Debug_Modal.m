clear all
close all
clc

m_panel = 43.2; 
lx=4.143;
ly=2.200;
t=0.04;
rho=m_panel/lx/ly/t;
E=70*10^9;
ni=0.35;
xi=0.003;
P_xy=[0,ly/2];   % P est au millieu du petit coté
C_xy=[lx,ly/2];


xP = P_xy(1); yP = P_xy(2);
xC = C_xy(:,1); yC = C_xy(:,2);

if xP>lx+eps | yP>ly+eps | xP<-eps | yP<-eps
    error('Not valid location for the Parent')
end

if xC>lx+eps | yC>ly+eps | xC<-eps | yC<-eps
    error('Not valid location for the Children')
end

% xA1 = lx; xA2 = 0; xA3 = lx;
% yA1 = 0; yA2 = ly; yA3 = ly;
xA1 = lx/2; xA2 = lx/2; xA3 = lx;
yA1 = 0; yA2 = ly; yA3 = 0;

dxPA = [xA1-xP; xA2-xP; xA3-xP];
dyPA = [yA1-yP; yA2-yP; yA3-yP];

D = E*t^3/12/(1-ni^2);          % Flexural Rigidity
DD = D*[1 ni 0;
        ni 1 0;
        0 0 (1-ni)/2];
    
syms x y;
xtilde = x - xP;
ytilde = y - yP;

% Rigid Part
phi_r_wT = [1, ytilde, -xtilde];
phi_r_thxT = [0, 1, 0];
phi_r_thyT = [0, 0, 1];

Phi_rT = [phi_r_wT; phi_r_thxT; phi_r_thyT]; 

% Flexible Part
P = [xtilde^2, xtilde*ytilde, ytilde^2, xtilde^3, xtilde^2*ytilde, xtilde*ytilde^2, ytilde^3, xtilde^3*ytilde, xtilde*ytilde^3; ...
     0, xtilde, 2*ytilde, 0, xtilde^2, 2*xtilde*ytilde, 3*ytilde^2, xtilde^3, 3*xtilde*ytilde^2; ...
     -2*xtilde, -ytilde, 0, -3*xtilde^2, -2*xtilde*ytilde, -ytilde^2, 0, -3*xtilde^2*ytilde, -ytilde^3];

n_modes = 9;
P_Ai = zeros(n_modes,n_modes);
for i = 1:3
   P_Ai((3*i-2):3*i,:) = [dxPA(i)^2, dxPA(i)*dyPA(i), dyPA(i)^2, dxPA(i)^3, dxPA(i)^2*dyPA(i), dxPA(i)*dyPA(i)^2, dyPA(i)^3, dxPA(i)^3*dyPA(i), dxPA(i)*dyPA(i)^3;
                       0, dxPA(i), 2*dyPA(i), 0, dxPA(i)^2, 2*dxPA(i)*dyPA(i), 3*dyPA(i)^2, dxPA(i)^3, 3*dxPA(i)*dyPA(i)^2;
                      -2*dxPA(i), -dyPA(i), 0, -3*dxPA(i)^2, -2*dxPA(i)*dyPA(i), -dyPA(i)^2, 0, -3*dxPA(i)^2*dyPA(i), -dyPA(i)^3];
end

% Shape function matrix
invP_Ai = inv(P_Ai);
phi_f_wT = P(1,:)*invP_Ai;
phi_f_thxT = P(2,:)*invP_Ai;
phi_f_thyT = P(3,:)*invP_Ai;

Phi_fT = [phi_f_wT; phi_f_thxT; phi_f_thyT];

% Mass Matrix
% N = [Phi_rT Phi_fT];
I = rho*diag([t,t^3/12,t^3/12]);
% M = double(int(int(N'*I*N,x,0,lx),y,0,ly));
% Subcomponents
Mrr = double(int(int(Phi_rT'*I*Phi_rT,x,0,lx),y,0,ly));
Mrf = double(int(int(Phi_rT'*I*Phi_fT,x,0,lx),y,0,ly));
MrfT = Mrf';
Mff = double(int(int(Phi_fT'*I*Phi_fT,x,0,lx),y,0,ly));

Q = [-2  0  0 -6*x -2*y 0 0 -6*x*y 0;
      0  0 -2 0 0 -2*x -6*y 0 -6*x*y;
      0 -2  0 0 -4*x  -4*y 0 -6*x^2 -6*y^2];
 
B = Q*inv(P_Ai);
% Stiffness Matrix
K = double(int((int(B'*DD*B,x,0,lx)),y,0,ly));

[V,omega2] = eig(K,Mff);
L_P = V'*MrfT;


prModes = 1;
if prModes == 1
set(0,'DefaultFigureWindowStyle','docked');
fontsize = 20;
% Modes 
    n_x = 21;
    n_y = 21;
    x_ev = linspace(0, lx, n_x);
    y_ev = linspace(0, ly, n_y);
    [X_ev,Y_ev] = meshgrid(x_ev, y_ev);
    W_def = zeros(n_y,n_x);

    for i = 1 : n_modes

       for jj = 1:n_x
           for kk=1:n_y
               x=x_ev(jj);  y= y_ev(kk);
               W_def(kk,jj) = subs(P(1,:))*invP_Ai*V(:,i);          
           end
       end

       figure();
       surf(X_ev,Y_ev,W_def); 
       hold on
       plot_xP = plot3(xP,yP,0,'r*');
       
       set(gca,'FontSize',fontsize)
       lgd = legend(plot_xP, 'Parent Node');
       set(lgd, 'Interpreter','latex')
       set(lgd,'Location','southwest')
       xlabel('$x [m]$', 'Interpreter','latex')
       ylabel('$y [m]$', 'Interpreter','latex')
       zlabel('$z [m]$', 'Interpreter','latex')
       title(['Mode ',num2str(i),': ',num2str(sqrt(omega2(i,i))./(2*pi)),' Hz'], 'Interpreter', 'latex')
       grid on
       
       %print(gcf,['Mode_',num2str(i)],'-depsc2');
    end

end
% Kinematic model 
n_C = length(C_xy(:,1));
tauCP = zeros(3,3,n_C);
for i = 1:n_C
    dx = xC(i) - xP; % x-distance of node C(i) from node P
    dy = yC(i) - yP; % y-distance of node C(i) from node P
    tauCP(:,:,i) =[1 dy -dx;0 1 0;0 0 1];
end

% Phi matrices 
PhiV_C = zeros(3,n_modes,n_C);
for i = 1:n_C
    x = xC(i);
    y = yC(i);
    PhiV_C(:,:,i) = subs(Phi_fT)*V;
end


R = 2*xi*omega2.^(1/2);
a = [zeros(n_modes) eye(n_modes); -omega2  -R]; 

n_input = 3*(n_C+1);
b = zeros(2*n_modes, n_input);
for i = 1:n_C
    b(:,(3*i-2):3*i) = [zeros(n_modes,3); PhiV_C(:,:,i)'];
end
b(:, (n_input-2):n_input) = [zeros(n_modes,3); -L_P];

c = zeros(n_input, 2*n_modes);
for i = 1:n_C
    c((3*i-2):3*i, :) = [-PhiV_C(:,:,i)*omega2,  -PhiV_C(:,:,i)*R];
end
c((n_input-2):n_input,:) = [L_P'*omega2, L_P'*R];

d = zeros(n_input,n_input);
for i = 1:n_C
    d((3*i-2):3*i, (n_input-2):n_input) =  tauCP(:,:,i)-PhiV_C(:,:,i)*L_P;
    d((n_input-2):n_input, (3*i-2):3*i) = [tauCP(:,:,i)-PhiV_C(:,:,i)*L_P]';
    for j = 1:n_C
      d((3*i-2):3*i, (3*j-2):3*j) = PhiV_C(:,:,i)*PhiV_C(:,:,j)';
    end
end

d( (n_input-2):n_input,(n_input-2):n_input ) = L_P'*L_P-Mrr;

MFzTxTy = ss(a,b,c,d);

