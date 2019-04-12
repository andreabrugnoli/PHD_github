function [] = printModes_1elem(M,P_xy,lx,ly)

[A,B,C,D]=ssdata(M);

xP = P_xy(1); yP = P_xy(2);

syms x y;
xtilde = x - xP;
ytilde = y - yP;

P_w = [xtilde^2, xtilde*ytilde, ytilde^2, xtilde^3, xtilde^2*ytilde, xtilde*ytilde^2, ytilde^3, xtilde^3*ytilde, xtilde*ytilde^3];

xA1 = lx; xA2 = 0; xA3 = lx;
yA1 = 0; yA2 = ly; yA3 = ly;

dxPA = [xA1-xP; xA2-xP; xA3-xP];
dyPA = [yA1-yP; yA2-yP; yA3-yP]; 
 
n_modes = 9;
P_Ai = zeros(n_modes,n_modes);
for i = 1:3
   P_Ai((3*i-2):3*i,:) = [dxPA(i)^2, dxPA(i)*dyPA(i), dyPA(i)^2, dxPA(i)^3, dxPA(i)^2*dyPA(i), dxPA(i)*dyPA(i)^2, dyPA(i)^3, dxPA(i)^3*dyPA(i), dxPA(i)*dyPA(i)^3;
                       0, dxPA(i), 2*dyPA(i), 0, dxPA(i)^2, 2*dxPA(i)*dyPA(i), 3*dyPA(i)^2, dxPA(i)^3, 3*dxPA(i)*dyPA(i)^2;
                      -2*dxPA(i), -dyPA(i), 0, -3*dxPA(i)^2, -2*dxPA(i)*dyPA(i), -dyPA(i)^2, 0, -3*dxPA(i)^2*dyPA(i), -dyPA(i)^3];
end

invP_Ai = inv(P_Ai);
invMffK = -A((n_modes+1):end,1:n_modes);
[V,omega2] = eig(invMffK);

set(0,'DefaultFigureWindowStyle','docked');
fontsize = 25;
% Modes 
n_x = 21;
n_y = 21;
x_ev = linspace(0, lx, n_x);
y_ev = linspace(0, ly, n_y);
[X_ev,Y_ev] = meshgrid(x_ev, y_ev);
w_def = zeros(n_y,n_x);

for i = 1 : n_modes

   for jj = 1:n_x
       for kk=1:n_y
           x=x_ev(jj);  y=y_ev(kk);
           w_def(kk,jj) = subs(P_w)*invP_Ai*V(:,i);           
       end
   end

   figure();
   surf(X_ev,Y_ev,w_def); 
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

return;