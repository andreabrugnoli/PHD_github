clear all;
close all;

% domain dimensions
Lx = 1;
Ly = 1;

% number of elements
Nex = 10;
Ney = 10;

% element size
dx = Lx / Nex;
dy = Ly /Ney;
% basis functions
syms x y
phix = [(dx/2-x)/dx; (x+dx/2)/dx];
phiy = [(y+dy/2)/dy; (dy/2-y)/dy];
phixy = phix * phiy.';
phixy = phixy.';
phixy = phixy(:);

phi2 = 1 *(x/x);
phi3 = 1 *(x/x);

% element matrices
M1 = int(int(phixy * phixy.',-dx/2,dx/2),-dy/2,dy/2);
M2 = int(int((phi2 * phi2.'),-dx/2,dx/2),-dy/2,dy/2);
M3 = int(int((phi3 * phi3.'),-dx/2,dx/2),-dy/2,dy/2);
Dx = int(int( diff(phixy,x) * phi2.',-dy/2,dy/2),-dx/2,dx/2);
Dy = int(int( diff(phixy,y) * phi3.',-dx/2,dx/2),-dy/2,dy/2);

% boundary input left
Bpartial_left = int(phiy, -dy/2,dy/2);
Bpartial_down = int(phix, -dx/2,dx/2);
% assemblage of matrices

vectorinc = [1:(Nex+1)*(Ney+1)];
matrixindex = reshape(vectorinc,Nex+1,Ney+1)';
elmatrix = @(nelx, nely) matrixindex((nely:nely+1), (nelx:nelx+1));
%
vectorinc2 = [1:(Nex)*(Ney)];
matrixindex2 = reshape(vectorinc2,Nex,Ney)';


Massmatrix1 = zeros((Nex+1)*(Ney+1));
Massmatrix2 = zeros(Nex*Ney);
Massmatrix3 = zeros(Nex*Ney);
Dxmatrix = zeros((Nex+1)*(Ney+1), Nex*Ney);
Dymatrix = zeros((Nex+1)*(Ney+1), Nex*Ney);
Bpartial_left_full = zeros((Nex+1)*(Ney+1),Ney);
Bpartial_right_full = zeros((Nex+1)*(Ney+1),Ney);
Bpartial_up_full = zeros((Nex+1)*(Ney+1),Nex);
Bpartial_down_full = zeros((Nex+1)*(Ney+1),Nex);
%
for ix = 1:Nex
    for iy = 1:Ney
        elMatrix = elmatrix(ix,iy);
        elMatrix = elMatrix(:);
        for i = 1:4
            for j = 1:4
            Massmatrix1(elMatrix(i), elMatrix(j)) = Massmatrix1(elMatrix(i), elMatrix(j)) +  M1(i,j);
            end
            Dxmatrix(elMatrix(i), matrixindex2(iy,ix)) = Dxmatrix(elMatrix(i), matrixindex2(iy,ix)) + Dx(i);
            Dymatrix(elMatrix(i), matrixindex2(iy,ix)) = Dymatrix(elMatrix(i), matrixindex2(iy,ix)) + Dy(i);
        end
       Massmatrix2(matrixindex2(iy,ix),matrixindex2(iy,ix)) = M2;
       Massmatrix3(matrixindex2(iy,ix),matrixindex2(iy,ix)) = M3;
       if ix == 1
           [iy, ix]
           Bpartial_left_full(elMatrix(1), iy) = Bpartial_left_full(elMatrix(1), iy)+Bpartial_left(1);
           Bpartial_left_full(elMatrix(2), iy) = Bpartial_left_full(elMatrix(2), iy )+Bpartial_left(2);
       end
       if ix == Nex
           [iy, ix]
           Bpartial_right_full(elMatrix(3), iy) = Bpartial_right_full(elMatrix(3), iy)+Bpartial_left(1);
           Bpartial_right_full(elMatrix(4), iy) = Bpartial_right_full(elMatrix(4), iy)+Bpartial_left(2);
       end
       if iy == 1
           Bpartial_up_full(elMatrix(1), ix) = Bpartial_up_full(elMatrix(1), ix)+Bpartial_down(1);
           Bpartial_up_full(elMatrix(3), ix) = Bpartial_up_full(elMatrix(3), ix)+Bpartial_down(1);
       end
       if iy == Ney
           Bpartial_down_full(elMatrix(2), ix) = Bpartial_down_full(elMatrix(2), ix)+Bpartial_down(1);
           Bpartial_down_full(elMatrix(4), ix) = Bpartial_down_full(elMatrix(4), ix)+Bpartial_down(1);
       end
    end
end

%% saint-venant equations
N1 = size(Dxmatrix,1);
N2 = size(Dxmatrix,2);
J = [zeros(size(Massmatrix1)) Dxmatrix Dymatrix; -Dxmatrix' zeros(N2,2*N2); -Dymatrix' zeros(N2,2*N2)];
Q = blkdiag(inv(Massmatrix1), inv(Massmatrix2), inv(Massmatrix3));
%
[v,a] = eig(J*Q);
freq = imag(diag(a));
positivefreqs = freq>.001;
freq = freq(positivefreqs);
[freqord freqindex] = sort(freq)
vpositive = v(:, positivefreqs);
%
modalshapes = vpositive(1:N1,:);
Nfreq = 4;
modalshapen = Massmatrix1 \ modalshapes(:,freqindex(Nfreq));
freq(freqindex(Nfreq))

figure(1);
subplot(2,1,1)
surf(reshape(real(modalshapen), Nex+1,Ney+1)')
subplot(2,1,2)
surf(reshape(imag(modalshapen), Nex+1,Ney+1)')

%% 
A = J*Q;

xpos = linspace(0,Lx, Nex+1);
ypos = linspace(0,Ly, Ney+1);
[XX, YY] = meshgrid(xpos, ypos)
XX = XX'; YY = YY';
%
x0 = zeros(size(A,1),1);
x0(1:N1) = Massmatrix1 * (1+XX(:)*0.0+YY(:)*0.0 - XX(:).*YY(:)*0.5*0);
%x0(1:N1) = Massmatrix1 * (1+XX(:)*0.0+(YY(:)>Ly/2) );
figure;
surf(reshape((Massmatrix1 \ x0(1:N1))',  Nex+1,Ney+1))
%
B = zeros(size(A,1),1);
%B = [Bpartial_left_full; zeros(size(A,1)-(Nex+1)*(Ney+1),1)]*0.2;
B = [Bpartial_up_full*ones(Nex,1) -Bpartial_left_full*ones(Ney,1); zeros(size(A,1)-(Nex+1)*(Ney+1),2)]*0.2;
C = eye(size(A,1));
sys = ss(A,B,C,0);
%
%y = lsim(sys,ones(length(0:0.01:10),2)*0.6,0:0.01:10, x0);
tvec = 0:0.01:10;
inps = sin(pi*tvec) .* (tvec < 1);
inputs = repmat(inps,2,1)*1

y = lsim(sys,inputs,tvec, x0);



xl = linspace(0,1,Nex+1);
yl = linspace(0,1,Ney+1);
[XX YY] = meshgrid(xl, yl);
figure('color','w');
NN = 4
subplot(221);
in = 10*NN;
surf(XX,YY,reshape(Massmatrix1 \ y(in,1:N1)', Nex+1,Ney+1)')
view(-70,80)
title(strcat(num2str(tvec(in)), ' s'))
subplot(222);
in = 20*NN;
surf(XX,YY,reshape(Massmatrix1 \ y(in,1:N1)', Nex+1,Ney+1)')
view(-70,80)
title(strcat(num2str(tvec(in)), ' s'))
subplot(223);
in = 30*NN;
surf(XX,YY,reshape(Massmatrix1 \ y(in,1:N1)', Nex+1,Ney+1)')
view(-70,80)
title(strcat(num2str(tvec(in)), ' s'))
subplot(224);
in = 40*NN;
surf(XX,YY,reshape(Massmatrix1 \ y(in,1:N1)', Nex+1,Ney+1)')
view(-70,80)
title(strcat(num2str(tvec(in)), ' s'))

fig = gcf;
set(fig,'Units','Inches');
set(fig,'Position', [7.0208    0.8125    9.9479    9.3229]);
pos = get(fig,'Position');

fig.PaperPositionMode = 'auto';
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print('simulation2D_borderinflow_exc','-dpdf')


%%
for i = 1:size(y,1)
   figure(2)
   surf(reshape(Massmatrix1 \ y(i,1:N1)', Nex+1,Ney+1)')
   height(i) = sum(y(i,1:N1));
   axis([0 20 0 20 0.5 1.5]) 
   title(strcat(num2str(tvec(i)), ' s'))
   pause(0.01)
end

%%
for i = 1:size(y,1)
   height(i) = sum(y(i,1:N1));
   discreteHamiltonian(i) = 0.5*(y(i,1:N1)*inv(Massmatrix1)*y(i,1:N1)' + y(i,N1+1:N1+N2)*inv(Massmatrix2)*y(i,N1+1:N1+N2)'+y(i,N1+N2+1:N1+N2+N2)*inv(Massmatrix3)*y(i,N1+N2+1:N1+N2+N2)');
end
%%
figure('color','w'); plot(tvec,discreteHamiltonian); grid on;
xlabel('time (s)');
ylabel('Hamiltonian');
fig = gcf;
set(fig,'Units','Inches');
pos = get(fig,'Position');
fig.PaperPositionMode = 'auto';
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)+0.2])
print('Hamiltonian2Dsimu_boundexc','-dpdf')
%%
figure('color','w');
plot(tvec,height-1); grid on;
xlabel('time (s)');
ylabel('Integral of \alpha_1(x,y,t) in \Omega');
fig = gcf;
set(fig,'Units','Inches');
pos = get(fig,'Position');
fig.PaperPositionMode = 'auto';
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)+0.2])
print('Volumeoffluid2Dsimu_boundexc','-dpdf')



%% snapshots figure
E = blkdiag(eye(size(A)),1);
BB1 = [Bpartial_left_full(:,1)+Bpartial_up_full(:,1); zeros(size(A,1)-(Nex+1)*(Ney+1),1)]/2

BB2 = [inv(Massmatrix1)*Bpartial_left_full(:,1)+inv(Massmatrix1)*Bpartial_up_full(:,1); zeros(size(A,1)-(Nex+1)*(Ney+1),1)]/2
An = [A BB1; -BB2' 0];
Bn = [zeros(size(B(:,1))); 1];
Cn = [C zeros(length(C),1)];
dsys = dss(An,Bn,Cn,0,E);
%
inps = sin((0:0.005:5)*pi).^2 .*  ((0:0.005:5) <1);
%inputs = repmat(inps,2,1)*1
figure();plot(inps)
%%
tvec = 0:0.005:5;
y = lsim(dsys,inps,tvec, [x0;0]);
%%
for i = 1:size(y,1)
   figure(2)
   surf(reshape(Massmatrix1 \ y(i,1:N1)', Nex+1,Ney+1)')
   sum(y(i,1:N1))
   axis([0 20 0 20 0.5 1.5]) 
   title(strcat(num2str(tvec(i)), ' s'))
   pause(0.01)
end

%%
for i = 1:size(y,1)
   height(i) = sum(y(i,1:N1));
   discreteHamiltonian(i) = 0.5*(y(i,1:N1)*inv(Massmatrix1)*y(i,1:N1)' + y(i,N1+1:N1+N2)*inv(Massmatrix2)*y(i,N1+1:N1+N2)'+y(i,N1+N2+1:N1+N2+N2)*inv(Massmatrix3)*y(i,N1+N2+1:N1+N2+N2)');
end
figure('color','w'); plot(tvec,discreteHamiltonian); grid on;
xlabel('time (s)');
ylabel('Hamiltonian');
%%
fig = gcf;
set(fig,'Units','Inches');
pos = get(fig,'Position');
fig.PaperPositionMode = 'auto';
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)+0.2])
print('Hamiltonian2Dsimu_cornerexc','-dpdf')

%%
xl = linspace(0,1,Nex+1);
yl = linspace(0,1,Ney+1);
[XX YY] = meshgrid(xl, yl);
figure('color','w');
subplot(221);
in = 100;
surf(XX,YY,reshape(Massmatrix1 \ y(in,1:N1)', Nex+1,Ney+1)')
view(-70,80)
title(strcat(num2str(tvec(in)), ' s'))
subplot(222);
in = 200;
surf(XX,YY,reshape(Massmatrix1 \ y(in,1:N1)', Nex+1,Ney+1)')
view(-70,80)
title(strcat(num2str(tvec(in)), ' s'))
subplot(223);
in = 250;
surf(XX,YY,reshape(Massmatrix1 \ y(in,1:N1)', Nex+1,Ney+1)')
view(-70,80)
title(strcat(num2str(tvec(in)), ' s'))
subplot(224);
in = 300;
surf(XX,YY,reshape(Massmatrix1 \ y(in,1:N1)', Nex+1,Ney+1)')
view(-70,80)
title(strcat(num2str(tvec(in)), ' s'))

fig = gcf;
set(fig,'Units','Inches');
set(fig,'Position', [7.0208    0.8125    9.9479    9.3229]);
pos = get(fig,'Position');

fig.PaperPositionMode = 'auto';
set(fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print('simulation2D_corner_exc','-dpdf')

