close all,clear all,bdclose all
E = 71016000000; % Pa
S1 = 0.167*(2.54e-2)^2;  % m^2
rho = 2714.4716; % Kg/m^3
Iz1 = 3.881e-4*(2.54e-2)^4;  % m^2
nu = 0.3;
l0=10*2.54e-2;

l1=4.25*2.54e-2; % m

S2=0.063*(2.54e-2)^2;
Iz2=2.084e-5*(2.54e-2)^4;
l2=11*2.54e-2;% m

S3=S2;
Iz3=Iz2;
l3=10.65*2.54e-2;% m

theta1=0; 
R1=[cos(theta1) -sin(theta1);sin(theta1) cos(theta1)];
[r1,r2] = four_bars(theta1,l0,l1,l2,l3);
theta2=r1(2,1); 
R2=[cos(theta2) -sin(theta2);sin(theta2) cos(theta2)];
theta3=r1(3,1); 
R3=[cos(theta3) -sin(theta3);sin(theta3) cos(theta3)];
%l4=0.2667;% m
%theta4=-theta1-theta2-theta3;

path(path,'MFBD_dir');
M1=TwoPortBeamTxTyRz(rho,S1,l1,E,Iz1,0);
M2=TwoPortBeamTxTyRz(rho,S2,l2,E,Iz2,0);
M3=TwoPortBeamTxTyRz(rho,S3,l3,E,Iz3,0);

mt=0.0925*4.4482216152605/9.81;

my3bars;
sys=linmod('my3bars');
G=ss(sys.a,sys.b,sys.c,sys.d);

Gm1=invio(G,1);   % On suppose le CRANCK encastré à l'angle theta1
%figure(fig1);
%hold on
%bodemag(Gm1(1,1),logspace(2,3,500));
% Déformée modale:
[V,D]=eig(Gm1.a);
[puls,I]=sort(diag(D));
Vp=V(:,I(1));Vp=Vp/Vp(1);Vp=-Vp/norm(Vp);  %mode le plus lend;
sys.StateName  % Vérification de l'ordre des états
% beam 1:
Q1=[0;0;real(Gm1.a(5:8,:)*Vp)];
U1=beamshapeTyRz(Q1,l1);  % deformée de flexion
Ux1=[0:0.01:1]'*real(Gm1.a(10,:)*Vp); % déformée de traction
% beam 2:
Q2=[0;real(Gm1.c(2,:)*Vp);real(Gm1.a(15:18,:)*Vp)];
U2=beamshapeTyRz(Q2,l2);
Ux2=[0:0.01:1]'*real(Gm1.a(20,:)*Vp); 
% beam 2:
Q3=[0;real(Gm1.c(3,:)*Vp);real(Gm1.a(25:28,:)*Vp)];
U3=beamshapeTyRz(Q3,l3);
Ux3=[0:0.01:1]'*real(Gm1.a(30,:)*Vp); 
% Normalization
U=[U1;U2;U3;Ux1;Ux2;Ux3];
U1=U1'/max(abs(U))*0.05*l2;
Ux1=Ux1'/max(abs(U))*0.05*l2;
U2=U2'/max(abs(U))*0.05*l2;
Ux2=Ux2'/max(abs(U))*0.05*l2;
U3=U3'/max(abs(U))*0.05*l2;
Ux3=Ux3'/max(abs(U))*0.05*l2;

x1=[0:0.01:1]*l1;
x2=[0:0.01:1]*l2;
x3=[0:0.01:1]*l3;
XY1=R1*[x1;0*x1];
XY1d=R1*[x1+Ux1;U1];
%
XY2=R1*R2*[x2;0*x2];
XY2d=R1*R2*[x2+Ux2;U2];
XY2(1,:)=XY2(1,:)+XY1(1,end);
XY2(2,:)=XY2(2,:)+XY1(2,end);
XY2d(1,:)=XY2d(1,:)+XY1d(1,end);
XY2d(2,:)=XY2d(2,:)+XY1d(2,end);
%
XY3=R1*R2*R3*[x3;0*x3];
XY3d=R1*R2*R3*[x3+Ux2;U3];
XY3(1,:)=XY3(1,:)+XY2(1,end);
XY3(2,:)=XY3(2,:)+XY2(2,end);
XY3d(1,:)=XY3d(1,:)+XY2d(1,end);
XY3d(2,:)=XY3d(2,:)+XY2d(2,end);
%
XY=[XY1 XY2 XY3];
XYd=[XY1d XY2d XY3d];
figure
plot(XY(1,:),XY(2,:),'g--')
hold on
plot(XYd(1,:),XYd(2,:),'k-','LineWidth',2)

% Evolution des modes en fonction de l'angle theta1:
mode1=[];mode2=[];mode3=[];
[r1,r2] = four_bars(0,l0,l1,l2,l3);
%rm=r1(3,1);
for theta1=[0:0.02:1]*2*pi;
    R1=[cos(theta1) -sin(theta1);sin(theta1) cos(theta1)];
    [r1,r2] = four_bars(theta1,l0,l1,l2,l3);
    %if abs(r1(3,1)-rm)<abs(r2(3,1)-rm),
        theta2=r1(2,1); 
        theta3=r1(3,1);
    %    rm=r1(3,1);
    %else
    %    theta2=r2(2,1); 
    %    theta3=r2(3,1);
    %    rm=r2(3,1);
    %end
    R2=[cos(theta2) -sin(theta2);sin(theta2) cos(theta2)];
    theta3=r1(3,1); 
    R3=[cos(theta3) -sin(theta3);sin(theta3) cos(theta3)];
    sys=linmod('my3bars');
    G=ss(sys.a,sys.b,sys.c,sys.d);
    Gm1=invio(G,1);   
    [V,D]=eig(Gm1.a);
    [puls,I]=sort(diag(D));
    mode1=[mode1 abs(D(I(1),I(1)))];
    mode2=[mode2 abs(D(I(3),I(3)))];
    mode3=[mode3 abs(D(I(5),I(5)))];
end
figure,
subplot(1,2,1)
plot([0:0.02:1]*360,mode1,'b-')
hold on
plot([0:0.02:1]*360,mode2,'r-')
xlabel('Cranck angle (deg)');ylabel('Pulsation (rad/s');
legend('mode # 1','mode # 2');
subplot(1,2,2)
plot([0:0.02:1]*360,mode3)
xlabel('Cranck angle (deg)');ylabel('Pulsation (rad/s');
legend('mode # 3');

