% With superelement:
path(path,'../Correction Package HUB BEAM');

% Half length normelized beam
M=TwoPortBeamTyRz(1,1,0.5,1,1,0);
% Augmented model with pivot joint
Ma=[eye(4);0 0 0 -1]*M*[eye(4) [0;0;0;1]];
Mam1_5=invio(Ma,5);

% Angular configuration:
theta2=0;
T21=[cos(theta2) -sin(theta2) 0;sin(theta2) cos(theta2) 0;0 0 1];

% Model:
m=1*1*1; % mass for x-axis model
[a,b,c,d]=linmod('TwoBeamSuperElem');
G=ss(a,b,c,d);
damp(G)

% VALIDATION:
% Cas de 2 asservissements très raides (2 articulations sont bloquées):
W=damp(inv(G));
format short e
W(1:6)
%  ==> On retrouve les pulsations encastrée-libre d'une poutre unitaire
% Cas du second asservissement très raide (seconde articulation loquées):
W=damp(invio(G,2));
W(1:6)
%  ==> On retrouve les pulsation articulée-libre d'une poutre unitaire
%
theta2=pi/2;
T21=[cos(theta2) -sin(theta2) 0;sin(theta2) cos(theta2) 0;0 0 1];
[a,b,c,d]=linmod('TwoBeamSuperElem');
G=ss(a,b,c,d);
W=damp(G);
W(1:12)/4
bodemag(G)
% On y retrouve les pulsations articulée-libre du second beam (de longueur
% 0.5 donc les pulsations sont multipliée par 4).

% Si m (utilisé uniquement sur l'axe x) tends vers l'infini alors :
m=10^6;
[a,b,c,d]=linmod('TwoBeamSuperElem');
G=ss(a,b,c,d);
W=damp(G);
W(1:12)/4
% ==> on retrouve les pulsations articulé-articulé pour le beam 1 et
% articulié libre pour le beam 2:

