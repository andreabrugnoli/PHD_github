% Test modes
clear all
close all
clc

m_panel = 43.2; 
lx=4.143;
ly=2.200;
t=0.04;
rho=m_panel/lx/ly/t;
E=70*10^9;
ni = 0.35;
xi=0.0;
P_xy=[0,ly/2];
C_xy=[lx,ly/2];
MFzTxTy = Kirchhoff_1elem_modal(lx, ly, t, rho, E, ni, xi, P_xy, C_xy, 1);
% MFzTxTy_inv = invio(MFzTxTy,[1:6]);
% 
% damp(MFzTxTy) 
% damp(MFzTxTy_inv) 
% 
P0=[0, 0];
C0=[lx, ly/2;
    0,ly/2];

Mod0 = Kirchhoff_1elem(lx, ly, t, rho, E, ni, P0, C0);
Mod1=invio(Mod0,[4:9]);
Mod1=Mod1(1:6,1:6);
damp(Mod1)
damp(MFzTxTy) 

printModes_1elem(Mod1,P_xy,lx,ly)

